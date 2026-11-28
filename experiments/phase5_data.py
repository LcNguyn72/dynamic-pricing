"""Verified, store-sharded inputs for the Phase 5 V2 held-out benchmark."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
from collections.abc import Callable, Iterator, Sequence
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .phase1_finalize import _observed, _require_full_week_grid, _sha256, _split_counts, _weekly_dates


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE1_ROOT = PROJECT_ROOT / "experiments" / "artifacts" / "phase1_v2"
LOCK_PATH = PROJECT_ROOT / "experiments" / "phase5_settings_v2.json"
PROTOCOL_PATH = PROJECT_ROOT / "experiments" / "protocol_v2.json"
DATASET_IDS = {"uci": "uci_online_retail", "m5": "m5"}
SPLITS = ("train", "validation", "test")
LOOKBACK = 12
KEY_COLUMNS = ("sku", "week")
OBS_COLUMNS = ("qty", "price", "split", "observed_sales")
CALENDAR_COLUMNS = ("month", "week_of_year", "event_days", "snap_days")
WEEKLY_COLUMNS = (*KEY_COLUMNS, *OBS_COLUMNS, *CALENDAR_COLUMNS)
CACHE_SCHEMA_VERSION = 1


class _KeyDigest:
    """Use the same canonical CSV text as the Phase 1 frozen key file."""

    def __init__(self) -> None:
        self.digest = sha256()
        self.csv = csv.writer(self, lineterminator="\n")
        self.n = 0
        self.csv.writerow(("dataset", "sku", "week"))

    def write(self, content: str) -> int:
        self.digest.update(content.encode("utf-8"))
        return len(content)

    def add(self, row: tuple[str, str, str]) -> None:
        self.csv.writerow(row)
        self.n += 1

    def result(self) -> dict:
        return {"n": self.n, "sha256": self.digest.hexdigest()}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_file(path: Path, expected: str) -> None:
    if _sha256(path) != expected:
        raise ValueError(f"{path.name}: SHA256 differs from the frozen source")


def _frozen_key_shards(path: Path, dataset: str, shards: Sequence[str]) -> dict[str, dict]:
    canonical_id = DATASET_IDS[dataset]
    digests = {shard: _KeyDigest() for shard in shards}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != ["dataset", "sku", "week"]:
            raise ValueError(f"{path.name}: unexpected frozen key schema")
        previous = None
        for row in reader:
            if len(row) != 3 or row[0] != canonical_id:
                raise ValueError(f"{path.name}: unexpected dataset in frozen test keys")
            _, sku, week = row
            key = sku, week
            if previous is not None and key <= previous:
                raise ValueError(f"{path.name}: frozen test keys are unordered or duplicated")
            previous = key
            shard = sku.rpartition("@")[-1] if dataset == "m5" else "uci"
            if shard not in digests:
                raise ValueError(f"{path.name}: unexpected store in frozen test keys")
            digests[shard].add(tuple(row))
    return {name: digest.result() for name, digest in digests.items()}


def load_contract(root: str | Path = PHASE1_ROOT, lock_path: str | Path = LOCK_PATH) -> dict:
    """Validate once and hold the provenance and per-shard frozen key digests."""
    root, lock_path = Path(root).resolve(), Path(lock_path)
    phase1_path = root / "phase1_manifest.json"
    phase1, protocol, lock = map(_read_json, (phase1_path, PROTOCOL_PATH, lock_path))
    protocol_hash = _sha256(PROTOCOL_PATH)
    if (phase1.get("protocol_id") != protocol.get("protocol_id")
            or lock.get("protocol_id") != protocol.get("protocol_id")
            or phase1.get("protocol_sha256") != protocol_hash
            or lock.get("protocol_sha256") != protocol_hash
            or lock.get("selection_split") != "validation"
            or protocol["split"]["lookback_weeks"] != LOOKBACK):
        raise ValueError("Phase 1, Phase 5 settings and the V2 protocol do not match")
    stores = protocol["datasets"]["m5"]["store_ids"]
    reports = {row["store"]: row for row in phase1["m5"]["stores"]}
    if set(reports) != set(stores) or len(reports) != len(phase1["m5"]["stores"]):
        raise ValueError("Phase 1 manifest does not cover exactly the frozen M5 stores")
    key_shards = {}
    for dataset, canonical_id in DATASET_IDS.items():
        config = protocol["datasets"][canonical_id]
        locked = lock["frozen_test_keys"][dataset]
        audited = phase1[dataset]["test_keys"]
        if (locked["sha256"] != audited["sha256"]
                or locked["sha256"] != config["primary_test_key_sha256"]
                or locked["count"] != audited["count"]
                or locked["count"] != config["primary_test_key_count"]):
            raise ValueError(f"{dataset}: disagreement in frozen test key provenance")
        path = root / f"test_keys_{dataset}.csv"
        _verify_file(path, locked["sha256"])
        key_shards[dataset] = _frozen_key_shards(path, dataset, stores if dataset == "m5" else ("uci",))
        if sum(entry["n"] for entry in key_shards[dataset].values()) != locked["count"]:
            raise ValueError(f"{dataset}: mismatch in number of frozen test keys")
    for store, info in key_shards["m5"].items():
        if info["n"] != reports[store]["observed_test_targets"]:
            raise ValueError(f"{store}: frozen test keys disagree with the store manifest")
    return {
        "root": root, "phase1": phase1, "protocol": protocol, "lock": lock,
        "phase1_sha256": _sha256(phase1_path), "protocol_sha256": protocol_hash,
        "lock_sha256": _sha256(lock_path), "test_key_shards": key_shards,
        "m5_stores": reports,
    }


def _root(path: str | Path, contract: dict) -> Path:
    root = Path(path).resolve()
    if root != contract["root"]:
        raise ValueError("Phase 1 root differs from the verified contract")
    return root


def _dataset_shards(dataset: str, contract: dict, shards: Sequence[str] | None) -> Sequence[str]:
    if dataset not in DATASET_IDS:
        raise ValueError(f"Unknown Phase 5 dataset: {dataset}")
    required = tuple(contract["test_key_shards"][dataset])
    selected = required if shards is None else tuple(shards)
    if not selected or len(selected) != len(set(selected)) or not set(selected).issubset(required):
        raise ValueError(f"{dataset}: unexpected or duplicate requested shards")
    return selected


def _manifest(root: Path, dataset: str, shard: str, contract: dict) -> dict:
    if dataset == "uci":
        path = root / "uci" / "panel" / "manifest.json"
        expected = contract["phase1"]["uci"]["panel_manifest_sha256"]
    else:
        path = root / "m5" / f"manifest_{shard}.json"
        expected = contract["m5_stores"][shard]["manifest_sha256"]
    _verify_file(path, expected)
    return _read_json(path)


def load_scalers(root: str | Path, dataset: str, contract: dict,
                 *, shards: Sequence[str] | None = None) -> dict:
    """Read verified train-only scalers and the global SKU vocabulary without panels."""
    root = _root(root, contract)
    combined = {}
    for shard in _dataset_shards(dataset, contract, shards):
        current = _manifest(root, dataset, shard, contract)["scalers"]
        if combined.keys() & current.keys():
            raise ValueError(f"{dataset}: scaler SKU occurs in multiple shards")
        combined.update(current)
    return dict(sorted(combined.items()))


def _uci_weekly(root: Path, report: dict, manifest: dict) -> pd.DataFrame:
    # The upstream aggregate is hashed. Reconstruct the causal fill from it
    # rather than trusting an unsigned panel CSV or recomputing scalers.
    path = root / "uci" / "weekly.csv"
    _verify_file(path, report["weekly_sha256"])
    if manifest["weekly_source_sha256"] != report["weekly_sha256"]:
        raise ValueError("UCI panel and upstream weekly hashes differ")
    source = pd.read_csv(path, usecols=[*KEY_COLUMNS, "qty", "qty_all", "price"],
                         dtype={"sku": str}, parse_dates=["week"])
    skus = sorted(manifest["scalers"])
    heldout = source["week"].gt(pd.Timestamp(manifest["split_weeks"]["train_end"]))
    if not source.loc[heldout, "qty"].eq(source.loc[heldout, "qty_all"]).all():
        raise ValueError("UCI held-out weekly quantities differ from valid transactions")
    source = source.loc[source["sku"].isin(skus), [*KEY_COLUMNS, "qty", "price"]]
    if set(source["sku"]) != set(skus):
        raise ValueError("UCI observed weekly data do not include every eligible SKU")
    dates = pd.date_range(manifest["first_week"], manifest["last_week"], freq="7D")
    grid = pd.MultiIndex.from_product([skus, dates], names=KEY_COLUMNS).to_frame(index=False)
    if not source["week"].isin(dates).all():
        raise ValueError("UCI source dates fall outside the frozen panel calendar")
    panel = grid.merge(source, on=list(KEY_COLUMNS), how="left", validate="one_to_one", indicator=True)
    panel["observed_sales"] = panel.pop("_merge").eq("both")
    panel["qty"] = panel["qty"].fillna(0.0).astype(float)
    panel["price"] = panel.groupby("sku", sort=False)["price"].ffill()
    panel["split"] = np.select(
        [panel["week"].le(pd.Timestamp(manifest["split_weeks"]["train_end"])),
         panel["week"].lt(pd.Timestamp(manifest["split_weeks"]["test_start"]))],
        ["train", "validation"], default="test",
    )
    panel["month"] = panel["week"].dt.month.astype("uint8")
    panel["week_of_year"] = panel["week"].dt.isocalendar().week.astype("uint8")
    return panel


def _m5_weekly(root: Path, shard: str, report: dict, manifest: dict) -> pd.DataFrame:
    path = root / "m5" / f"weekly_{shard}.csv"
    _verify_file(path, report["weekly_sha256"])
    if manifest["weekly_sha256"] != report["weekly_sha256"] or manifest["store_id"] != shard:
        raise ValueError(f"{shard}: weekly panel provenance differs from its manifest")
    panel = pd.read_csv(path, usecols=[*WEEKLY_COLUMNS, "priced_days"], parse_dates=["week"],
                        dtype={"sku": str, **{name: "uint8" for name in (*CALENDAR_COLUMNS, "priced_days")}})
    if not _observed(panel).eq(panel.pop("priced_days").eq(7)).all():
        raise ValueError(f"{shard}: observed labels differ from the fully priced weeks")
    return panel


def _weekly_shard(root: Path, dataset: str, shard: str, contract: dict, manifest: dict) -> pd.DataFrame:
    config = contract["protocol"]["datasets"][DATASET_IDS[dataset]]
    if dataset == "uci":
        panel = _uci_weekly(root, contract["phase1"]["uci"], manifest)
        series = config["currently_prepared_skus"]
    else:
        report = contract["m5_stores"][shard]
        panel = _m5_weekly(root, shard, report, manifest)
        series = report["eligible_series"]
    panel["week"] = _weekly_dates(panel)
    panel["observed_sales"] = _observed(panel)
    _require_full_week_grid(panel, series=series, weeks=config["calendar_weeks"])
    _split_counts(panel, config)
    observed = panel.loc[panel["observed_sales"]]
    if (set(panel["sku"]) != set(manifest["scalers"]) or not np.isfinite(panel["qty"]).all()
            or not panel["qty"].ge(0).all() or not np.isfinite(observed["price"]).all()
            or not observed["price"].gt(0).all()):
        raise ValueError(f"{shard}: invalid weekly SKU, quantity or observed price")
    return panel.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def _cache_identity(contract: dict, dataset: str, shard: str) -> dict:
    report = contract["phase1"]["uci"] if dataset == "uci" else contract["m5_stores"][shard]
    return {"schema_version": CACHE_SCHEMA_VERSION,
            "protocol_sha256": contract["protocol_sha256"],
            "weekly_sha256": report["weekly_sha256"],
            "manifest_sha256": report.get("panel_manifest_sha256", report.get("manifest_sha256")),
            "frozen_test_keys_sha256": contract["test_key_shards"][dataset][shard]["sha256"]}


def _encode_cache(frame: pd.DataFrame, skus: Sequence[str]) -> dict[str, np.ndarray]:
    arrays = {}
    for name in frame:
        if name in ("sku", "split"):
            vocabulary = skus if name == "sku" else SPLITS
            values = pd.Categorical(frame[name], categories=vocabulary).codes
            if np.any(values < 0):
                raise ValueError(f"Unknown {name} in Phase 5 cache")
        elif name == "week":
            values = frame[name].to_numpy(dtype="datetime64[D]").astype(np.int32)
        else:
            values = frame[name].to_numpy()
            if values.dtype.kind not in "biuf":
                raise ValueError(f"Unexpected cache dtype for {name}: {values.dtype}")
        arrays[name] = values
    return arrays


def _decode_cache(path: Path, skus: Sequence[str]) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["sku"] = np.asarray(skus, dtype=object)[arrays["sku"]]
    arrays["split"] = np.asarray(SPLITS, dtype=object)[arrays["split"]]
    arrays["week"] = arrays["week"].astype("datetime64[D]").astype("datetime64[us]")
    return pd.DataFrame(arrays, copy=False)


def _cache_frame(cache_root: str | Path | None, dataset: str, shard: str, kind: str, identity: dict,
                 skus: Sequence[str], build: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Resume verified reads from a columnar cache without executable serialization."""
    if cache_root is None:
        return build()
    directory = Path(cache_root) / dataset
    directory.mkdir(parents=True, exist_ok=True)
    identity_hash = sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    name = f"{shard}_{kind}_{identity_hash}"
    path, marker = directory / f"{name}.npz", directory / f"{name}.json"
    if marker.is_file() and path.is_file():
        try:
            record = _read_json(marker)
        except (OSError, ValueError):
            record = {}
        if record.get("identity") == identity and record.get("sha256") == _sha256(path):
            return _decode_cache(path, skus)
    frame = build()
    with TemporaryDirectory(prefix=".phase5-shard-", dir=directory) as tmp:
        data_tmp, marker_tmp = Path(tmp) / "columns.npz", Path(tmp) / "manifest.json"
        np.savez(data_tmp, **_encode_cache(frame, skus))
        record = {"identity": identity, "n": len(frame), "sha256": _sha256(data_tmp)}
        marker_tmp.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        data_tmp.replace(path)
        marker_tmp.replace(marker)
    return frame


def _cached_weekly(root: Path, dataset: str, shard: str, contract: dict, manifest: dict,
                   cache_root: str | Path | None) -> pd.DataFrame:
    identity = _cache_identity(contract, dataset, shard)
    return _cache_frame(cache_root, dataset, shard, "weekly", identity, sorted(manifest["scalers"]),
                        lambda: _weekly_shard(root, dataset, shard, contract, manifest))


def iter_weekly(root: str | Path, dataset: str, contract: dict,
                *, shards: Sequence[str] | None = None,
                cache_root: str | Path | None = None) -> Iterator[tuple[str, pd.DataFrame, dict]]:
    """Yield one verified full-calendar shard at a time in SKU/week order."""
    root = _root(root, contract)
    for shard in _dataset_shards(dataset, contract, shards):
        manifest = _manifest(root, dataset, shard, contract)
        weekly = _cached_weekly(root, dataset, shard, contract, manifest, cache_root)
        yield shard, weekly, manifest["scalers"]


def windows_from_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    """Build compact V2 12-week windows on the complete past calendar."""
    needed = {*KEY_COLUMNS, *OBS_COLUMNS}
    if not needed.issubset(weekly.columns):
        raise ValueError(f"Weekly panel lacks {sorted(needed - set(weekly.columns))}")
    ordered = weekly.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    dates = _weekly_dates(ordered)
    if ordered["sku"].isna().any() or not set(ordered["split"]).issubset(SPLITS):
        raise ValueError("Invalid SKU or split in weekly panel")
    by_sku = ordered.groupby("sku", sort=False)
    gap = dates.groupby(ordered["sku"], sort=False).diff().dropna()
    if not gap.eq(pd.Timedelta(days=7)).all():
        raise ValueError("Cannot build rolling windows across duplicate or missing weeks")

    observed = _observed(ordered)
    frame = {name: ordered[name].to_numpy() for name in ("sku", "qty", "price", "split")}
    frame.update(week=dates.to_numpy(), observed_sales=observed.to_numpy())
    frame["month"] = dates.dt.month.to_numpy(dtype=np.uint8)
    frame["week_of_year"] = dates.dt.isocalendar().week.to_numpy(dtype=np.uint8)
    extra = [name for name in ("event_days", "snap_days") if name in ordered.columns]
    for name in extra:
        if not ordered[name].between(0, 7).all():
            raise ValueError(f"Invalid calendar counts in {name}")
        frame[name] = ordered[name].to_numpy(dtype=np.uint8)
    eligible = np.isfinite(frame["qty"]) & np.isfinite(frame["price"]) & (frame["price"] > 0)
    for lag in range(1, LOOKBACK + 1):
        quantity = by_sku["qty"].shift(lag).to_numpy(dtype=np.float32)
        price = by_sku["price"].shift(lag).to_numpy(dtype=np.float32)
        eligible &= np.isfinite(quantity) & np.isfinite(price) & (price > 0)
        frame[f"qty_lag_{lag}"] = quantity
        frame[f"price_lag_{lag}"] = price
        for name in extra:
            frame[f"{name}_lag_{lag}"] = by_sku[name].shift(lag, fill_value=0).to_numpy(dtype=np.uint8)
    return pd.DataFrame({name: values[eligible] for name, values in frame.items()})


def verify_shard_test_keys(root: str | Path, dataset: str, contract: dict,
                           shard: str, windows: pd.DataFrame) -> dict:
    """Reject even a same-count replacement of one frozen test key."""
    _root(root, contract)
    expected = contract["test_key_shards"][dataset][shard]
    target = windows.loc[windows["split"].eq("test") & _observed(windows), list(KEY_COLUMNS)].copy()
    target["week"] = _weekly_dates(target).dt.strftime("%Y-%m-%d")
    if target["sku"].isna().any() or target.duplicated().any():
        raise ValueError(f"{shard}: null or duplicate frozen test keys")
    digest = _KeyDigest()
    for sku, week in target.sort_values(list(KEY_COLUMNS)).itertuples(index=False, name=None):
        digest.add((DATASET_IDS[dataset], sku, week))
    actual = digest.result()
    if actual != expected:
        raise ValueError(f"{shard}: windows do not match frozen test keys ({actual['n']} vs {expected['n']})")
    return actual


def iter_windows(root: str | Path, dataset: str, contract: dict,
                 *, shards: Sequence[str] | None = None,
                 cache_root: str | Path | None = None) -> Iterator[tuple[str, pd.DataFrame, dict]]:
    """Yield compact windows after verifying the shard's exact test key digest."""
    root = _root(root, contract)
    for shard in _dataset_shards(dataset, contract, shards):
        manifest = _manifest(root, dataset, shard, contract)
        scalers = manifest["scalers"]
        windows = _cache_frame(cache_root, dataset, shard, "windows", _cache_identity(contract, dataset, shard),
                               sorted(scalers), lambda: windows_from_weekly(
                                   _cached_weekly(root, dataset, shard, contract, manifest, cache_root)))
        verify_shard_test_keys(root, dataset, contract, shard, windows)
        if dataset == "m5":
            zeros = (windows["split"].eq("test") & _observed(windows) & windows["qty"].eq(0)).sum()
            if int(zeros) != contract["m5_stores"][shard]["zero_sales_test_targets"]:
                raise ValueError(f"{shard}: observed zero-sale test labels changed")
        yield shard, windows, scalers
