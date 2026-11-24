"""Fit and reload Song demand checkpoints from the verified Phase 1 V2 panels."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import json
import math

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .data import make_windows
from .phase1_finalize import DEFAULT_OUTPUT as PHASE1_ROOT, _observed
from .song_forecast import (
    CURRENT_FEATURES, HISTORY_FEATURES, LOOKBACK, SongDemandForecaster,
    SongTransformer, _scalers, _sha256, feature_arrays,
)

PROTOCOL = Path(__file__).with_name("protocol_v2.json")
OUTPUT_ROOT = PHASE1_ROOT.with_name("phase2_v2")
SPLITS = ("train", "validation")
M5_COLUMNS = ("sku", "week", "split", "qty", "price", "qty_z", "price_z",
              "event_days", "snap_days", "observed_sales")
INPUT_COLUMNS = ["sku", "week", "price", "event_days", "snap_days", "promotion_flag",
                 *(f"{field}_lag_{lag}" for lag in range(1, LOOKBACK + 1)
                   for field in ("qty", "price", "event_days", "snap_days", "promotion_flag"))]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _verify(path: Path, expected: str) -> None:
    if _sha256(path) != expected:
        raise ValueError(f"{path.name}: SHA256 differs from the frozen source")


def m5_store_windows(root: Path, report: dict, pricing_sku: str,
                     max_skus: int) -> tuple[pd.DataFrame, dict, dict]:
    """Read one store and rank by observed training units, retaining its pricing SKU."""
    store = report["store"]
    manifest_path, weekly_path = root / f"manifest_{store}.json", root / f"weekly_{store}.csv"
    _verify(manifest_path, report["manifest_sha256"])
    _verify(weekly_path, report["weekly_sha256"])
    store_scalers = _read_json(manifest_path)["scalers"]
    weekly = pd.read_csv(weekly_path, usecols=M5_COLUMNS, dtype={"sku": str}, parse_dates=["week"])
    weekly = weekly.loc[weekly["split"].isin(SPLITS)]
    train = weekly.loc[weekly["split"].eq("train") & _observed(weekly)]
    rank = (train.groupby("sku", sort=True)["qty"].sum()
            .sort_values(ascending=False, kind="stable"))
    if pricing_sku not in rank.index or max_skus < 1:
        raise ValueError(f"{store}: invalid trial scope or missing Phase 1 pricing SKU")
    selected = [pricing_sku, *rank.drop(pricing_sku).head(max_skus - 1).index]
    scalers = {sku: store_scalers[sku] for sku in selected}
    # Keep unobserved weeks as history. Filter labels only after building lags.
    windows = make_windows(weekly.loc[weekly["sku"].isin(selected)], lookback=LOOKBACK)
    windows = windows.loc[_observed(windows)]
    source = {"store": store, "weekly_sha256": report["weekly_sha256"],
              "manifest_sha256": report["manifest_sha256"], "pricing_sku": pricing_sku}
    return windows, scalers, source


def _uci_windows(root: Path, report: dict) -> tuple[pd.DataFrame, dict, dict]:
    manifest_path, windows_path = root / "manifest.json", root / "windows.csv"
    _verify(manifest_path, report["panel_manifest_sha256"])
    _verify(windows_path, report["windows_sha256"])
    columns = {*INPUT_COLUMNS, "qty", "split", "observed_sales"}
    windows = pd.read_csv(windows_path, usecols=columns.__contains__,
                          dtype={"sku": str}, parse_dates=["week"])
    windows = windows.loc[windows["split"].isin(SPLITS) & _observed(windows)]
    return windows, _read_json(manifest_path)["scalers"], {
        "panel_manifest_sha256": report["panel_manifest_sha256"],
        "windows_sha256": report["windows_sha256"],
    }


def _sources(root: Path, phase1: dict, protocol: dict, dataset: str, max_skus: int):
    if dataset == "uci":
        yield _uci_windows(root / "uci" / "panel", phase1["uci"])
        return
    report = phase1["m5"]
    reports = {row["store"]: row for row in report["stores"]}
    cohort_path = root / "m5_pricing_cohort.csv"
    _verify(cohort_path, report["pricing_cohort_sha256"])
    cohort = pd.read_csv(cohort_path, dtype=str).set_index("store")["sku"]
    stores = protocol["datasets"]["m5"]["store_ids"]
    if len(cohort) != len(stores) or set(cohort.index) != set(stores) or set(reports) != set(stores):
        raise ValueError("Phase 1 M5 manifests or pricing cohort do not cover the protocol stores")
    for store in stores:
        yield m5_store_windows(root / "m5", reports[store], cohort[store], max_skus)


def _example(windows: pd.DataFrame, pricing_sku: str | None) -> dict:
    train = windows.loc[windows["split"].eq("train")]
    validation = windows.loc[windows["split"].eq("validation")]
    support = train.groupby("sku")["price"].agg(["min", "max"])
    eligible = support.loc[support["max"].gt(support["min"]) & support.index.isin(validation["sku"])]
    sku = pricing_sku
    if sku is None and not eligible.empty:
        sku = str(eligible.index[0])
    if sku not in eligible.index:
        raise ValueError("An example needs distinct training prices and a validation window")
    low, high = map(float, eligible.loc[sku, ["min", "max"]])
    fields = [name for name in INPUT_COLUMNS if name in validation.columns]
    row = validation.loc[validation["sku"].eq(sku), fields].sort_values("week").iloc[0]
    window = {name: float(row[name]) for name in fields if name not in {"sku", "week"}}
    return {"window": {**window, "sku": sku, "week": row["week"].date().isoformat()},
            "candidate_prices": [low, (low + high) / 2, high]}


def _cache_data(cache: Path, sources) -> tuple[list[dict], dict, dict]:
    """Encode one source at a time, then retain only memory-mapped feature blocks."""
    shards, all_scalers, example = [], {}, None
    for index, (windows, scalers, source) in enumerate(sources):
        if set(scalers) & all_scalers.keys():
            raise ValueError("The same series appears in multiple input shards")
        all_scalers.update(scalers)
        if example is None:
            example = _example(windows, source.get("pricing_sku"))
        counts, series_counts = {}, {}
        for split in SPLITS:
            rows = windows.loc[windows["split"].eq(split)]
            if rows.empty:
                raise ValueError(f"{source.get('store', 'UCI')}: no usable {split} windows")
            history, current = feature_arrays(rows, scalers)
            mean, std, _, _ = _scalers(rows, scalers)
            actual = rows["qty"].to_numpy(dtype=np.float32)
            target = (actual - mean) / std
            arrays = {"history": history, "current": current, "target": target}
            if split == "validation":
                arrays.update(actual=actual, mean=mean, std=std)
            for name, array in arrays.items():
                np.save(cache / f"{index}_{split}_{name}.npy", array, allow_pickle=False)
            counts[split] = len(rows)
            series_counts[split] = int(rows["sku"].nunique())
        print(f"[Phase 2] {source.get('store', 'UCI')}: {len(scalers)} series with scalers, "
              f"{counts['train']} train, {counts['validation']} validation", flush=True)
        shards.append({**source, "series_with_scalers": len(scalers),
                       "series_with_windows": series_counts, "windows": counts})
    if example is None:
        raise ValueError("No input shards")
    return shards, all_scalers, example


def _batches(cache: Path, shards: list[dict], split: str, batch_size: int,
             rng: np.random.Generator | None = None):
    names = ("history", "current", "target")
    if split == "validation":
        names += ("actual", "mean", "std")
    order = range(len(shards)) if rng is None else rng.permutation(len(shards))
    for shard in order:
        arrays = [np.load(cache / f"{shard}_{split}_{name}.npy", mmap_mode="r",
                          allow_pickle=False) for name in names]
        size = shards[shard]["windows"][split]
        rows = np.arange(size) if rng is None else rng.permutation(size)
        for start in range(0, size, batch_size):
            indices = rows[start:start + batch_size]
            yield tuple(torch.from_numpy(array[indices]) for array in arrays)


def _fit(cache: Path, shards: list[dict], seed: int, epochs: int, batch_size: int,
         device: torch.device) -> tuple[dict, list[dict]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SongTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_train = sum(shard["windows"]["train"] for shard in shards)
    n_val = sum(shard["windows"]["validation"] for shard in shards)
    best_rmse, best_epoch, best_state, log = math.inf, 0, {}, []
    for epoch in range(1, epochs + 1):
        started = perf_counter()
        model.train()
        train_sse = 0.0
        for history, current, target in _batches(cache, shards, "train", batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            z = model(history.to(device), current.to(device))
            loss = F.mse_loss(z, target.to(device))
            loss.backward()
            optimizer.step()
            train_sse += float(loss.detach()) * len(target)
        model.eval()
        val_sse, val_z_sse = 0.0, 0.0
        with torch.no_grad():
            for history, current, target, actual, mean, std in _batches(
                    cache, shards, "validation", batch_size):
                z = model(history.to(device), current.to(device)).cpu().double()
                qty = (z * std + mean).clamp_min(0)
                val_sse += float((qty - actual).square().sum())
                val_z_sse += float((z - target).square().sum())
        rmse = math.sqrt(val_sse / n_val)
        if not all(math.isfinite(value) for value in (train_sse, val_z_sse, rmse)):
            raise ValueError(f"Non-finite training or validation error in epoch {epoch}")
        log.append({"epoch": epoch, "train_mse_z": train_sse / n_train,
                    "validation_mse_z": val_z_sse / n_val, "validation_micro_rmse": rmse,
                    "seconds": round(perf_counter() - started, 2)})
        print(f"[Phase 2] epoch {epoch}: validation RMSE {rmse:.4f} "
              f"({log[-1]['seconds']} s)", flush=True)
        if rmse < best_rmse:
            best_rmse, best_epoch = rmse, epoch
            best_state = {name: tensor.detach().cpu().clone()
                          for name, tensor in model.state_dict().items()}
        if epoch - best_epoch >= 8:
            break
    return {"state_dict": best_state, "best_epoch": best_epoch}, log


def predict(run_dir: Path, candidate_prices: list[float] | None = None) -> dict:
    """Run the saved validation example with new prices and the original history."""
    example = _read_json(run_dir / "example.json")
    saved = torch.load(run_dir / "song_transformer.pt", map_location="cpu", weights_only=True)
    if (saved["history_features"] != HISTORY_FEATURES or saved["current_features"] != CURRENT_FEATURES
            or saved["lookback"] != LOOKBACK):
        raise ValueError("Checkpoint feature schema is incompatible")
    model = SongTransformer()
    model.load_state_dict(saved["state_dict"])
    forecaster = SongDemandForecaster(model, saved["scalers"])
    row = example["window"]
    prices = example["candidate_prices"] if candidate_prices is None else candidate_prices
    results = [{"price": price, "predicted_qty": forecaster.predict_q(row, price),
                "dq_dp": forecaster.price_gradient(row, price)} for price in prices]
    if not all(math.isfinite(value) for result in results for value in result.values()):
        raise ValueError("Checkpoint produced a non-finite prediction or price derivative")
    return {"sku": row["sku"], "week": row["week"], "prices": results}


def train(dataset: str, phase1_root: Path, output: Path, *, seed: int, epochs: int,
          batch_size: int, m5_skus_per_store: int) -> dict:
    if min(epochs, batch_size, m5_skus_per_store) < 1:
        raise ValueError("Epochs, batch size and M5 SKUs per store must be positive")
    if output.exists():
        raise FileExistsError(f"Run directory already exists: {output}")
    phase1_path = phase1_root / "phase1_manifest.json"
    phase1, protocol = _read_json(phase1_path), _read_json(PROTOCOL)
    if (phase1["protocol_id"] != protocol["protocol_id"]
            or phase1["protocol_sha256"] != _sha256(PROTOCOL)):
        raise ValueError("Phase 1 panel was not verified against this V2 protocol")
    if protocol["split"]["lookback_weeks"] != LOOKBACK:
        raise ValueError("Model lookback differs from the frozen protocol")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_id = "uci_online_retail" if dataset == "uci" else "m5"
    config = protocol["datasets"][dataset_id]
    output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with TemporaryDirectory(prefix=".phase2-", dir=output.parent) as tmp:
        work = Path(tmp)
        cache, run = work / "cache", work / "run"
        cache.mkdir()
        run.mkdir()
        sources = _sources(phase1_root, phase1, protocol, dataset, m5_skus_per_store)
        shards, scalers, example = _cache_data(cache, sources)
        extraction_seconds = round(perf_counter() - started, 2)
        best, log = _fit(cache, shards, seed, epochs, batch_size, device)
        metadata = {
            "dataset": dataset_id, "protocol_id": protocol["protocol_id"],
            "phase1_manifest_sha256": _sha256(phase1_path),
            "protocol_sha256": phase1["protocol_sha256"],
            "split_weeks": {name: config[name]
                            for name in ("train_end", "validation_start", "validation_end", "test_start")},
            "seed": seed, "fit_scope": "train_only", "selection_metric": "validation_micro_rmse",
            "lookback": LOOKBACK, "history_features": HISTORY_FEATURES, "current_features": CURRENT_FEATURES,
            "architecture": {"layers": 4, "heads": 8, "width": 128},
            "optimizer": "Adam", "learning_rate": 1e-3, "loss": "MSE on train-scaled quantity",
        }
        torch.save({**best, **metadata, "scalers": scalers}, run / "song_transformer.pt")
        _write_json(run / "example.json", example)
        check = predict(run)
        _write_json(run / "prediction_check.json", check)
        pd.DataFrame(log).to_csv(run / "training.csv", index=False)
        result = {
            **metadata, "best_epoch": best["best_epoch"], "device": str(device),
            "scope": ("all usable UCI windows from the Phase 1 eligible cohort" if dataset == "uci" else
                      f"trial: up to {m5_skus_per_store} train-ranked series per store, including Phase 1 pricing SKU"),
            "shards": shards, "series_with_scalers": len(scalers),
            "series_with_windows": {split: sum(shard["series_with_windows"][split] for shard in shards)
                                    for split in SPLITS},
            "windows": {split: sum(shard["windows"][split] for shard in shards) for split in SPLITS},
            "checkpoint_sha256": _sha256(run / "song_transformer.pt"),
            "epoch_cap": epochs, "batch_size": batch_size, "epochs_run": len(log),
            "validation_micro_rmse": log[best["best_epoch"] - 1]["validation_micro_rmse"],
            "feature_preparation_seconds": extraction_seconds, "seconds": round(perf_counter() - started, 2),
        }
        _write_json(run / "run.json", result)
        run.rename(output.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    fit = subcommands.add_parser("train")
    fit.add_argument("--dataset", choices=("uci", "m5"), required=True)
    fit.add_argument("--phase1-root", type=Path, default=PHASE1_ROOT)
    fit.add_argument("--output-dir", type=Path)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--epochs", type=int, default=3)
    fit.add_argument("--batch-size", type=int, default=64)
    fit.add_argument("--m5-skus-per-store", type=int, default=12)
    demo = subcommands.add_parser("predict")
    demo.add_argument("--run-dir", type=Path, required=True)
    demo.add_argument("--prices", nargs="+", type=float)
    args = parser.parse_args()
    if args.command == "predict":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        print(json.dumps(predict(args.run_dir, args.prices), indent=2))
    else:
        output = args.output_dir or OUTPUT_ROOT / f"{args.dataset}_seed{args.seed}"
        train(args.dataset, args.phase1_root, output, seed=args.seed, epochs=args.epochs,
              batch_size=args.batch_size, m5_skus_per_store=args.m5_skus_per_store)
        print(f"[Phase 2] saved {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
