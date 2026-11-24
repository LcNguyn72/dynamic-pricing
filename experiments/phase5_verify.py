"""Independently verify V2 test predictions and simulated decisions before reporting."""

from __future__ import annotations

import argparse
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .phase1_finalize import DEFAULT_OUTPUT as PHASE1_ROOT, _observed
from .phase2_song import _read_json, _write_json

OUTPUT_ROOT = PHASE1_ROOT.with_name("phase5_v2")
LOCK_PATH = Path(__file__).with_name("phase5_settings_v2.json")
KEYS = ["dataset", "sku", "week"]
DETERMINISTIC = {"naive_last", "arima_sparse_fallback"}
DATASET_IDS = {"uci": "uci_online_retail", "m5": "m5"}
SUMS = ["n", "qty", "qty_squared", "absolute_error", "squared_error", "mape_n", "ape"]


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _check_sha(path: Path, wanted: str) -> None:
    if _sha(path) != wanted:
        raise ValueError(f"{path}: SHA256 differs from its manifest")


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"dataset": str, "sku": str, "week": str})


def align_predictions(predictions: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Require the identical unique forecast keys, quantities and observed prices."""
    left = predictions.sort_values(KEYS).reset_index(drop=True)
    right = reference.sort_values(KEYS).reset_index(drop=True)
    if left.duplicated(KEYS).any() or right.duplicated(KEYS).any() or not left[KEYS].equals(right[KEYS]):
        raise ValueError("Prediction keys differ from the frozen test keys")
    for column, tolerance in (("qty", 1e-8), ("price", 1e-7)):
        if not np.allclose(left[column], right[column], rtol=tolerance, atol=1e-9):
            raise ValueError(f"Prediction {column} differs from the frozen test labels")
    values = left["predicted_qty"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Predicted demand must be finite and nonnegative")
    return right.assign(predicted_qty=values)


def sufficient(rows: pd.DataFrame) -> pd.DataFrame:
    actual = rows["qty"].to_numpy(dtype=np.float64)
    error = np.abs(rows["predicted_qty"].to_numpy(dtype=np.float64) - actual)
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.divide(error, actual, out=np.zeros_like(error), where=(actual > 0))
    return pd.DataFrame({"n": np.ones(len(rows), dtype=np.int64),
                         "qty": actual, "qty_squared": actual ** 2,
                         "absolute_error": error, "squared_error": error ** 2,
                         "mape_n": (actual > 0).astype(np.int64), "ape": ape}, index=rows.index)


def _divide(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def summarize(row, *, train_mean: float, train_std: float) -> dict:
    n, mape_n = int(row["n"]), int(row["mape_n"])
    rmse = float(np.sqrt(row["squared_error"] / n)) if n else None
    sst = float(row["qty_squared"] - row["qty"] ** 2 / n) if n else 0.0
    relative_r2 = _divide(float(row["squared_error"]), sst)
    return {
        "n": n, "rmse": rmse, "mae": _divide(float(row["absolute_error"]), n),
        "r2": 1.0 - relative_r2 if relative_r2 is not None else None,
        "wape_pct": _divide(100 * float(row["absolute_error"]), float(row["qty"])),
        "mape_pct_positive_targets": _divide(100 * float(row["ape"]), mape_n),
        "mape_positive_target_count": mape_n,
        "rmse_train_mean_relative": _divide(rmse, train_mean) if n else None,
        "rmse_train_z": _divide(rmse, train_std) if n else None,
    }


def matrix(lock: dict, dataset: str):
    """Run one complete seed across algorithms before filling the next seed."""
    for index, seed in enumerate(lock["phase5_seeds"]):
        for model in lock["forecast"][dataset]["settings_by_model"]:
            if index == 0 or model not in DETERMINISTIC:
                yield model, seed


def _identity(contract: dict) -> dict:
    return {"phase1_sha256": contract["phase1_sha256"], "lock_sha256": contract["lock_sha256"]}


def prepare_reference(root: Path, output: Path, dataset: str, contract: dict) -> dict:
    """Cache independently sourced test labels and train-only price cut points."""
    from .phase5_data import iter_weekly

    identity = _identity(contract)
    frozen = contract["lock"]["frozen_test_keys"][dataset]
    output.parent.mkdir(parents=True, exist_ok=True)
    marker = output / "reference.json"
    if marker.is_file():
        cached = _read_json(marker)
        if cached["identity"] != identity or cached["frozen_test_keys_sha256"] != frozen["sha256"]:
            raise ValueError(f"{output}: reference belongs to a different configuration")
        for metadata in cached["shards"].values():
            _check_sha(output / metadata["file"], metadata["sha256"])
        return cached
    key_file = root / f"test_keys_{dataset}.csv"
    _check_sha(key_file, frozen["sha256"])
    keys = _csv(key_file)
    if len(keys) != frozen["count"] or keys.duplicated(KEYS).any():
        raise ValueError("Frozen test key count or uniqueness does not match")
    n_train = 0
    sum_qty = sum_squared_qty = 0.0
    price_parts = []
    shards = {}
    with TemporaryDirectory(dir=output.parent, prefix=".phase5-reference-") as tmp:
        work = Path(tmp) / "reference"
        work.mkdir()
        for shard, weekly, _ in iter_weekly(root, dataset, contract):
            observed = _observed(weekly)
            observed_labels = weekly.loc[weekly["split"].eq("test") & observed,
                                         ["sku", "week", "qty", "price"]].copy()
            observed_labels["week"] = pd.to_datetime(observed_labels["week"]).dt.strftime("%Y-%m-%d")
            local = keys if dataset == "uci" else keys.loc[keys["sku"].str.endswith("@" + shard)]
            truth = local.merge(observed_labels, on=["sku", "week"], how="left", validate="one_to_one")
            if truth[["qty", "price"]].isna().any().any():
                raise ValueError(f"{shard}: Phase 1 does not supply every frozen label")
            prices = weekly.loc[weekly["split"].eq("train") & observed, ["qty", "price"]]
            values = prices["qty"].to_numpy(dtype=np.float64)
            price_parts.append(prices["price"].to_numpy(dtype=np.float64, copy=True))
            n_train += len(values)
            sum_qty += float(values.sum())
            sum_squared_qty += float(np.square(values).sum())
            table = f"labels_{shard}.csv"
            truth.to_csv(work / table, index=False)
            shards[shard] = {"file": table, "count": len(truth), "sha256": _sha(work / table)}
        if not n_train or sum(item["count"] for item in shards.values()) != frozen["count"]:
            raise ValueError("Reference does not cover the whole requested dataset")
        mean = sum_qty / n_train
        std = float(np.sqrt(max(0.0, sum_squared_qty / n_train - mean ** 2)))
        q1, q2 = map(float, np.quantile(np.concatenate(price_parts), [1 / 3, 2 / 3]))
        result = {"identity": identity, "dataset": dataset, "shards": shards,
                  "frozen_test_keys_sha256": frozen["sha256"],
                  "n_train": n_train, "train_qty_mean": mean, "train_qty_std": std,
                  "price_q1": q1, "price_q2": q2}
        _write_json(work / "reference.json", result)
        work.rename(output.resolve())
    return result


def _run_rows(dataset, model, seed, directory, reference_root, reference, frozen, identity):
    run = _read_json(directory / "run.json")
    wanted = {**identity, "dataset": dataset, "model": model, "seed": seed,
              "status": "complete", "test_keys_sha256": frozen["sha256"]}
    for name, value in wanted.items():
        if run.get(name) != value:
            raise ValueError(f"{directory}: recorded {name} does not match the requested run")
    checkpoint = run.get("checkpoint")
    if checkpoint:
        _check_sha(directory / checkpoint["path"], checkpoint["sha256"])
    mean, std = reference["train_qty_mean"], reference["train_qty_std"]
    q1, q2 = reference["price_q1"], reference["price_q2"]
    parts, sku_scores, by_store, tiers = [], [], [], []
    wanted_shards = reference["shards"]
    if set(run["prediction_files"]) != set(wanted_shards):
        raise ValueError(f"{directory}: prediction file manifest misses or adds a shard")
    for shard, meta in wanted_shards.items():
        entry = run["prediction_files"][shard]
        name = entry["file"] if isinstance(entry, dict) else f"predictions_{shard}.csv"
        path = directory / name
        _check_sha(path, entry["sha256"] if isinstance(entry, dict) else entry)
        predictions = _csv(path)
        if dataset == "uci":
            predictions["dataset"] = predictions["dataset"].replace({"uci": DATASET_IDS["uci"]})
        rows = align_predictions(predictions, _csv(reference_root / meta["file"]))
        quantities = sufficient(rows)
        totals = quantities.sum()
        by_store.append({"shard": shard, **summarize(totals, train_mean=mean, train_std=std)})
        grouped = quantities.groupby(rows["sku"], sort=False).sum().reset_index()
        sku_scores.extend({"sku": row["sku"], **summarize(row, train_mean=mean, train_std=std)}
                          for row in grouped.to_dict("records"))
        tier = pd.Series(np.select([rows["price"].le(q1), rows["price"].le(q2)],
                                   ["low", "mid"], default="high"), index=rows.index)
        tiers.append(quantities.groupby(tier).sum())
        parts.append(totals)
    total = pd.concat(parts, axis=1).sum(axis=1)
    tier_frame = pd.concat(tiers).groupby(level=0).sum()
    price_scores = []
    for name in ("low", "mid", "high"):
        row = tier_frame.loc[name] if name in tier_frame.index else dict.fromkeys(SUMS, 0)
        price_scores.append({"tier": name, "q1_train": q1, "q2_train": q2,
                             **summarize(row, train_mean=mean, train_std=std)})
    common = {"dataset": dataset, "model": model, "seed": seed}
    if int(total["n"]) != frozen["count"]:
        raise ValueError(f"{directory}: pooled score differs from the frozen count")
    return ({**common, **summarize(total, train_mean=mean, train_std=std),
             "seconds": run.get("seconds"), "device": run.get("device")},
            pd.DataFrame(sku_scores).assign(**common),
            pd.DataFrame(by_store).assign(**common),
            pd.DataFrame(price_scores).assign(**common))


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    with TemporaryDirectory(dir=path.parent) as tmp:
        temp = Path(tmp) / path.name
        frame.to_csv(temp, index=False)
        temp.replace(path)


def audit_forecast_runs(root: Path, output: Path, dataset: str, contract: dict) -> dict:
    audit = output / "audit" / dataset
    audit.mkdir(parents=True, exist_ok=True)
    lock = contract["lock"]
    pending, rows, stores, tiers = [], [], [], []
    matrix_jobs = list(matrix(lock, dataset))
    available = set()
    for model, seed in matrix_jobs:
        marker = output / "forecast" / dataset / model / f"seed{seed}" / "run.json"
        if marker.is_file() and _read_json(marker).get("status") == "complete":
            available.add((model, seed))
    reference_root = audit / "reference"
    if available:
        reference = prepare_reference(root, reference_root, dataset, contract)
    for model, seed in matrix_jobs:
        run_root = output / "forecast" / dataset / model / f"seed{seed}"
        if (model, seed) not in available:
            pending.append({"model": model, "seed": seed})
            continue
        metrics, sku_scores, store_scores, price_scores = _run_rows(
            dataset, model, seed, run_root, reference_root, reference,
            lock["frozen_test_keys"][dataset], _identity(contract))
        rows.append(metrics)
        stores.append(store_scores)
        tiers.append(price_scores)
        sku_dir = audit / "by_sku"
        sku_dir.mkdir(exist_ok=True)
        _atomic_csv(sku_scores, sku_dir / f"{model}_seed{seed}.csv")
    complete_models, final_rows = [], []
    if rows:
        frame = pd.DataFrame(rows)
        _atomic_csv(frame, audit / "forecast_test_by_seed.csv")
        _atomic_csv(pd.concat(stores, ignore_index=True), audit / "forecast_test_by_store.csv")
        tier_frame = pd.concat(tiers, ignore_index=True)
        _atomic_csv(tier_frame, audit / "forecast_test_by_price_range_by_seed.csv")
        for model, group in frame.groupby("model", sort=False):
            needs = [s for m, s in matrix_jobs if model == m]
            if set(group["seed"]) != set(needs):
                continue
            complete_models.append(model)
            fields = ("rmse", "mae", "r2", "wape_pct", "mape_pct_positive_targets",
                      "rmse_train_mean_relative", "rmse_train_z")
            final_rows.append({
                "dataset": dataset, "model": model, "n": int(group["n"].iloc[0]),
                "seeds": len(group), "mape_positive_target_count": int(group["mape_positive_target_count"].iloc[0]),
                "test_keys_sha256": lock["frozen_test_keys"][dataset]["sha256"],
                "rmse_std_sample": float(group["rmse"].std(ddof=1)) if len(group) > 1 else None,
                **{field: group[field].mean() for field in fields},
            })
        if complete_models:
            chosen_tiers = tier_frame.loc[tier_frame["model"].isin(complete_models)]
            tier_summary = chosen_tiers.groupby(["dataset", "model", "tier"], as_index=False).agg(
                n=("n", "first"), seeds=("seed", "size"),
                q1_train=("q1_train", "first"), q2_train=("q2_train", "first"),
                rmse=("rmse", "mean"), rmse_std_sample=("rmse", "std"),
                mae=("mae", "mean"),
            )
            _atomic_csv(tier_summary, audit / "forecast_test_by_price_range.csv")
    final_frame = pd.DataFrame(final_rows)
    if len(final_frame):
        final_frame = final_frame.sort_values(["rmse", "model"])
    else:
        final_frame = pd.DataFrame(columns=["dataset", "model", "n", "seeds", "rmse"])
    _atomic_csv(final_frame, audit / "forecast_test.csv")
    report = {"status": "complete" if not pending else "partial", "dataset": dataset,
              "verified_model_seed_runs": len(rows), "expected_model_seed_runs": len(matrix_jobs),
              "complete_models": complete_models, "pending": pending,
              "deferred_models": lock["forecast"][dataset]["deferred_models"], **_identity(contract)}
    _write_json(audit / "audit.json", report)
    return report


def check_pricing_noise(rows: pd.DataFrame) -> None:
    expected_units = rows["expected_revenue"] / rows["chosen_price"]
    valid = expected_units.gt(0)
    ratio = (rows.loc[valid, "simulated_units"] / expected_units.loc[valid])
    shocks = ratio.groupby([rows.loc[valid, name] for name in ("simulator", "seed", "sku", "week")]).agg(["min", "max"])
    if not np.allclose(shocks["min"], shocks["max"], rtol=1e-8, atol=1e-10):
        raise ValueError("Pricing policies did not share the same draw of demand noise")
    if not rows.loc[~valid, "simulated_units"].eq(0).all():
        raise ValueError("Zero expected demand has nonzero simulated units")


def audit_pricing(root: Path, output: Path, contract: dict) -> dict:
    pricing = output / "pricing"
    marker = pricing / "pricing_run.json"
    if not marker.is_file():
        return {"status": "partial", "verified_decisions": 0}
    run = _read_json(marker)
    if run["status"] != "complete" or any(run[k] != value for k, value in _identity(contract).items()):
        raise ValueError("Pricing run is not a completed run of the locked V2 protocol")
    path = pricing / "pricing_test_decisions.csv"
    _check_sha(path, run["decisions_sha256"])
    rows = _csv(path)
    cfg = contract["lock"]["pricing"]
    cohort_file = root / "m5_pricing_cohort.csv"
    _check_sha(cohort_file, contract["phase1"]["m5"]["pricing_cohort_sha256"])
    cohort = _csv(cohort_file)
    if list(cohort["sku"]) != cfg["cohort_skus"]:
        raise ValueError("Pricing cohort differs from the locked config")
    keys = {(row.sku, date.strftime("%Y-%m-%d"))
            for row in cohort.itertuples() for date in pd.date_range(row.first_test_week, row.last_test_week, freq="7D")}
    if len(keys) != cfg["test_decisions_per_seed"]:
        raise ValueError("Incorrect frozen pricing calendar")
    policies = {*cfg["phase5_methods"], cfg["additional_phase5_control"]}
    simulators = {cfg["primary_song_simulator"], cfg["sensitivity_simulator"]}
    seeds = set(contract["lock"]["phase5_seeds"])
    expected = {(sim, policy, seed) for sim in simulators for policy in policies for seed in seeds}
    groups = rows.groupby(["simulator", "policy", "seed"], sort=False)
    if set(groups.groups) != expected:
        raise ValueError("Pricing output is missing a required simulator / policy / seed combination")
    for group, table in groups:
        actual_keys = list(table[["sku", "week"]].itertuples(index=False, name=None))
        if len(actual_keys) != len(keys) or set(actual_keys) != keys:
            raise ValueError(f"{group}: incomplete or duplicate frozen pricing keys")
        applies_feedback = group[1] in {"ppo_feedback_gradient", "ppo_feedback_no_gradient"}
        ordered = table.sort_values(["sku", "week"])
        step = ordered.groupby("sku", sort=False).cumcount() + 1
        expected_updates = step // cfg["feedback_config"]["update_every"] if applies_feedback else step * 0
        if not ordered["forecast_update_count"].eq(expected_updates).all():
            raise ValueError(f"{group}: incorrect price feedback update count")
    prices = rows["chosen_price"]
    if not (prices.ge(rows["train_price_min"] - 1e-8) & prices.le(rows["train_price_max"] + 1e-8)).all():
        raise ValueError("A policy priced outside the trained bounds")
    if not np.allclose(rows["simulated_revenue"], prices * rows["simulated_units"], rtol=1e-10):
        raise ValueError("Simulated revenue does not equal price times simulated units")
    check_pricing_noise(rows)
    fixed_max = rows.loc[rows["policy"].eq(cfg["additional_phase5_control"])]
    if not np.allclose(fixed_max["chosen_price"], fixed_max["train_price_max"], rtol=0, atol=1e-8):
        raise ValueError("The fixed upper-bound control did not hold the upper price")
    paired_keys = ["simulator", "seed", "sku", "week"]
    ddpg = rows.loc[rows["policy"].eq("ddpg"), [*paired_keys, "chosen_price", "simulated_revenue"]]
    paired = ddpg.merge(fixed_max[[*paired_keys, "chosen_price", "simulated_revenue"]],
                       on=paired_keys, suffixes=("_ddpg", "_fixed"), validate="one_to_one")
    paired["price_matches_fixed_max"] = np.isclose(paired["chosen_price_ddpg"], paired["chosen_price_fixed"],
                                                   rtol=0, atol=1e-8)
    paired["revenue_difference"] = paired["simulated_revenue_ddpg"] - paired["simulated_revenue_fixed"]
    report = paired.groupby(["simulator", "seed"], as_index=False).agg(
        n=("price_matches_fixed_max", "size"), n_prices_equal_upper_bound=("price_matches_fixed_max", "sum"),
        ddpg_minus_fixed_max=("revenue_difference", "sum"),
        simulated_fixed_max=("simulated_revenue_fixed", "sum"),
    )
    report["uplift_vs_fixed_max_pct"] = 100 * report["ddpg_minus_fixed_max"] / report["simulated_fixed_max"].where(
        report["simulated_fixed_max"].gt(0))
    by_sku = ddpg.groupby(["simulator", "seed", "sku"])["chosen_price"].agg(["min", "max"])
    by_sku["span"] = by_sku["max"] - by_sku["min"]
    spans = by_sku.groupby(level=["simulator", "seed"])["span"].agg(
        max_weekly_price_span="max",
        skus_with_weekly_change_gt_one_cent=lambda values: int(values.gt(0.01).sum()),
    ).reset_index()
    report = report.merge(spans, on=["simulator", "seed"], validate="one_to_one")
    audit = output / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    _atomic_csv(report, audit / "pricing_ddpg_check.csv")
    return {"status": "complete", "verified_decisions": len(rows),
            "policies": len(policies), "seeds": sorted(seeds), "simulators": sorted(simulators),
            "ddpg_equal_to_fixed_max_decisions": int(paired["price_matches_fixed_max"].sum()),
            "ddpg_decisions": len(paired),
            "ddpg_sku_seed_simulator_runs_with_weekly_change_gt_one_cent": int(by_sku["span"].gt(0.01).sum()),
            "ddpg_sku_seed_simulator_runs": len(by_sku)}


def main() -> None:
    from .phase5_data import load_contract

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase1-root", type=Path, default=PHASE1_ROOT)
    p.add_argument("--lock-path", type=Path, default=LOCK_PATH)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--part", choices=("all", "uci", "m5", "pricing"), default="all")
    args = p.parse_args()
    contract = load_contract(args.phase1_root, args.lock_path)
    report = {}
    for dataset in ("uci", "m5"):
        if args.part in {"all", dataset}:
            report[dataset] = audit_forecast_runs(args.phase1_root, args.output_root, dataset, contract)
    if args.part in {"all", "pricing"}:
        report["pricing"] = audit_pricing(args.phase1_root, args.output_root, contract)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / f"audit_{args.part}.json", report)
    for section, status in report.items():
        print(f"[Phase 5 audit] {section}: {status['status']}; "
              f"verified={status.get('verified_model_seed_runs', status.get('verified_decisions'))}", flush=True)


if __name__ == "__main__":
    main()
