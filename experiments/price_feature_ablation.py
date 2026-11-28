"""Check XGBoost price grids on held-out weeks with observed historical context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .forecast_feature_ablation import ARTIFACTS, FEATURES, KEYS, _json, _sha, build_features
from .phase1_finalize import _observed
from .phase4_pricing import grid_price
from .phase5_data import PHASE1_ROOT, iter_weekly, load_contract
from .pricing_policy_compare import ArimaxDemand
from .pricing_simulator import KernelDemandSimulator, LagAdjustedKernelSimulator


def candidate_features(row, prices, previous_price):
    """Replace the target-week price and all features derived from that price."""
    prices = np.asarray(prices, dtype=np.float64)
    rows = pd.DataFrame(np.tile(row.to_numpy(dtype=np.float32), (len(prices), 1)), columns=row.index)
    rows["price"] = prices.astype(np.float32)
    rows["price_change"] = (prices / previous_price - 1).astype(np.float32)
    return rows


def _verified(path, digest):
    if _sha(path) != digest:
        raise ValueError(f"Unexpected SHA-256: {path}")
    return path


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_references(step3, feedback, contract):
    step3_file = step3 / "run.json"
    run = _read(feedback / "run.json")
    identity = run["identity"]
    if (run["status"] != "complete" or identity["phase1_sha256"] != contract["phase1_sha256"]
            or identity["lock_sha256"] != contract["lock_sha256"]):
        raise ValueError("Reference pricing runs must use the same frozen protocol")
    _verified(step3_file, identity["step3_sha256"])
    keys = []
    for scope in ("core", "expanded"):
        entry = run["scopes"][scope]
        path = _verified(feedback / scope / "test_keys.csv", entry["files"]["test_keys.csv"])
        data = pd.read_csv(path, dtype={"sku": str}, parse_dates=["week"])
        if len(data) != entry["n_keys"] or data["sku"].nunique() != entry["n_sku"]:
            raise ValueError(f"{scope}: unexpected test cohort size")
        keys.append(data.assign(cohort=scope))
    keys = pd.concat(keys, ignore_index=True)
    fixed = pd.read_csv(_verified(feedback / "validation_fixed_prices.csv", identity["fixed_price_sha256"]))
    if keys.duplicated(KEYS).any() or fixed["sku"].duplicated().any() or set(keys["sku"]) != set(fixed["sku"]):
        raise ValueError("Pricing cohorts and validation price reference must cover the same SKU–stores")
    return keys, fixed.set_index("sku")["price"].to_dict(), _read(step3_file)


def _load_store_models(store, selected_skus, ablation, step3, baseline, frozen):
    location = ablation / store
    marker = _read(location / "scores.json")
    if marker["test_keys"] != frozen:
        raise ValueError(f"{store}: XGBoost training provenance does not match the protocol")
    model_name = "xgb_price.ubj"
    model = xgb.XGBRegressor(n_jobs=2)
    model.load_model(_verified(location / model_name, marker["models"][model_name]))
    model.set_params(n_jobs=2)
    source = baseline["models_by_store"][store]
    rows = pd.read_csv(_verified(step3 / source["models"], source["models_sha256"]), dtype={"sku": str})
    rows = rows.loc[rows["sku"].isin(selected_skus) & rows["model"].eq("arimax_price") & rows["status"].eq("ok")]
    if len(rows) != len(selected_skus) or set(rows["sku"]) != selected_skus:
        raise ValueError(f"{store}: missing matched ARIMAX models")
    return model, {row.sku: json.loads(row.parameters) for row in rows.itertuples(index=False)}


def _store_decisions(weekly, targets, model, states, fixed, config):
    columns = list(FEATURES["xgb_price"])
    features = build_features(weekly).loc[:, columns]
    previous_prices = weekly["price"].where(_observed(weekly)).groupby(weekly["sku"], sort=False).shift(1)
    target_rows = weekly.reset_index(names="panel_row").merge(targets, on=KEYS, validate="one_to_one")
    if (len(target_rows) != len(targets) or not target_rows["split"].eq("test").all()
            or not _observed(target_rows).all()):
        raise ValueError("Simulation targets must be observed weeks on the frozen test roster")
    training = weekly.loc[weekly["split"].eq("train") & _observed(weekly)]
    bandwidth = config["gaussian_seasonal_bandwidth_weeks"]
    simulators = {
        config["primary_song_simulator"]: KernelDemandSimulator(bandwidth).fit(training),
        config["sensitivity_simulator"]: LagAdjustedKernelSimulator(
            config["sensitivity_weight"], bandwidth, config["ratio_limit"]).fit(training),
    }
    arimax = ArimaxDemand(states)
    grids, blocks = [], []
    for row in target_rows.itertuples(index=False):
        lo, hi = (states[row.sku][key] for key in ("price_min", "price_max"))
        context = features.loc[row.panel_row]
        prior = previous_prices.at[row.panel_row]
        if not np.isfinite([context["qty_lag_1"], prior]).all() or prior <= 0:
            raise ValueError(f"{row.sku}: observed anchor history is missing")
        grid = np.unique(np.append(np.linspace(lo, hi, config["grid_points"]), np.clip(prior, lo, hi)))
        grids.append(grid)
        blocks.append(candidate_features(context, grid, prior))
        if not lo - 1e-6 <= fixed[row.sku] <= hi + 1e-6:
            raise ValueError(f"{row.sku}: fixed price is outside the training range")
    flat_predictions = model.predict(pd.concat(blocks, ignore_index=True)).clip(min=0)
    prediction_grids = np.split(flat_predictions, np.cumsum([len(grid) for grid in grids])[:-1])
    results = []
    for row, grid, xgb_qty in zip(target_rows.itertuples(index=False), grids, prediction_grids, strict=True):
        context = features.loc[row.panel_row, ["qty_lag_1", "event_days", "snap_days"]].to_dict()
        context["price_lag_1"] = previous_prices.at[row.panel_row]
        arimax_qty = arimax.predict_prices({"sku": row.sku, **context}, grid)
        picks = {"fixed_validation_sku": fixed[row.sku]}
        curves = {"grid_arimax_step3": arimax_qty, "grid_xgb_features": xgb_qty}
        upward = {name: bool((np.diff(curve) > 1e-4).any()) for name, curve in curves.items()}
        picks.update({name: grid_price(grid, curve, context["price_lag_1"]) for name, curve in curves.items()})
        for sim_name, sim in simulators.items():
            for method, price in picks.items():
                demand = sim.predict_mean(row.sku, price, row.week, context)
                results.append({"sku": row.sku, "week": row.week, "cohort": row.cohort,
                                "simulator": sim_name, "method": method, "price": price,
                                "simulated_expected_qty": demand, "simulated_expected_revenue": price * demand,
                                "curve_has_increase": upward.get(method)})
    return results


def run(root, ablation, step3, feedback, cache):
    contract = load_contract(root)
    trained = _read(ablation / "run.json")
    identity = trained["identity"]
    if (trained["status"] != "complete" or trained["validation_selected_xgb"] != "xgb_price"
            or set(trained["stores"]) != set(contract["m5_stores"])
            or identity["phase1_sha256"] != contract["phase1_sha256"]
            or identity["baseline_run_sha256"] != _sha(step3 / "run.json")
            or identity["features"]["xgb_price"] != list(FEATURES["xgb_price"])):
        raise ValueError("Run and select the full price-aware ablation before the pricing check")
    quality = pd.read_csv(ablation / "metrics.csv").pivot(index="split", columns="model", values=["rmse", "mae"])
    if any(not (quality[metric]["xgb_price"] < quality[metric]["arimax_price"]).all()
           for metric in ("rmse", "mae")):
        raise ValueError("Price-aware XGBoost must beat ARIMAX on validation and test for this check")
    keys, fixed, baseline = _load_references(step3, feedback, contract)
    stores = keys["sku"].str.rsplit("@", n=1).str[-1]
    decisions = []
    for store, weekly, _ in iter_weekly(root, "m5", contract, cache_root=cache):
        targets = keys.loc[stores.eq(store)]
        selected = set(targets["sku"])
        model, states = _load_store_models(store, selected, ablation, step3, baseline,
                                          contract["test_key_shards"]["m5"][store])
        subset = weekly.loc[weekly["sku"].isin(selected)].reset_index(drop=True)
        decisions.extend(_store_decisions(subset, targets, model, states, fixed, contract["lock"]["pricing"]))
    detail = pd.DataFrame(decisions)
    summary = detail.groupby(["cohort", "simulator", "method"], as_index=False).agg(
        n=("price", "size"), simulated_expected_revenue=("simulated_expected_revenue", "sum"),
        upward_curve_share=("curve_has_increase", "mean"),
    )
    fixed_summary = summary.loc[summary["method"].eq("fixed_validation_sku"),
                                ["cohort", "simulator", "simulated_expected_revenue"]]
    summary = summary.merge(fixed_summary.rename(columns={"simulated_expected_revenue": "fixed_revenue"}),
                            on=["cohort", "simulator"], validate="many_to_one")
    summary["uplift_vs_fixed_pct"] = 100 * (summary["simulated_expected_revenue"] / summary["fixed_revenue"] - 1)
    detail.to_csv(ablation / "price_one_step_decisions.csv", index=False)
    summary.to_csv(ablation / "price_one_step_summary.csv", index=False)
    _json(ablation / "price_one_step_metadata.json", {
        "mode": "one_week_expected_revenue_with_observed_history_reset",
        "source_sha256": _sha(Path(__file__)),
        "ablation_run_sha256": _sha(ablation / "run.json"),
        "pricing_reference_run_sha256": _sha(feedback / "run.json"),
        "n_heldout_keys": len(keys), "n_sku_stores": keys["sku"].nunique(),
        "decisions_sha256": _sha(ablation / "price_one_step_decisions.csv"),
        "summary_sha256": _sha(ablation / "price_one_step_summary.csv"),
    })
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--ablation-dir", type=Path, default=ARTIFACTS / "forecast_feature_ablation_v2")
    parser.add_argument("--baseline-dir", type=Path, default=ARTIFACTS / "pricing_arimax_v2")
    parser.add_argument("--reference-pricing-dir", type=Path, default=ARTIFACTS / "pricing_arimax_feedback_v2")
    parser.add_argument("--cache-dir", type=Path, default=ARTIFACTS / "phase5_v2" / "data_cache")
    args = parser.parse_args()
    run(args.prepared_root, args.ablation_dir, args.baseline_dir, args.reference_pricing_dir, args.cache_dir)


if __name__ == "__main__":
    main()
