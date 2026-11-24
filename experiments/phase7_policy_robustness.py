"""Rescore frozen Phase 6 prices under train-fitted demand assumptions.

This is evaluator sensitivity, not policy retraining or a causal estimate.
The PPO/DDPG/SAC observations and actions remain exactly as saved in Phase 6.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pricing_simulator import KernelDemandSimulator, LagAdjustedKernelSimulator


DEFAULT_ROOT = Path(__file__).resolve().parent / "artifacts" / "song_m5_pilot"
VARIANTS = ("song_kernel", "lag_half", "lag_selected", "lag_full",
            "flat_cv_level", "flat_price_response")
SCORABLE_METHODS = ("hold_current", "fixed_mean", "heuristic", "forecast_grid_ensemble",
                    "ppo_frozen", "ddpg", "sac")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_hash(path: Path, expected: str, name: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"{name} changed since Phase 6")


def _models(weekly: pd.DataFrame, selected_weight: float, ratio_limit: float,
            cv_flat_weight: float) -> dict:
    if not 0 < selected_weight <= 1:
        raise ValueError("The selected lag weight must be positive")
    # The surface fit is identical for all lag weights. A single train-only
    # fit also avoids spending time refitting unchanged kernel parameters.
    song = KernelDemandSimulator().fit(weekly)
    models = {"song_kernel": song}
    for name, weight in (("lag_half", .5), ("lag_selected", selected_weight),
                         ("lag_full", 1.0), ("cv_flat_reference", cv_flat_weight)):
        model = LagAdjustedKernelSimulator(weight, ratio_limit=ratio_limit)
        model.surfaces = song.surfaces
        models[name] = model
    return models


def _load_cohort(pilot_dir: Path, phase6_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    phase6 = json.loads((phase6_dir / "manifest.json").read_text(encoding="utf-8"))
    _check_hash(pilot_dir / "weekly.csv", phase6["weekly_sha256"], "Weekly panel")
    _check_hash(pilot_dir / "windows.csv", phase6["windows_sha256"], "Forecast windows")
    decisions = pd.read_csv(phase6_dir / "one_step_decisions.csv", dtype={"sku": str},
                            parse_dates=["week"])
    selected = pd.read_csv(phase6_dir / "selected_price_decisions.csv",
                           dtype={"sku": str}, parse_dates=["week"])
    decisions = decisions.loc[decisions["method"].isin(SCORABLE_METHODS)].copy()
    if decisions.duplicated(["split", "sku", "week", "seed", "method"]).any():
        raise ValueError("Phase 6 has duplicate method decisions")
    if selected.duplicated(["split", "sku", "week"]).any():
        raise ValueError("Phase 6 selected more than one price per SKU-week")
    if ((decisions["chosen_price"] < decisions["local_min"] - 1e-8)
            | (decisions["chosen_price"] > decisions["local_max"] + 1e-8)).any():
        raise ValueError("A Phase 6 decision exceeds its common local price range")
    weekly = pd.read_csv(pilot_dir / "weekly.csv", dtype={"sku": str}, parse_dates=["week"])
    windows = pd.read_csv(pilot_dir / "windows.csv", dtype={"sku": str}, parse_dates=["week"])
    needed = ["split", "sku", "week", "qty", "price", "observed_sales",
              "qty_lag_1", "price_lag_1", "event_days", "snap_days"]
    context = windows[needed]
    decisions = decisions.merge(context, on=["split", "sku", "week"],
                                how="left", validate="many_to_one")
    if decisions[needed[3:]].isna().any().any() or not decisions["observed_sales"].all():
        raise ValueError("Phase 6 decisions lack an observed target and past context")
    if not np.allclose(decisions["reference_price"], decisions["price_lag_1"], atol=1e-8):
        raise ValueError("Phase 6 reference price differs from the previous panel week")
    expected = phase6["cohort"]
    keys = decisions[["split", "sku", "week"]].drop_duplicates()
    for split in ("validation", "test"):
        count = int(keys["split"].eq(split).sum())
        if count != expected[split]["eligible_rows"]:
            raise ValueError(f"Phase 6 {split} cohort changed")
    return weekly, decisions, {"selected": selected, "phase6": phase6, "keys": keys}


def _score_actions(decisions: pd.DataFrame, models: dict) -> pd.DataFrame:
    output = []
    for (split, sku, week), group in decisions.groupby(["split", "sku", "week"], sort=False):
        first = group.iloc[0]
        reference = float(first["reference_price"])
        context = {name: float(first[name]) for name in
                   ("qty_lag_1", "price_lag_1", "event_days", "snap_days")}
        base = {name: float(model.predict_mean(str(sku), reference, week, context))
                for name, model in models.items()}
        price_map: dict[float, dict[str, float]] = {}
        for price in group["chosen_price"].unique():
            price = float(price)
            values = {name: float(model.predict_mean(str(sku), price, week, context))
                      for name, model in models.items()}
            values["flat_cv_level"] = base["cv_flat_reference"]
            values["flat_price_response"] = context["qty_lag_1"]
            price_map[price] = values
        base["flat_cv_level"] = base["cv_flat_reference"]
        base["flat_price_response"] = context["qty_lag_1"]
        for decision in group.itertuples(index=False):
            price = float(decision.chosen_price)
            for variant in VARIANTS:
                units = price_map[price][variant]
                output.append({
                    "variant": variant, "split": split, "sku": str(sku),
                    "week": pd.Timestamp(week).date().isoformat(),
                    "seed": int(decision.seed), "method": str(decision.method),
                    "reference_price": reference, "chosen_price": price,
                    "expected_units": units,
                    "expected_revenue": price * units,
                    "hold_expected_revenue": reference * base[variant],
                })
    return pd.DataFrame(output)


def _summarize(scored: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = scored.groupby(["variant", "split", "method", "seed"], as_index=False).agg(
        decisions=("sku", "size"), expected_revenue=("expected_revenue", "sum"),
        hold_expected_revenue=("hold_expected_revenue", "sum"))
    runs["uplift_vs_hold_pct"] = 100 * (
        runs["expected_revenue"] / runs["hold_expected_revenue"] - 1)
    summary = runs.groupby(["variant", "split", "method"], as_index=False).agg(
        decisions=("decisions", "first"), seeds=("seed", "nunique"),
        uplift_mean_pct=("uplift_vs_hold_pct", "mean"),
        uplift_sd_pct=("uplift_vs_hold_pct", "std"))
    selected_keys = selected[["split", "sku", "week", "seed", "method", "chosen_price"]].copy()
    selected_keys["week"] = pd.to_datetime(selected_keys["week"])
    scored_keys = scored.copy()
    scored_keys["week"] = pd.to_datetime(scored_keys["week"])
    picked = scored_keys.merge(selected_keys, on=["split", "sku", "week", "seed", "method"],
                               how="inner", suffixes=("", "_selected"),
                               validate="many_to_one")
    if not np.allclose(picked["chosen_price"], picked["chosen_price_selected"], atol=1e-8):
        raise ValueError("The Phase 6 selected price changed")
    if len(picked) != len(selected) * len(VARIANTS):
        raise ValueError("The frozen selected rule does not cover every variant")
    chosen = picked.groupby(["variant", "split"], as_index=False).agg(
        decisions=("sku", "size"), expected_revenue=("expected_revenue", "sum"),
        hold_expected_revenue=("hold_expected_revenue", "sum"))
    chosen["uplift_vs_hold_pct"] = 100 * (
        chosen["expected_revenue"] / chosen["hold_expected_revenue"] - 1)
    by_sku = picked.groupby(["variant", "split", "sku"], as_index=False).agg(
        decisions=("week", "size"), expected_revenue=("expected_revenue", "sum"),
        hold_expected_revenue=("hold_expected_revenue", "sum"))
    by_sku["uplift_vs_hold_pct"] = 100 * (
        by_sku["expected_revenue"] / by_sku["hold_expected_revenue"] - 1)
    return runs, summary, chosen, by_sku


def _ranking(summary: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ("validation", "test"):
        base = summary.loc[summary["split"].eq(split)
                           & summary["variant"].eq("lag_selected")].set_index("method")
        base_rank = base["uplift_mean_pct"].rank(ascending=False)
        for variant in VARIANTS:
            part = summary.loc[summary["split"].eq(split)
                               & summary["variant"].eq(variant)].set_index("method")
            winner = part.sort_values(["uplift_mean_pct", "method"],
                                      ascending=[False, True]).iloc[0]
            top_name = part.sort_values(["uplift_mean_pct", "method"],
                                        ascending=[False, True]).index[0]
            chosen = selected.loc[selected["split"].eq(split)
                                  & selected["variant"].eq(variant)].iloc[0]
            rows.append({
                "variant": variant, "split": split,
                "best_method_diagnostic_only": top_name,
                "best_method_uplift_pct": float(winner["uplift_mean_pct"]),
                "rank_correlation_with_selected_simulator": float(
                    base_rank.corr(part["uplift_mean_pct"].rank(ascending=False), method="spearman")),
                "frozen_per_sku_rule_uplift_pct": float(chosen["uplift_vs_hold_pct"]),
            })
    return pd.DataFrame(rows)


def _observed_price_check(keys: pd.DataFrame, decisions: pd.DataFrame,
                          models: dict) -> pd.DataFrame:
    targets = decisions.drop_duplicates(["split", "sku", "week"])
    if len(targets) != len(keys):
        raise ValueError("The observed-price cohort differs from Phase 6")
    records = []
    for row in targets.itertuples(index=False):
        context = {name: float(getattr(row, name)) for name in
                   ("qty_lag_1", "price_lag_1", "event_days", "snap_days")}
        bound = models["song_kernel"].surfaces[str(row.sku)]
        in_support = bound.price_min - 1e-8 <= row.price <= bound.price_max + 1e-8
        for variant in ("naive_last", "song_kernel", "lag_selected"):
            prediction = (float(row.qty_lag_1) if variant == "naive_last" else
                          float(models[variant].predict_mean(str(row.sku), float(row.price),
                                                             row.week, context)))
            records.append({"split": row.split, "sku": row.sku, "week": row.week,
                            "variant": variant, "observed_price_in_train_support": in_support,
                            "actual_units": float(row.qty), "predicted_units": prediction})
    predictions = pd.DataFrame(records)
    scores = []
    for (variant, split), frame in predictions.groupby(["variant", "split"]):
        for segment, subset in (("all_phase6_rows", frame),
                                ("target_price_in_train_support",
                                 frame.loc[frame["observed_price_in_train_support"]])):
            if subset.empty:
                continue
            error = subset["predicted_units"] - subset["actual_units"]
            scores.append({"variant": variant, "split": split, "segment": segment,
                           "rows": len(subset),
                           "rmse": float(np.sqrt(np.mean(error ** 2))),
                           "mae": float(np.mean(np.abs(error)))})
    return pd.DataFrame(scores)


def run(pilot_dir: Path = DEFAULT_ROOT, phase6_dir: Path | None = None,
        output_dir: Path | None = None) -> dict:
    pilot_dir = Path(pilot_dir)
    phase6_dir = Path(phase6_dir) if phase6_dir else pilot_dir / "phase6_price_selection"
    output_dir = Path(output_dir) if output_dir else pilot_dir / "phase7_policy_robustness"
    weekly, decisions, info = _load_cohort(pilot_dir, phase6_dir)
    config_path = pilot_dir / "simulator_selection" / "selected_simulator.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    demand_manifest_path = pilot_dir / "phase7_demand_validation" / "manifest.json"
    demand_manifest = json.loads(demand_manifest_path.read_text(encoding="utf-8"))
    _check_hash(pilot_dir / "weekly.csv", demand_manifest["prepared_weekly_sha256"],
                "Train-CV demand panel")
    chosen_demand = str(demand_manifest["selection"]["model"])
    if not chosen_demand.startswith("flat_response_"):
        raise ValueError("This robustness protocol expects the Phase 7 CV-selected flat response")
    cv_flat_weight = float(chosen_demand.rsplit("_", 1)[1])
    _check_hash(pilot_dir / "weekly.csv", config["prepared_weekly_sha256"],
                "Simulator training data")
    models = _models(weekly, float(config["recent_weight"]),
                     float(config["ratio_limit"]), cv_flat_weight)
    scored = _score_actions(decisions, models)
    cv_predictions_path = pilot_dir / "phase7_demand_validation" / "observed_price_predictions.csv"
    cv_predictions = pd.read_csv(
        cv_predictions_path, usecols=["split", "sku", "week", "model", "predicted_qty"],
        dtype={"sku": str})
    cv_predictions = cv_predictions.loc[cv_predictions["model"].eq(chosen_demand)]
    flat_scores = scored.loc[scored["variant"].eq("flat_cv_level"),
                             ["split", "sku", "week", "expected_units"]]
    if flat_scores.groupby(["split", "sku", "week"])["expected_units"].nunique().gt(1).any():
        raise AssertionError("Flat demand changed with a frozen policy action")
    flat_scores = flat_scores.drop_duplicates(["split", "sku", "week"])
    flat_check = flat_scores.merge(cv_predictions, on=["split", "sku", "week"],
                                   how="left", validate="one_to_one")
    if flat_check["predicted_qty"].isna().any():
        raise ValueError("Train-CV-selected demand predictions do not cover Phase 6")
    flat_error = float(np.max(np.abs(flat_check["expected_units"]
                                     - flat_check["predicted_qty"])))
    if flat_error > 1e-6:
        raise AssertionError(f"Flat evaluator differs from the CV-selected model: {flat_error}")
    selected_scores = scored.loc[scored["variant"].eq("lag_selected")].copy()
    selected_scores["week"] = pd.to_datetime(selected_scores["week"])
    same = selected_scores.merge(
        decisions[["split", "sku", "week", "seed", "method",
                   "simulator_expected_units", "simulator_expected_revenue"]],
        on=["split", "sku", "week", "seed", "method"], validate="one_to_one")
    error = float(np.max(np.abs(same["expected_revenue"] - same["simulator_expected_revenue"])))
    if error > 1e-6:
        raise AssertionError(f"Selected simulator no longer reproduces Phase 6: {error}")
    runs, summary, chosen, by_sku = _summarize(scored, info["selected"])
    ranking = _ranking(summary, chosen)
    observed = _observed_price_check(info["keys"], decisions, models)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_dir / "frozen_action_scores.csv", index=False)
    runs.to_csv(output_dir / "policy_runs_by_assumption.csv", index=False)
    summary.to_csv(output_dir / "policy_summary_by_assumption.csv", index=False)
    chosen.to_csv(output_dir / "frozen_selected_rule_by_assumption.csv", index=False)
    by_sku.to_csv(output_dir / "frozen_selected_rule_by_sku.csv", index=False)
    ranking.to_csv(output_dir / "ranking_stability.csv", index=False)
    observed.to_csv(output_dir / "phase6_cohort_observed_price_metrics.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase6_manifest_sha256": sha256(phase6_dir / "manifest.json"),
        "phase6_decisions_sha256": sha256(phase6_dir / "one_step_decisions.csv"),
        "phase6_selected_decisions_sha256": sha256(phase6_dir / "selected_price_decisions.csv"),
        "simulator_selection_sha256": sha256(config_path),
        "train_cv_demand_manifest_sha256": sha256(demand_manifest_path),
        "train_cv_demand_predictions_sha256": sha256(cv_predictions_path),
        "weekly_sha256": sha256(pilot_dir / "weekly.csv"),
        "same_price_action_reproduction_max_abs_revenue_diff": error,
        "cv_selected_flat_model_max_abs_units_diff": flat_error,
        "variants": {
            "song_kernel": "Train-fitted original Gaussian kernel response",
            "lag_half": "Same fitted kernel with fixed last-demand weight 0.5",
            "lag_selected": f"Phase 4 selected last-demand weight {config['recent_weight']}",
            "lag_full": "Same fitted kernel with fixed last-demand weight 1.0",
            "flat_cv_level": f"Phase 7 CV-selected {chosen_demand} at the reference price, held constant across candidates",
            "flat_price_response": "Last observed demand held constant across candidate prices; zero-response diagnostic only",
        },
        "policy_actions": "Frozen Phase 6 one-step observed-history prices; no retraining or action recomputation",
        "selection": "Phase 6 global and per-SKU validation selections stay frozen; variant winners are diagnostic only",
        "observed_price_gate_scope": "Both full Phase 6 action cohort and subset whose target observed price is within train support",
        "interpretation": "Evaluator sensitivity only; observational M5 cannot validate demand at unchosen prices",
        "test_status": "Exploratory; the same test period had already been inspected in earlier phases",
        "business_deployment_supported": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--phase6-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run(args.pilot_dir, args.phase6_dir, args.output_dir)
    print(json.dumps({"action_reproduction_max_abs_revenue_diff":
                      result["same_price_action_reproduction_max_abs_revenue_diff"],
                      "variants": list(result["variants"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
