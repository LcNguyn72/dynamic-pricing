"""Audit M5 demand simulators using rolling origins entirely inside train.

The observed-price check is a necessary predictive diagnostic, never evidence
for counterfactual demand at an unchosen price. The original validation and
test periods are reported after train-only model selection and remain
exploratory because earlier phases have already inspected them.
The fixed 20-SKU pilot cohort was ranked using the full original train span;
rolling folds do not reselect SKUs at each earlier origin, so CV is conditional
on that cohort and is not a fully nested product-selection estimate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .audit_prepared import audit as audit_panel
from .forecast_benchmark import _score
from .pricing_simulator import KernelDemandSimulator, LagAdjustedKernelSimulator


DEFAULT_ROOT = Path(__file__).resolve().parent / "artifacts" / "song_m5_pilot"
WEIGHTS = (0.25, 0.50, 0.75, 1.0)
FOLD_FRACTIONS = (0.55, 0.70, 0.85)
FOLD_HORIZON = 20
RATIO_LIMIT = 2.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FlatResponseLagKernel(LagAdjustedKernelSimulator):
    """No-price-response stress case with the same last-demand adjustment.

    For a given observed history and calendar, the candidate price does not
    change predicted units. This is a null scenario, not an elasticity
    estimate. It retains the fitted train support and price bounds.
    """

    def predict_mean(self, sku: str, price: float, week,
                     context: Mapping | None = None) -> float:
        if not np.isfinite(price) or price <= 0:
            raise ValueError("price must be positive and finite")
        reference = (context or {}).get("price_lag_1", price)
        return super().predict_mean(sku, float(reference), week, context)


def _rolling_folds(train_weeks: list[pd.Timestamp]) -> list[dict]:
    """Three non-overlapping 20-week one-step evaluation blocks within train."""
    if len(train_weeks) < 150:
        raise ValueError("The M5 pilot needs at least 150 train weeks for this fixed protocol")
    origins = [int(len(train_weeks) * fraction) for fraction in FOLD_FRACTIONS]
    if origins[-1] + FOLD_HORIZON > len(train_weeks):
        raise ValueError("Last rolling fold exceeds train")
    if any(a + FOLD_HORIZON > b for a, b in zip(origins, origins[1:])):
        raise ValueError("Rolling evaluation blocks overlap")
    return [
        {
            "fold": index + 1,
            "fit_weeks": train_weeks[:origin],
            "eval_weeks": train_weeks[origin:origin + FOLD_HORIZON],
        }
        for index, origin in enumerate(origins)
    ]


def _fit_models(train: pd.DataFrame) -> dict[str, KernelDemandSimulator]:
    # ProductSurface fitting is independent of the lag-adjustment weight.
    base = KernelDemandSimulator().fit(train)
    models: dict[str, KernelDemandSimulator] = {"song_kernel": base}
    for weight in WEIGHTS:
        label = f"{weight:.2f}"
        for prefix, cls in (("lag_kernel", LagAdjustedKernelSimulator),
                            ("flat_response", FlatResponseLagKernel)):
            model = cls(weight, ratio_limit=RATIO_LIMIT)
            model.surfaces = base.surfaces
            models[f"{prefix}_{label}"] = model
    return models


def _predict_rows(models: dict[str, KernelDemandSimulator], rows: pd.DataFrame,
                  split: str, fold: int | None = None) -> pd.DataFrame:
    """Score all models on identical observed one-step rows and fit support."""
    support = next(iter(models.values())).surfaces
    eligible = rows.loc[rows["observed_sales"].astype(bool)
                        & rows["sku"].astype(str).isin(support)].copy()
    if eligible.empty or eligible[["qty_lag_1", "price_lag_1"]].isna().any().any():
        raise ValueError("Missing observed rows or last-week context")
    eligible["train_price_min"] = eligible["sku"].map({sku: s.price_min for sku, s in support.items()})
    eligible["train_price_max"] = eligible["sku"].map({sku: s.price_max for sku, s in support.items()})
    eligible["price_in_train_support"] = (
        eligible["price"].ge(eligible["train_price_min"] - 1e-8)
        & eligible["price"].le(eligible["train_price_max"] + 1e-8)
    )
    records = []
    for row in eligible.itertuples(index=False):
        context = {name: getattr(row, name) for name in
                   ("event_days", "snap_days", "qty_lag_1", "price_lag_1")
                   if hasattr(row, name)}
        common = {
            "split": split, "fold": fold, "sku": str(row.sku),
            "week": pd.Timestamp(row.week).date().isoformat(),
            "observed_price": float(row.price), "observed_qty": float(row.qty),
            "last_qty": float(row.qty_lag_1), "last_price": float(row.price_lag_1),
            "train_price_min": float(row.train_price_min),
            "train_price_max": float(row.train_price_max),
            "price_in_train_support": bool(row.price_in_train_support),
        }
        records.append({**common, "model": "naive_last", "predicted_qty": float(row.qty_lag_1)})
        for name, model in models.items():
            estimate = model.predict_mean(str(row.sku), float(row.price), row.week, context)
            records.append({**common, "model": name, "predicted_qty": float(estimate)})
    frame = pd.DataFrame(records)
    if frame.duplicated(["split", "fold", "sku", "week", "model"]).any():
        raise AssertionError("Duplicate observed-price prediction")
    return frame


def _metrics(predictions: pd.DataFrame, *, phase6_keys: set[tuple] | None = None) -> pd.DataFrame:
    frame = predictions.copy()
    if phase6_keys is not None:
        frame["phase6_pricing_eligible"] = [
            (row.split, row.sku, row.week) in phase6_keys
            for row in frame.itertuples(index=False)
        ]
    records = []
    scopes = [("all", frame),
              ("in_support", frame.loc[frame["price_in_train_support"]]),
              ("out_of_support", frame.loc[~frame["price_in_train_support"]])]
    if phase6_keys is not None:
        scopes.append(("phase6_pricing_eligible", frame.loc[frame["phase6_pricing_eligible"]]))
        scopes.append(("phase6_pricing_and_target_in_support", frame.loc[
            frame["phase6_pricing_eligible"] & frame["price_in_train_support"]]))
    for segment, segment_rows in scopes:
        for (split, model), rows in segment_rows.groupby(["split", "model"]):
            score = _score(rows.rename(columns={"observed_qty": "qty"}),
                           rows["predicted_qty"].to_numpy())
            records.append({"split": split, "segment": segment, "model": model, **score})
    return pd.DataFrame(records).sort_values(["split", "segment", "rmse", "model"])


def _by_sku(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (split, sku, model), rows in predictions.loc[
            predictions["price_in_train_support"]].groupby(["split", "sku", "model"]):
        score = _score(rows.rename(columns={"observed_qty": "qty"}),
                       rows["predicted_qty"].to_numpy())
        records.append({"split": split, "sku": sku, "model": model, **score})
    return pd.DataFrame(records).sort_values(["split", "sku", "model"])


def _fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (fold, model), rows in predictions.groupby(["fold", "model"]):
        for segment, subset in (("all", rows),
                                ("in_support", rows.loc[rows["price_in_train_support"]])):
            if subset.empty:
                continue
            score = _score(subset.rename(columns={"observed_qty": "qty"}),
                           subset["predicted_qty"].to_numpy())
            records.append({"fold": int(fold), "model": model, "segment": segment, **score})
    return pd.DataFrame(records).sort_values(["fold", "segment", "rmse", "model"])


def _response_audit(models: dict[str, KernelDemandSimulator], train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sku, group in train.loc[train["observed_sales"].astype(bool)].groupby("sku"):
        surface = next(iter(models.values())).surfaces.get(str(sku))
        if surface is None:
            continue
        reference = float(group["price"].median())
        low = max(surface.price_min, reference * .99)
        high = min(surface.price_max, reference * 1.01)
        if high <= low:
            continue
        last = group.sort_values("week").iloc[-1]
        context = {
            "event_days": float(last.get("event_days", 0)),
            "snap_days": float(last.get("snap_days", 0)),
            "qty_lag_1": float(last["qty"]), "price_lag_1": reference,
        }
        for name, model in models.items():
            demand_low = model.predict_mean(str(sku), low, last["week"], context)
            demand_high = model.predict_mean(str(sku), high, last["week"], context)
            delta = float(demand_high - demand_low)
            rows.append({
                "sku": str(sku), "model": name, "train_price_levels": surface.price_levels,
                "reference_price": reference, "price_low": low, "price_high": high,
                "demand_low": demand_low, "demand_high": demand_high,
                "demand_change": delta,
                "response_direction": ("positive" if delta > 1e-8 else
                                       "negative" if delta < -1e-8 else "flat"),
            })
    return pd.DataFrame(rows).sort_values(["model", "sku"])


def _phase6_keys(prepared_root: Path, phase6_dir: Path) -> tuple[set[tuple] | None, str | None]:
    path = phase6_dir / "selected_price_decisions.csv"
    manifest_path = phase6_dir / "manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        return None, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["weekly_sha256"] != _sha256(prepared_root / "weekly.csv"):
        raise ValueError("Phase 6 decisions belong to a different weekly panel")
    decisions = pd.read_csv(path, usecols=["split", "sku", "week"], dtype={"sku": str})
    if decisions.duplicated(["split", "sku", "week"]).any():
        raise ValueError("Phase 6 cohort has duplicate SKU-week keys")
    return set(decisions.itertuples(index=False, name=None)), _sha256(path)


def run(prepared_root: Path = DEFAULT_ROOT, output_dir: Path | None = None,
        phase6_dir: Path | None = None) -> dict:
    prepared_root = Path(prepared_root)
    output_dir = Path(output_dir) if output_dir else prepared_root / "phase7_demand_validation"
    phase6_dir = Path(phase6_dir) if phase6_dir else prepared_root / "phase6_price_selection"
    panel = audit_panel(prepared_root)
    if not panel["dataset"].startswith("M5"):
        raise ValueError("Phase 7 demand audit expects the prepared M5 pilot")
    weekly = pd.read_csv(prepared_root / "weekly.csv", dtype={"sku": str}, parse_dates=["week"])
    weekly = weekly.sort_values(["sku", "week"]).copy()
    weekly["qty_lag_1"] = weekly.groupby("sku")["qty"].shift(1)
    weekly["price_lag_1"] = weekly.groupby("sku")["price"].shift(1)
    train = weekly.loc[weekly["split"].eq("train")].copy()
    train_weeks = list(pd.to_datetime(sorted(train["week"].unique())))
    folds = _rolling_folds(train_weeks)
    cv_frames = []
    fold_metadata = []
    for item in folds:
        fit = train.loc[train["week"].isin(item["fit_weeks"])].copy()
        evaluate = train.loc[train["week"].isin(item["eval_weeks"])].copy()
        models = _fit_models(fit)
        cv_frames.append(_predict_rows(models, evaluate, "train_cv", item["fold"]))
        fold_metadata.append({
            "fold": item["fold"], "fit_start": item["fit_weeks"][0].date().isoformat(),
            "fit_end": item["fit_weeks"][-1].date().isoformat(),
            "eval_start": item["eval_weeks"][0].date().isoformat(),
            "eval_end": item["eval_weeks"][-1].date().isoformat(),
            "fit_weeks": len(item["fit_weeks"]), "eval_weeks": len(item["eval_weeks"]),
            "eligible_skus": len(models["song_kernel"].surfaces),
        })
        print(f"[Phase 7 demand] fold {item['fold']}: {len(models['song_kernel'].surfaces)} SKUs", flush=True)
    cv_predictions = pd.concat(cv_frames, ignore_index=True)
    cv_metrics = _metrics(cv_predictions)
    fold_metrics = _fold_metrics(cv_predictions)
    selection_rows = cv_metrics.loc[cv_metrics["split"].eq("train_cv")
                                     & cv_metrics["segment"].eq("in_support")]
    baseline = selection_rows.loc[selection_rows["model"].eq("naive_last")].iloc[0]
    candidates = selection_rows.loc[~selection_rows["model"].eq("naive_last")]
    winner = candidates.sort_values(["rmse", "model"]).iloc[0]

    full_models = _fit_models(train)
    holdout_frames = []
    for split in ("validation", "test"):
        rows = weekly.loc[weekly["split"].eq(split)]
        holdout_frames.append(_predict_rows(full_models, rows, split))
    holdout_predictions = pd.concat(holdout_frames, ignore_index=True)
    phase6_keys, phase6_sha = _phase6_keys(prepared_root, phase6_dir)
    if phase6_keys is not None:
        actual_keys = set(holdout_predictions.loc[
            holdout_predictions["model"].eq("naive_last"),
            ["split", "sku", "week"]].itertuples(index=False, name=None))
        if not phase6_keys <= actual_keys:
            raise ValueError("Phase 6 cohort contains rows outside the Phase 7 holdout")
    holdout_metrics = _metrics(holdout_predictions, phase6_keys=phase6_keys)
    by_sku = _by_sku(holdout_predictions)
    response = _response_audit(full_models, train)
    gates = {}
    for split, segment in (("validation", "in_support"), ("test", "in_support"),
                           ("validation", "phase6_pricing_and_target_in_support"),
                           ("test", "phase6_pricing_and_target_in_support")):
        subset = holdout_metrics.loc[holdout_metrics["split"].eq(split)
                                     & holdout_metrics["segment"].eq(segment)].set_index("model")
        if {str(winner["model"]), "naive_last"} <= set(subset.index):
            selected_rmse = float(subset.loc[str(winner["model"]), "rmse"])
            naive_rmse = float(subset.loc["naive_last", "rmse"])
            gates[f"{split}:{segment}"] = {
                "n": int(subset.loc["naive_last", "n"]),
                "selected_model_rmse": selected_rmse,
                "naive_last_rmse": naive_rmse,
                "beats_naive_last": bool(selected_rmse < naive_rmse),
            }
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_predictions.to_csv(output_dir / "train_cv_predictions.csv", index=False)
    cv_metrics.to_csv(output_dir / "train_cv_metrics.csv", index=False)
    fold_metrics.to_csv(output_dir / "train_cv_fold_metrics.csv", index=False)
    holdout_predictions.to_csv(output_dir / "observed_price_predictions.csv", index=False)
    holdout_metrics.to_csv(output_dir / "observed_price_metrics.csv", index=False)
    by_sku.to_csv(output_dir / "observed_price_by_sku.csv", index=False)
    response.to_csv(output_dir / "price_response.csv", index=False)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": panel["dataset"],
        "prepared_manifest_sha256": _sha256(prepared_root / "manifest.json"),
        "prepared_weekly_sha256": _sha256(prepared_root / "weekly.csv"),
        "phase6_selected_decisions_sha256": phase6_sha,
        "protocol": "Three expanding train-only rolling origins; each evaluates 20 next weeks one step at observed prices",
        "cohort_selection_scope": (
            "Fixed 20 M5 SKU series selected using the full original train period; "
            "earlier rolling folds do not reselect products at their own origins"
        ),
        "folds": fold_metadata,
        "candidates": list(full_models),
        "selection": {
            "metric": "pooled train-CV RMSE on observed prices inside each fold's fit support",
            "model": str(winner["model"]), "rmse": float(winner["rmse"]),
            "naive_last_rmse": float(baseline["rmse"]),
            "n_in_support": int(winner["n"]),
            "beats_naive_last": bool(winner["rmse"] < baseline["rmse"]),
        },
        "selected_model_observed_price_audit": gates,
        "holdout_status": "Validation and test were inspected in earlier phases; scores here are exploratory and did not choose the Phase 7 model",
        "phase6_cohort_scope": (
            "phase6_pricing_eligible uses the previous-week reference inside train support "
            "(200 validation, 174 test); phase6_pricing_and_target_in_support additionally "
            "requires the target week's observed price inside train support; both are observed-price checks"
        ),
        "counterfactual_identified": False,
        "business_deployment_supported": False,
        "limits": [
            "Price-demand associations in M5 are not randomized price interventions",
            "Observed-price RMSE cannot validate demand or revenue at unchosen prices",
            "A flat price-response model is only a stress case, not an estimated elasticity",
            "Earlier phases inspected validation and test, so neither is a pristine new holdout",
            "Rolling CV is conditional on the 20-SKU cohort ranked over full train, so SKU selection looks ahead relative to early CV origins",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--phase6-dir", type=Path)
    args = parser.parse_args()
    result = run(args.prepared_root, args.output_dir, args.phase6_dir)
    print(json.dumps({"selection": result["selection"], "folds": result["folds"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
