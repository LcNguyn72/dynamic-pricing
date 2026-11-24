"""Select a lag-adjusted M5 simulator on validation, then inspect test once."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audit_prepared import audit as audit_panel
from .forecast_benchmark import _score
from .pricing_benchmark import validate_simulator
from .pricing_simulator import KernelDemandSimulator, LagAdjustedKernelSimulator


CANDIDATE_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
RATIO_LIMIT = 2.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scoring_rows(simulator: KernelDemandSimulator, weekly: pd.DataFrame,
                  split: str) -> pd.DataFrame:
    predicted = validate_simulator(simulator, weekly, splits=(split,))
    lag = weekly.sort_values(["sku", "week"])[["sku", "week", "qty"]].copy()
    lag["last_qty"] = lag.groupby("sku")["qty"].shift(1)
    predicted = predicted.merge(lag[["sku", "week", "last_qty"]],
                                on=["sku", "week"], how="left", validate="one_to_one")
    if predicted["last_qty"].isna().any():
        raise ValueError("A scored week lacks its previous demand")
    bounds = simulator.diagnostics().set_index("sku")
    predicted["price_in_train_support"] = (
        predicted["price"].ge(predicted["sku"].map(bounds["price_min"]) - 1e-8)
        & predicted["price"].le(predicted["sku"].map(bounds["price_max"]) + 1e-8)
    )
    return predicted


def _scores(rows: pd.DataFrame, split: str, model: str) -> list[dict]:
    output = []
    for segment, subset in (("all", rows),
                            ("in_support", rows.loc[rows["price_in_train_support"]]),
                            ("out_of_support", rows.loc[~rows["price_in_train_support"]])):
        if subset.empty:
            continue
        column = "last_qty" if model == "naive_last" else "simulator_pred_qty"
        output.append({"split": split, "segment": segment, "model": model,
                       **_score(subset, subset[column].to_numpy())})
    return output


def _price_response(simulator: KernelDemandSimulator, weekly: pd.DataFrame) -> pd.DataFrame:
    training = weekly.loc[weekly["split"].eq("train") & weekly["observed_sales"]]
    rows = []
    for sku, group in training.groupby("sku"):
        surface = simulator.surfaces.get(str(sku))
        if surface is None:
            continue
        price = float(group["price"].median())
        low = max(surface.price_min, price * 0.99)
        high = min(surface.price_max, price * 1.01)
        if high <= low:
            continue
        week = group.iloc[-1]["week"]
        context = {"event_days": group.iloc[-1].get("event_days", 0),
                   "snap_days": group.iloc[-1].get("snap_days", 0),
                   "qty_lag_1": float(group["qty"].median()),
                   "price_lag_1": price}
        demand_low = simulator.predict_mean(str(sku), low, week, context)
        demand_high = simulator.predict_mean(str(sku), high, week, context)
        rows.append({"sku": sku, "reference_price": price, "price_low": low,
                     "price_high": high, "demand_low": demand_low,
                     "demand_high": demand_high,
                     "demand_increases_with_price": bool(demand_high > demand_low)})
    return pd.DataFrame(rows)


def run(prepared_root: str | Path, output_dir: str | Path,
        *, candidate_weights: tuple[float, ...] = CANDIDATE_WEIGHTS) -> dict:
    prepared_root, output_dir = Path(prepared_root), Path(output_dir)
    panel = audit_panel(prepared_root)
    if not panel["dataset"].startswith("M5"):
        raise ValueError("Simulator selection is only defined for M5")
    if not candidate_weights or len(set(candidate_weights)) != len(candidate_weights):
        raise ValueError("Provide distinct candidate weights")
    weekly = pd.read_csv(prepared_root / "weekly.csv", dtype={"sku": str}, parse_dates=["week"])
    candidates = []
    for weight in candidate_weights:
        simulator = LagAdjustedKernelSimulator(weight, ratio_limit=RATIO_LIMIT).fit(weekly)
        validation = _scoring_rows(simulator, weekly, "validation")
        for row in _scores(validation, "validation", f"lag_kernel_{weight:g}"):
            candidates.append({"recent_weight": weight, **row})
    candidate_frame = pd.DataFrame(candidates)
    support = candidate_frame.loc[candidate_frame["segment"].eq("in_support")]
    chosen = support.sort_values(["rmse", "recent_weight"]).iloc[0]
    weight = float(chosen["recent_weight"])
    selected = LagAdjustedKernelSimulator(weight, ratio_limit=RATIO_LIMIT).fit(weekly)
    base = KernelDemandSimulator().fit(weekly)
    scores = []
    for split in ("validation", "test"):
        for name, simulator in (("song_kernel", base), ("selected_lag_kernel", selected)):
            rows = _scoring_rows(simulator, weekly, split)
            scores.extend(_scores(rows, split, name))
            if name == "song_kernel":
                scores.extend(_scores(rows, split, "naive_last"))
    metric_frame = pd.DataFrame(scores)
    on_validation = metric_frame.loc[metric_frame["split"].eq("validation")
                                     & metric_frame["segment"].eq("in_support")].set_index("model")
    gate = bool(on_validation.loc["selected_lag_kernel", "rmse"]
                <= on_validation.loc["naive_last", "rmse"])
    response = _price_response(selected, weekly)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_frame.to_csv(output_dir / "validation_candidates.csv", index=False)
    metric_frame.to_csv(output_dir / "observed_price_metrics.csv", index=False)
    response.to_csv(output_dir / "price_response.csv", index=False)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": panel["dataset"],
        "prepared_manifest_sha256": _sha256(prepared_root / "manifest.json"),
        "prepared_weekly_sha256": _sha256(prepared_root / "weekly.csv"),
        "simulator_variant": "lag_adjusted_kernel",
        "recent_weight": weight,
        "ratio_limit": RATIO_LIMIT,
        "candidate_weights": list(candidate_weights),
        "selection_metric": "validation RMSE at observed prices inside train price support",
        "necessary_validation_gate": "selected simulator RMSE <= last-week naive RMSE on the same validation rows",
        "necessary_validation_gate_passed": gate,
        "test_scope_note": "Test is an exploratory held-out period; earlier work already inspected its Song-kernel error. No candidate was chosen using test scores in this run.",
        "counterfactual_limit": "Observed-price checks and price-curve shape do not identify causal demand at untried prices",
        "price_response_positive_skus": int(response["demand_increases_with_price"].sum()),
        "price_response_checked_skus": len(response),
    }
    (output_dir / "selected_simulator.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.prepared_root, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
