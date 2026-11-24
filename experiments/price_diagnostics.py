"""Audit learned price gradients on held-out observed SKU-weeks.

This is a model sensitivity check, not a causal estimate of price elasticity.
Historical price changes coincide with many unobserved business conditions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEFAULT_SELECTED_PATH, DEFAULT_WEEKLY_PATH, load_experiment_data, make_windows
from .forecast_torch import load_price_forecaster
from .pricing_simulator import KernelDemandSimulator


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "artifacts" / "online_retail_ii_20" / "forecast" / "models"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "online_retail_ii_20" / "price_diagnostics"


def evaluate_price_sensitivity(
    windows: pd.DataFrame,
    model_dir: str | Path,
    output_dir: str | Path,
    simulator: KernelDemandSimulator | None = None,
) -> tuple[pd.DataFrame, dict]:
    model = load_price_forecaster(model_dir, train_only=True)
    test = windows.loc[windows["split"].eq("test") & windows["observed_sales"].astype(bool)].copy()
    if test.empty:
        raise ValueError("No observed test SKU-weeks")
    records = []
    for row in test.itertuples(index=False):
        item = pd.Series(row._asdict())
        price = float(item["price"])
        old_price = float(item["price_lag_1"])
        q_hat = model.predict_q(item, price)
        gradient = model.price_gradient(item, price)
        old_price_q_hat = model.predict_q(item, old_price)
        log_price_change = float(np.log(price / old_price))
        simulator_elasticity = np.nan
        if simulator is not None and str(item["sku"]) in simulator.surfaces:
            surface = simulator.surfaces[str(item["sku"])]
            lo = max(surface.price_min, price * 0.99)
            hi = min(surface.price_max, price * 1.01)
            if hi > lo:
                q_lo = simulator.predict_mean(str(item["sku"]), lo, item["week"])
                q_hi = simulator.predict_mean(str(item["sku"]), hi, item["week"])
                simulator_elasticity = float((np.log(q_hi + 1e-9) - np.log(q_lo + 1e-9)) /
                                             (np.log(hi) - np.log(lo)))
        records.append({
            "sku": str(item["sku"]), "week": item["week"], "tier": item.get("tier", ""),
            "observed_price": price, "observed_qty": float(item["qty"]),
            "predicted_qty": q_hat, "price_gradient": gradient,
            "local_elasticity_proxy": gradient * price / max(q_hat, 1e-6),
            "simulator_local_elasticity": simulator_elasticity,
            "log_price_change": log_price_change,
            "observed_log_qty_change": float(np.log1p(item["qty"]) - np.log1p(item["qty_lag_1"])),
            "predicted_log_qty_change_from_price_only": float(np.log1p(q_hat) - np.log1p(old_price_q_hat)),
            "substantial_price_change": abs(log_price_change) >= 0.10,
        })
    frame = pd.DataFrame(records)
    events = frame.loc[frame["substantial_price_change"]]
    corr = (float(events[["observed_log_qty_change", "predicted_log_qty_change_from_price_only"]].corr().iloc[0, 1])
            if len(events) >= 3 else None)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_rows": int(len(frame)),
        "price_change_events_abs_log_ge_0_10": int(len(events)),
        "negative_gradient_fraction": float(frame["price_gradient"].lt(0).mean()),
        "median_price_gradient_units_per_price_unit": float(frame["price_gradient"].median()),
        "median_local_elasticity_proxy": float(frame["local_elasticity_proxy"].median()),
        "median_simulator_local_elasticity": (
            float(frame["simulator_local_elasticity"].median()) if simulator is not None else None
        ),
        "forecaster_simulator_gradient_sign_agreement_fraction": (
            float((np.sign(frame["local_elasticity_proxy"]) ==
                   np.sign(frame["simulator_local_elasticity"])).mean()) if simulator is not None else None
        ),
        "event_response_correlation_observational": corr if corr is not None and np.isfinite(corr) else None,
        "interpretation": "A gradient of the trained predictor, not a causal price effect. Event comparison is confounded by time, promotions and selection.",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "price_sensitivity.csv", index=False)
    with (output / "price_sensitivity_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    return frame, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekly-path", type=Path, default=DEFAULT_WEEKLY_PATH)
    parser.add_argument("--selected-path", type=Path, default=DEFAULT_SELECTED_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    weekly, manifest = load_experiment_data(args.weekly_path, args.selected_path)
    simulator = KernelDemandSimulator().fit(weekly)
    _, summary = evaluate_price_sensitivity(make_windows(weekly, manifest["lookback_weeks"]),
                                            args.model_dir, args.output_dir, simulator)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
