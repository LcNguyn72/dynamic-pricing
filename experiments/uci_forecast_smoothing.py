"""Diagnose and compare simple UCI forecasts on the frozen V2 test keys."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .forecast_feature_ablation import ARTIFACTS, _json, metric_totals, scores
from .phase1_finalize import _observed
from .phase5_data import PHASE1_ROOT, iter_windows, load_contract


def evaluate(root, output, cache):
    contract = load_contract(root)
    _, windows, _ = next(iter_windows(root, "uci", contract, cache_root=cache))
    history = windows[[f"qty_lag_{lag}" for lag in range(1, 9)]].astype(float)
    last = history.iloc[:, 0]
    mean_4 = history.iloc[:, :4].mean(axis=1)
    observed_mean_4 = history.iloc[:, :4].where(history.iloc[:, :4].gt(0)).mean(axis=1).fillna(last)
    observed_median_8 = history.where(history.gt(0)).median(axis=1).fillna(last)
    candidates = {
        "previous_week": last,
        "mean_4": mean_4,
        "last_50_mean_4_50": 0.5 * last + 0.5 * mean_4,
        "observed_mean_4": observed_mean_4,
        "observed_median_8": observed_median_8,
        "last_50_observed_median_8_50": 0.5 * last + 0.5 * observed_median_8,
    }
    observed = _observed(windows)
    result = []
    for split in ("validation", "test"):
        mask = observed & windows["split"].eq(split)
        result.extend({"split": split, "variant": name,
                       **scores(metric_totals(windows.loc[mask, "qty"], prediction[mask]))}
                      for name, prediction in candidates.items())
    metrics = pd.DataFrame(result)
    valid = metrics.loc[metrics["split"].eq("validation")]
    winners = {name: valid.loc[valid[name].idxmin(), "variant"] for name in ("rmse", "mae")}
    baseline = windows.loc[observed & windows["split"].eq("test"), ["sku", "week", "qty"]].copy()
    baseline["predicted_qty"] = last.loc[baseline.index]
    baseline["abs_error"] = (baseline["qty"] - baseline["predicted_qty"]).abs()
    baseline["squared_error"] = baseline["abs_error"].pow(2)
    extreme = baseline.nlargest(int(np.ceil(0.01 * len(baseline))), "squared_error")
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "uci_baseline_exploration.csv", index=False)
    extreme.to_csv(output / "uci_largest_baseline_errors.csv", index=False)
    _json(output / "uci_diagnostic.json", {
        "test_keys_sha256": contract["phase1"]["uci"]["test_keys"]["sha256"],
        "selected_by_validation": winners,
        "top_error_rows": len(extreme),
        "top_error_share_of_squared_error": float(extreme["squared_error"].sum() / baseline["squared_error"].sum()),
        "top_error_share_of_absolute_error": float(extreme["abs_error"].sum() / baseline["abs_error"].sum()),
    })
    print(metrics.to_string(index=False))
    print(f"Selected on validation: {winners}")
    print(f"Top 1% contributes {extreme['squared_error'].sum() / baseline['squared_error'].sum():.2%} of squared error")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=ARTIFACTS / "phase5_v2" / "data_cache")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "forecast_feature_ablation_v2")
    args = parser.parse_args()
    evaluate(args.prepared_root, args.output_dir, args.cache_dir)


if __name__ == "__main__":
    main()
