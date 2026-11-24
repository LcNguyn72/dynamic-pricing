"""Score Song and shared baselines on exactly the same observed SKU-weeks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .forecast_benchmark import _score


def compare(common_dir: str | Path, song_dir: str | Path, output_dir: str | Path,
            *, seed: int = 42) -> pd.DataFrame:
    common_dir, song_dir, output_dir = Path(common_dir), Path(song_dir), Path(output_dir)
    common_run = json.loads((common_dir / "forecast_run.json").read_text(encoding="utf-8"))
    song_run = json.loads((song_dir / f"run_seed{seed}.json").read_text(encoding="utf-8"))
    if common_run["prepared_manifest_sha256"] != song_run["source_manifest_sha256"]:
        raise ValueError("Song and baseline use different prepared manifests")
    if common_run["prepared_windows_sha256"] != song_run["source_windows_sha256"]:
        raise ValueError("Song and baseline use different forecast windows")
    if common_run["dataset"] != song_run["dataset"]:
        raise ValueError("Dataset labels differ")
    if common_run["seed"] != seed or song_run["seed"] != seed:
        raise ValueError("Seed differs from requested comparison")
    if song_run["test_model_fit_scope"] != "train+validation":
        raise ValueError("Use --refit-train-validation for the Song forecast comparison")
    common = pd.read_csv(common_dir / "forecast_predictions.csv", dtype={"sku": str}, parse_dates=["week"])
    song = pd.read_csv(song_dir / f"predictions_seed{seed}.csv", dtype={"sku": str}, parse_dates=["week"])
    song = song.rename(columns={"qty": "y_true", "predicted_qty": "y_pred"})
    song["model"] = "song_transformer_price"
    shared_columns = ["model", "split", "sku", "week", "price", "y_true", "y_pred"]
    combined = pd.concat([common[shared_columns], song[shared_columns]], ignore_index=True)
    reference = song.loc[song["split"].eq("test"), ["sku", "week", "price", "y_true"]]
    reference = reference.sort_values(["sku", "week"]).reset_index(drop=True)
    if reference.empty or reference[["sku", "week"]].duplicated().any():
        raise ValueError("Song test prediction keys are missing or duplicated")
    metrics = []
    for model, group in combined.groupby("model", sort=True):
        test = group.loc[group["split"].eq("test")].sort_values(["sku", "week"]).reset_index(drop=True)
        if not test[["sku", "week"]].equals(reference[["sku", "week"]]):
            raise ValueError(f"Different observed test rows for {model}")
        if not np.allclose(test["y_true"], reference["y_true"], rtol=0, atol=1e-8):
            raise ValueError(f"Different test demand labels for {model}")
        if not np.allclose(test["price"], reference["price"], rtol=0, atol=1e-8):
            raise ValueError(f"Different observed test prices for {model}")
        score = _score(test.rename(columns={"y_true": "qty"}), test["y_pred"].to_numpy())
        metrics.append({
            "dataset": song_run["dataset"], "model": model, "split": "test", "seed": seed,
            "fit_scope": ("train+validation" if model != "naive_last" else "no fit"),
            **score,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(metrics).sort_values(["rmse", "model"]).reset_index(drop=True)
    frame.to_csv(output_dir / "comparison_metrics.csv", index=False)
    combined.sort_values(["model", "split", "sku", "week"]).to_csv(
        output_dir / "comparison_predictions.csv", index=False)
    (output_dir / "comparison_manifest.json").write_text(json.dumps({
        "dataset": song_run["dataset"],
        "source_manifest_sha256": song_run["source_manifest_sha256"],
        "source_windows_sha256": song_run["source_windows_sha256"],
        "seed": seed,
        "test_rows_per_model": len(reference),
        "song_best_epoch": song_run["best_epoch"],
        "song_test_fit_scope": song_run["test_model_fit_scope"],
        "metric_scale": "units per observed SKU-week, same test rows for all models",
        "limitation": "These are project cohort metrics, not the published Song/M5 competition scores",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-dir", type=Path, required=True)
    parser.add_argument("--song-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(compare(args.common_dir, args.song_dir, args.output_dir, seed=args.seed).to_string(index=False))


if __name__ == "__main__":
    main()
