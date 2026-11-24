"""Complete Song's available forecast baselines on the frozen UCI/M5 panels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common_forecast import run as run_common
from .compare_forecasts import compare


DEEP_MODELS = ("lstm", "lstm_price", "transformer", "transformer_price")
FIRST_MODELS = ("arima", *DEEP_MODELS)
SEEDS = (42, 123, 456, 789, 1024)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_cached_run(path: Path, *, seed: int, expected: tuple[str, ...],
                      manifest_hash: str, windows_hash: str) -> bool:
    record = path / "forecast_run.json"
    if not record.is_file():
        return False
    data = json.loads(record.read_text(encoding="utf-8"))
    return (
        data.get("seed") == seed
        and data.get("prepared_manifest_sha256") == manifest_hash
        and data.get("prepared_windows_sha256") == windows_hash
        and tuple(data.get("completed_models", [])) == expected
        and (path / "forecast_predictions.csv").is_file()
    )


def _per_sku_scores(predictions: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    test = predictions.loc[predictions["split"].eq("test")]
    for (model, sku), group in test.groupby(["model", "sku"]):
        actual = group["y_true"].to_numpy(dtype=float)
        forecast = group["y_pred"].to_numpy(dtype=float)
        rows.append({
            "seed": seed, "model": model, "sku": sku, "n": len(group),
            "rmse": float(np.sqrt(np.mean((forecast - actual) ** 2))),
            "mae": float(np.mean(np.abs(forecast - actual))),
        })
    return pd.DataFrame(rows)


def run(prepared_root: str | Path, previous_multiseed_root: str | Path,
        output_root: str | Path, *, seeds: tuple[int, ...] = SEEDS,
        torch_epochs: int = 25) -> pd.DataFrame:
    prepared_root = Path(prepared_root)
    previous_multiseed_root = Path(previous_multiseed_root)
    output_root = Path(output_root)
    if not seeds or len(set(seeds)) != len(seeds) or seeds[0] != 42:
        raise ValueError("Use distinct seeds with 42 first for the deterministic ARIMA run")
    manifest_hash = _sha256(prepared_root / "manifest.json")
    windows_hash = _sha256(prepared_root / "windows.csv")
    previous = json.loads((previous_multiseed_root / "run_manifest.json").read_text(encoding="utf-8"))
    if (previous["prepared_manifest_sha256"] != manifest_hash
            or previous["prepared_windows_sha256"] != windows_hash):
        raise ValueError("Earlier Song/XGBoost results use a different prepared panel")
    old_scores = pd.read_csv(previous_multiseed_root / "all_runs.csv")
    additional_scores = []
    per_sku = []
    for seed in seeds:
        common_dir = output_root / f"seed{seed}" / "common"
        comparison_dir = output_root / f"seed{seed}" / "comparison"
        expected = FIRST_MODELS if seed == 42 else DEEP_MODELS
        if not _valid_cached_run(common_dir, seed=seed, expected=expected,
                                 manifest_hash=manifest_hash, windows_hash=windows_hash):
            run_common(prepared_root, common_dir, models=expected, seed=seed,
                       torch_epochs=torch_epochs)
        song_dir = previous_multiseed_root / f"seed{seed}" / "song"
        scores = compare(common_dir, song_dir, comparison_dir, seed=seed)
        additional_scores.append(scores.loc[~scores["model"].eq("song_transformer_price")])
        new_predictions = pd.read_csv(comparison_dir / "comparison_predictions.csv",
                                      dtype={"sku": str})
        per_sku.append(_per_sku_scores(new_predictions, seed))
        old_predictions = pd.read_csv(previous_multiseed_root / f"seed{seed}" / "common"
                                      / "forecast_predictions.csv", dtype={"sku": str})
        per_sku.append(_per_sku_scores(old_predictions, seed))
        print(f"[phase3c] completed seed {seed}", flush=True)

    scores = pd.concat([old_scores, *additional_scores], ignore_index=True)
    if scores[["model", "seed", "split"]].duplicated().any():
        raise ValueError("A model/seed/split appears twice in Phase 3C")
    aggregate = scores.groupby(["dataset", "model", "split"], as_index=False).agg(
        runs=("seed", "nunique"), test_rows=("n", "first"),
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
    ).sort_values("rmse_mean")
    sku_runs = pd.concat(per_sku, ignore_index=True)
    if sku_runs[["model", "sku", "seed"]].duplicated().any():
        raise ValueError("Duplicate per-SKU model/seed metrics")
    sku_summary = sku_runs.groupby(["model", "sku"], as_index=False).agg(
        runs=("seed", "nunique"), n=("n", "first"),
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"),
    )
    effect = []
    for base, priced in (("xgboost", "xgboost_price"), ("lstm", "lstm_price"),
                         ("transformer", "transformer_price")):
        plain = sku_summary.loc[sku_summary["model"].eq(base), ["sku", "rmse_mean"]]
        with_price = sku_summary.loc[sku_summary["model"].eq(priced), ["sku", "rmse_mean"]]
        merged = plain.merge(with_price, on="sku", suffixes=("_plain", "_price"), validate="one_to_one")
        effect.append({
            "model_pair": f"{base} / {priced}",
            "skus": len(merged),
            "skus_improved_with_price": int((merged["rmse_mean_price"] < merged["rmse_mean_plain"]).sum()),
            "mean_per_sku_rmse_change_price_minus_plain": float(
                (merged["rmse_mean_price"] - merged["rmse_mean_plain"]).mean()),
            "overall_rmse_change_price_minus_plain": float(
                aggregate.loc[aggregate["model"].eq(priced), "rmse_mean"].iloc[0]
                - aggregate.loc[aggregate["model"].eq(base), "rmse_mean"].iloc[0]),
        })
    output_root.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_root / "all_runs.csv", index=False)
    aggregate.to_csv(output_root / "aggregate.csv", index=False)
    sku_summary.to_csv(output_root / "per_sku.csv", index=False)
    pd.DataFrame(effect).to_csv(output_root / "price_feature_effect.csv", index=False)
    (output_root / "run_manifest.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_manifest_sha256": manifest_hash,
        "prepared_windows_sha256": windows_hash,
        "seeds": list(seeds),
        "arima_seeds": [42],
        "torch_epochs_max": torch_epochs,
        "forecast_scope": "one-step observed SKU-week test rows; deep models refit train+validation",
        "tft_informer_status": "Not run: no validated implementations installed in the CPU-only 16.8 GB environment; their published scores remain literature context only",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--previous-multiseed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--torch-epochs", type=int, default=25)
    args = parser.parse_args()
    print(run(args.prepared_root, args.previous_multiseed_root, args.output_root,
              seeds=tuple(args.seeds), torch_epochs=args.torch_epochs).to_string(index=False))


if __name__ == "__main__":
    main()
