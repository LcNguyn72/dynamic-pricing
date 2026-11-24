"""Run selected forecast baselines on one audited, frozen research panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .audit_prepared import audit
from .forecast_benchmark import DEFAULT_MODELS, run_forecast_benchmark


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(prepared_root: str | Path, output_dir: str | Path, *, models: tuple[str, ...],
        seed: int = 42, torch_epochs: int = 25) -> dict:
    prepared_root, output_dir = Path(prepared_root), Path(output_dir)
    source_summary = audit(prepared_root)
    manifest_path, windows_path = prepared_root / "manifest.json", prepared_root / "windows.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prepared_manifest_sha256"] = _sha256(manifest_path)
    manifest["prepared_windows_sha256"] = _sha256(windows_path)
    windows = pd.read_csv(windows_path, dtype={"sku": str}, parse_dates=["week"])
    metrics, predictions = run_forecast_benchmark(
        windows, output_dir, manifest, models=models, seed=seed,
        torch_epochs=torch_epochs, strict=True,
    )
    return {
        "source": source_summary,
        "models": list(models),
        "test_rows_per_model": int(predictions.loc[predictions["split"].eq("test")]
                                   .groupby("model").size().min()),
        "test_metrics": metrics.loc[metrics["split"].eq("test"),
                                    ["model", "n", "rmse", "mae"]].to_dict(orient="records"),
        "output_dir": str(output_dir.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS,
                        default=["naive_last", "xgboost", "xgboost_price"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-epochs", type=int, default=25)
    args = parser.parse_args()
    result = run(args.prepared_root, args.output_dir, models=tuple(args.models),
                 seed=args.seed, torch_epochs=args.torch_epochs)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
