"""Repeat the common baseline/Song forecast comparison across fixed seeds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from .common_forecast import run as run_common
from .compare_forecasts import compare
from .song_forecast import run as run_song


DEFAULT_SEEDS = (42, 123, 456, 789, 1024)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(prepared_root: str | Path, output_root: str | Path,
        *, seeds: tuple[int, ...] = DEFAULT_SEEDS, epochs: int = 50,
        models: tuple[str, ...] = ("naive_last", "xgboost", "xgboost_price")) -> pd.DataFrame:
    prepared_root, output_root = Path(prepared_root), Path(output_root)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct seeds")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    all_scores = []
    for seed in seeds:
        seed_root = output_root / f"seed{seed}"
        common_dir, song_dir, comparison_dir = (
            seed_root / "common", seed_root / "song", seed_root / "comparison"
        )
        print(f"[multi-seed] starting seed {seed}", flush=True)
        run_common(prepared_root, common_dir, models=models, seed=seed)
        run_song(prepared_root, song_dir, seed=seed, epochs=epochs,
                 refit_train_validation=True)
        all_scores.append(compare(common_dir, song_dir, comparison_dir, seed=seed))
        print(f"[multi-seed] completed seed {seed}", flush=True)
    scores = pd.concat(all_scores, ignore_index=True)
    aggregate = scores.groupby(["dataset", "model", "split"], as_index=False).agg(
        runs=("seed", "nunique"),
        test_rows=("n", "first"),
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
    ).sort_values("rmse_mean")
    output_root.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_root / "all_runs.csv", index=False)
    aggregate.to_csv(output_root / "aggregate.csv", index=False)
    (output_root / "run_manifest.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_root": str(prepared_root.resolve()),
        "prepared_manifest_sha256": _sha256(prepared_root / "manifest.json"),
        "prepared_windows_sha256": _sha256(prepared_root / "windows.csv"),
        "seeds": list(seeds), "epochs_max": epochs, "baseline_models": list(models),
        "song_test_fit_scope": "train+validation",
        "song_pricing_checkpoint_fit_scope": "train",
        "comparison_rule": "Identical observed SKU-week test keys, price and demand labels per seed",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--models", nargs="+", default=["naive_last", "xgboost", "xgboost_price"])
    args = parser.parse_args()
    print(run(args.prepared_root, args.output_root, seeds=tuple(args.seeds),
              epochs=args.epochs, models=tuple(args.models)).to_string(index=False))


if __name__ == "__main__":
    main()
