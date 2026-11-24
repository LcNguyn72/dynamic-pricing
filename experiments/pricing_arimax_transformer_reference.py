"""Score the verified, train-only Step 2 Transformer on M5 validation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from . import phase5_training as training
from .phase1_finalize import _observed, _sha256
from .phase5_cuda_graph import _attention_scope
from .phase5_data import LOCK_PATH, LOOKBACK, PHASE1_ROOT, _KeyDigest, iter_weekly, iter_windows, load_contract
from .phase5_forecast import _atomic_json, _write_shard
from .phase5_verify import _check_sha, _csv, align_predictions, sufficient, summarize
from .pricing_transformer_tune import MODEL, OUTPUT as STEP2, PHASE5, encoder


OUTPUT = PHASE1_ROOT.with_name("pricing_arimax_v2") / "transformer_reference"


def validation_rows(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    weeks = pd.to_datetime(frame["week"], errors="raise")
    split_validation = frame["split"].eq("validation")
    if not split_validation.eq(weeks.between(pd.Timestamp(start), pd.Timestamp(end))).all():
        raise ValueError("Validation labels do not match the Phase 1 week bounds")
    return frame.loc[split_validation & _observed(frame)]


def complete_history(weekly: pd.DataFrame) -> pd.Series:
    valid = np.isfinite(weekly["qty"]) & np.isfinite(weekly["price"]) & weekly["price"].gt(0)
    by_sku = valid.groupby(weekly["sku"], sort=False)
    available = valid.copy()
    for lag in range(1, LOOKBACK + 1):
        available &= by_sku.shift(lag, fill_value=False)
    return available


def run(root: Path, step2: Path, cache: Path, output: Path, device: torch.device) -> dict:
    started = perf_counter()
    contract = load_contract(root, LOCK_PATH)
    root = contract["root"]
    audit_path = step2 / "m5_test_audit.json"
    audit = training._read_json(audit_path)
    checkpoint = step2 / "full_m5" / "song_transformer.pt"
    _check_sha(checkpoint, audit["checkpoint_sha256"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (audit["status"] != "complete" or audit["fit_scope"] != "train_only"
            or saved["dataset"] != "m5" or saved["fit_scope"] != "train_only"
            or saved["version"]["phase1_sha256"] != contract["phase1_sha256"]
            or saved["version"]["lock_sha256"] != contract["lock_sha256"]
            or audit["test_keys_sha256"] != contract["lock"]["frozen_test_keys"]["m5"]["sha256"]
            or saved["parameters"] != audit["settings"]):
        raise ValueError("Step 2 checkpoint does not match the frozen M5 contract")

    spec = contract["protocol"]["datasets"]["m5"]
    start, end = spec["validation_start"], spec["validation_end"]
    model = training._make_network(MODEL, {"lookback": saved["lookback"]}, saved["seed"], device)
    model.load_state_dict(saved["state_dict"])
    output.mkdir(parents=True, exist_ok=True)
    files, totals, series = {}, [], set()
    train_rows, train_qty, train_qty_squared = 0, 0.0, 0.0
    weekly_shards = iter_weekly(root, "m5", contract, cache_root=cache)
    window_shards = iter_windows(root, "m5", contract, cache_root=cache)
    for (store, weekly, _), (window_store, windows, scalers) in zip(weekly_shards, window_shards, strict=True):
        if store != window_store or any(saved["scalers"].get(sku) != scaler for sku, scaler in scalers.items()):
            raise ValueError(f"{store}: invalid window shard or mismatched train scalers")
        target = validation_rows(windows, start, end).sort_values(["sku", "week"])
        observed = validation_rows(weekly, start, end)
        truth = observed.loc[complete_history(weekly), ["sku", "week", "qty", "price"]].copy()
        truth.insert(0, "dataset", "m5")
        truth["week"] = truth["week"].dt.strftime("%Y-%m-%d")
        digest = _KeyDigest()
        for sku, week in target[["sku", "week"]].itertuples(index=False, name=None):
            digest.add(("m5", sku, week.strftime("%Y-%m-%d")))
        with _attention_scope(MODEL):
            forecast = training._model_eval(model, target, encoder(scalers, saved["lookback"]),
                                            scalers, device, song=True)
        path = output / f"validation_{store}.csv"
        meta = _write_shard(path, "m5", target, forecast)
        aligned = align_predictions(_csv(path), truth)
        totals.append(sufficient(aligned).sum())
        series.update(scalers)
        files[store] = {"file": path.name, **meta, "keys_sha256": digest.result()["sha256"],
                        "observed_rows_without_required_history": len(observed) - len(truth)}
        qty = weekly.loc[weekly["split"].eq("train") & _observed(weekly), "qty"].to_numpy(dtype=np.float64)
        train_rows += len(qty)
        train_qty += float(qty.sum())
        train_qty_squared += float(np.square(qty).sum())
        print(f"[Step 3 Transformer reference] {store}: {len(aligned)} verified validation keys", flush=True)

    if set(files) != set(spec["store_ids"]) or series != set(saved["scalers"]):
        raise ValueError("Validation run does not cover the Step 2 M5 cohort")
    mean = train_qty / train_rows
    std = math.sqrt(max(0.0, train_qty_squared / train_rows - mean * mean))
    metrics = summarize(pd.concat(totals, axis=1).sum(axis=1), train_mean=mean, train_std=std)
    report = {
        "status": "complete", "dataset": "m5", "split": "validation", "fit_scope": "train_only",
        "validation_start": start, "validation_end": end, "series": len(series), "seed": saved["seed"],
        "phase1_sha256": contract["phase1_sha256"], "lock_sha256": contract["lock_sha256"],
        "checkpoint_sha256": audit["checkpoint_sha256"], "step2_audit_sha256": _sha256(audit_path),
        "n_train_normalization": train_rows, "train_qty_mean": mean, "train_qty_std": std,
        "settings": saved["parameters"], "metrics": metrics, "prediction_files": files,
        "device": str(device), "wall_seconds": perf_counter() - started,
    }
    _atomic_json(output / "validation_run.json", report)
    print(f"[Step 3 Transformer reference] {metrics['n']} validation keys, RMSE {metrics['rmse']:.4f}, "
          f"{report['wall_seconds']:.1f} s", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--step2-dir", type=Path, default=STEP2)
    parser.add_argument("--cache-dir", type=Path, default=PHASE5 / "data_cache")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    run(args.phase1_root, args.step2_dir, args.cache_dir, args.output_dir, training.resolve_device(args.device))


if __name__ == "__main__":
    main()
