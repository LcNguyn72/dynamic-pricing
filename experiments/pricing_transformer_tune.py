"""Step 2: tune Song Transformer on M5 validation and audit a train-only full fit."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import phase5_training as training
from .phase1_finalize import _observed, _sha256
from .phase2_song import m5_store_windows
from .phase5_cuda_graph import MODE, _attention_scope, fit_graph
from .phase5_data import LOCK_PATH, PHASE1_ROOT, iter_windows, load_contract, load_scalers
from .phase5_forecast import _write_shard
from .phase5_verify import _check_sha, _csv, align_predictions, prepare_reference, sufficient, summarize
from .song_forecast import CURRENT_FEATURES, HISTORY_FEATURES, _scalers, feature_arrays


OUTPUT = PHASE1_ROOT.with_name("pricing_transformer_tune_v2")
PHASE5 = PHASE1_ROOT.with_name("phase5_v2")
MODEL = "song_transformer"
LOOKBACKS = (8, 12)
RATES = (0.0003, 0.0005, 0.001)
SCREEN_SEEDS = (42, 123)
FULL_SEED = 42
EPOCHS = 4
BATCH = 64
THREADS = 4


def train_rows(windows: pd.DataFrame) -> pd.DataFrame:
    return windows.loc[windows["split"].eq("train") & _observed(windows)]


def choose_config(trials: pd.DataFrame, required_seeds: tuple[int, ...]) -> dict:
    eligible = []
    for (lookback, rate, epoch), group in trials.groupby(["lookback", "learning_rate", "epoch"]):
        if (len(group) != len(required_seeds) or set(group["seed"]) != set(required_seeds)
                or not group["valid_curve"].all() or not np.isfinite(group["rmse"]).all()):
            continue
        eligible.append({
            "lookback": int(lookback), "learning_rate": float(rate), "best_epoch": int(epoch),
            "batch_size": BATCH, "validation_mean_rmse": float(group["rmse"].mean()),
            "validation_mean_mae": float(group["mae"].mean()) if "mae" in group else None,
        })
    if not eligible:
        raise ValueError("No configuration completed the validation and price-response checks")
    return min(eligible, key=lambda row: (row["validation_mean_rmse"], row["best_epoch"], row["lookback"],
                                          row["learning_rate"]))


def encoder(scalers, lookback):
    def encode(rows):
        history, current = feature_arrays(rows, scalers, lookback=lookback)
        return history, current, np.zeros(len(rows), dtype=np.int64)

    return encode


def tensor_shard(directory, shard, rows, encode, scalers):
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / f"{shard}.json"
    if marker.is_file():
        report = training._read_json(marker)
        if report["n"] != len(rows):
            raise ValueError(f"{shard}: training cache row count changed")
        for name, sha in report["files"].items():
            _check_sha(directory / f"{shard}_{name}.npy", sha)
        return report
    return training._materialize_tensors(directory, shard, rows, encode, scalers, song=True)


def fit(directory, tensors, records, settings, identity, seed, device):
    directory.mkdir(parents=True, exist_ok=True)
    function = fit_graph if device.type == "cuda" else training._fit_neural
    return function(MODEL, seed, directory, settings, device, {"lookback": settings["lookback"]},
                    tensors, records, identity)


def validation_data(root, contract):
    settings = contract["lock"]["pricing"]
    train_skus = set(settings["pricing_demand_training_skus"])
    pricing_skus = {sku.rsplit("@", 1)[1]: sku for sku in settings["cohort_skus"]}
    windows, scalers, cases = {}, {}, []
    for store, report in contract["m5_stores"].items():
        sku = pricing_skus[store]
        count = sum(item.endswith("@" + store) for item in train_skus)
        frame, local, _ = m5_store_windows(root / "m5", report, sku, count)
        if set(local) != {item for item in train_skus if item.endswith("@" + store)}:
            raise ValueError(f"{store}: screening cohort differs from the locked pricing models")
        train = train_rows(frame)
        bounds = train.loc[train["sku"].eq(sku), "price"]
        lo, hi = float(bounds.min()), float(bounds.max())
        validation = frame.loc[frame["split"].eq("validation")]
        candidates = validation.loc[validation["sku"].eq(sku)].sort_values("week")
        if lo >= hi or candidates.empty:
            raise ValueError(f"{store}: missing validation rows or observed train price variation")
        positions = np.linspace(0, len(candidates) - 1, min(5, len(candidates)), dtype=int)
        cases.append(candidates.iloc[positions].assign(price_min=lo, price_max=hi))
        windows[store] = (train, validation)
        scalers.update(local)
    return windows, scalers, pd.concat(cases, ignore_index=True)


def price_response(model, cases, scalers, lookback, device):
    model.eval()
    limits = cases[["price_min", "price_max"]].to_numpy(dtype=np.float64)
    midpoint = limits.mean(axis=1)
    step = np.minimum(0.001, (limits[:, 1] - limits[:, 0]) / 1000)
    grid = np.linspace(limits[:, 0], limits[:, 1], 9, axis=1)
    probes = np.column_stack((grid, midpoint - step, midpoint + step))
    rows = cases.iloc[np.repeat(np.arange(len(cases)), probes.shape[1])]
    history, current = feature_arrays(rows, scalers, probes.ravel(), lookback)
    q_mean, q_std, _, p_std = _scalers(rows, scalers)
    x = torch.from_numpy(history).to(device)
    p = torch.from_numpy(current).to(device).requires_grad_()
    with _attention_scope(MODEL):
        z = model(x, p)
        gradient_z = torch.autograd.grad(z.sum(), p)[0][:, 0].detach().cpu().numpy()
    raw_qty = z.detach().cpu().numpy() * q_std + q_mean
    quantity = np.maximum(0, raw_qty).reshape(probes.shape)
    slope = np.where(raw_qty > 0, gradient_z * q_std / p_std, 0).reshape(probes.shape)
    if not np.isfinite(quantity).all() or not np.isfinite(slope).all():
        return {"valid_curve": False}, pd.DataFrame()
    spans = np.ptp(quantity[:, :9], axis=1)
    numerical = (quantity[:, -1] - quantity[:, -2]) / (2 * step)
    learned = slope[:, 4]
    rows = cases[["sku", "week", "price", "price_min", "price_max"]].copy()
    rows["gradient_at_midpoint"] = learned
    rows["finite_difference_at_midpoint"] = numerical
    rows["quantity_span"] = spans
    rows["price_grid"] = [",".join(map(str, values)) for values in grid]
    rows["prediction_grid"] = [",".join(map(str, values)) for values in quantity[:, :9]]
    return {
        "valid_curve": bool(np.any(spans > 0.001)),
        "price_cases": len(cases),
        "variable_price_cases": int(np.sum(spans > 0.001)),
        "median_qty_span": float(np.median(spans)),
        "negative_midpoint_gradient_pct": float(100 * np.mean(learned < 0)),
        "gradient_abs_p95": float(np.quantile(np.abs(learned), 0.95)),
        "median_abs_gradient_check_error": float(np.median(np.abs(learned - numerical))),
    }, rows


def screen(root, output, contract, identity, device):
    windows, scalers, cases = validation_data(root, contract)
    validation = pd.concat([val for _, val in windows.values()], ignore_index=True)
    print(f"[Step 2 screen] {len(scalers)} series, {len(validation)} validation keys, {len(cases)} price cases",
          flush=True)
    records, all_curves = [], []
    for length in LOOKBACKS:
        encode = encoder(scalers, length)
        tensors = output / "screen_tensors" / str(length)
        shards = {shard: tensor_shard(tensors, shard, train, encode, scalers)
                  for shard, (train, _) in windows.items()}
        for rate, seed in itertools.product(RATES, SCREEN_SEEDS):
            trial = output / "screen_runs" / f"{length}_{rate}_{seed}"
            version = {**identity, "scope": "screen_train_only", "lookback": length, "learning_rate": rate,
                       "batch_size": BATCH, "seed": seed}
            for epoch in range(1, EPOCHS + 1):
                metadata = trial / f"epoch_{epoch}.json"
                curves_path = trial / f"curves_{epoch}.csv"
                if not metadata.is_file():
                    settings = {"lookback": length, "learning_rate": rate, "best_epoch": epoch, "batch_size": BATCH}
                    model, timing = fit(trial, tensors, shards, settings, version, seed, device)
                    with _attention_scope(MODEL):
                        predicted = training._model_eval(model, validation, encode, scalers, device, song=True)
                    if not np.isfinite(predicted).all():
                        raise FloatingPointError("Nonfinite validation forecast")
                    metrics = summarize(sufficient(validation.assign(predicted_qty=predicted)).sum(),
                                        train_mean=0.0, train_std=0.0)
                    diagnostic, curves = price_response(model, cases, scalers, length, device)
                    curves.to_csv(curves_path, index=False)
                    report = {**metrics, **diagnostic, "lookback": length, "learning_rate": rate,
                              "seed": seed, "epoch": epoch, "fit_seconds": timing["fit_seconds"],
                              "identity": version}
                    training._write_json(metadata, report)
                    del model
                    print(f"[Step 2 screen] lookback={length} lr={rate} seed={seed} epoch={epoch}: "
                          f"RMSE {metrics['rmse']:.4f}", flush=True)
                entry = training._read_json(metadata)
                if entry.pop("identity") != version:
                    raise ValueError(f"Invalid screening checkpoint identity: {metadata}")
                records.append(entry)
                if entry["valid_curve"]:
                    all_curves.append(pd.read_csv(curves_path).assign(
                        lookback=length, learning_rate=rate, seed=seed, epoch=epoch))
    trials = pd.DataFrame(records)
    trials.to_csv(output / "validation_trials.csv", index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(output / "validation_price_curves.csv", index=False)
    selected = choose_config(trials, SCREEN_SEEDS)
    baseline = trials.loc[(trials["lookback"] == 12) & (trials["learning_rate"] == 0.001)
                          & (trials["epoch"] == 3)]
    record = {
        "selection_split": "validation", "fit_scope": "train_only", "screen_skus": len(scalers),
        "screen_train_rows": sum(len(train) for train, _ in windows.values()),
        "screen_validation_rows": len(validation), "screen_seeds": list(SCREEN_SEEDS),
        "price_cases": len(cases), "selection_rule": "minimum mean validation micro RMSE with finite non-flat price checks",
        "original_settings_validation_rmse": float(baseline["rmse"].mean()),
        "original_settings_validation_mae": float(baseline["mae"].mean()),
        "selected": selected, "identity": identity,
    }
    training._write_json(output / "selection.json", record)
    return selected


def full_fit(root, output, cache, contract, identity, device, settings):
    scalers = load_scalers(root, "m5", contract)
    encode = encoder(scalers, settings["lookback"])
    full = output / "full_m5"
    full.mkdir(exist_ok=True)
    path = full / "song_transformer.pt"
    experiment = {**identity, "scope": "full_m5_train_only", "seed": FULL_SEED, "settings": settings}
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["version"] != experiment or saved["scalers"] != scalers:
            raise ValueError("Full M5 checkpoint belongs to a different tune")
    else:
        tensors = full / "train_tensors"
        shards = {}
        for shard, windows, local in iter_windows(root, "m5", contract, cache_root=cache):
            rows = train_rows(windows)
            shards[shard] = tensor_shard(tensors, shard, rows, encoder(local, settings["lookback"]), local)
            print(f"[Step 2 full] {shard}: {len(rows)} train rows", flush=True)
        del windows, rows
        model, timing = fit(full, tensors, shards, settings, experiment, FULL_SEED, device)
        timing["fit_rows"] = sum(item["n"] for item in shards.values())
        saved = {"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                 "scalers": scalers, "version": experiment, "seed": FULL_SEED, "dataset": "m5",
                 "fit_scope": "train_only", "lookback": settings["lookback"],
                 "history_features": HISTORY_FEATURES, "current_features": CURRENT_FEATURES,
                 "parameters": settings, "fit": timing}
        training._save_torch(path, saved)
        del model
    print(f"[Step 2 full] checkpoint saved: {path}", flush=True)
    return path


def forecast_and_audit(root, output, cache, reference, contract, device, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    network = training._make_network(MODEL, {"lookback": saved["lookback"]}, saved["seed"], device)
    network.load_state_dict(saved["state_dict"])
    checkpoint_sha = _sha256(checkpoint)
    directory = checkpoint.parent
    manifest = {}
    for shard, windows, scalers in iter_windows(root, "m5", contract, cache_root=cache):
        target = windows.loc[windows["split"].eq("test") & _observed(windows)]
        filename = f"predictions_{shard}.csv"
        existing = directory / f"prediction_{shard}.json"
        if existing.is_file():
            metadata = training._read_json(existing)
            if metadata["checkpoint_sha256"] != checkpoint_sha:
                raise ValueError(f"{shard}: predictions use another checkpoint")
            _check_sha(directory / filename, metadata["sha256"])
        else:
            with _attention_scope(MODEL):
                values = training._model_eval(network, target, encoder(scalers, saved["lookback"]),
                                              scalers, device, song=True)
            metadata = {**_write_shard(directory / filename, "m5", target, values),
                        "checkpoint_sha256": checkpoint_sha}
            training._write_json(existing, metadata)
        manifest[shard] = {"file": filename, **metadata}
        print(f"[Step 2 test] {shard}: {metadata['rows']} predictions", flush=True)
    del windows, target, network

    source = prepare_reference(root, reference, "m5", contract)
    totals, by_store = [], []
    normalizers = {"train_mean": source["train_qty_mean"], "train_std": source["train_qty_std"]}
    for shard, file in manifest.items():
        ref = source["shards"][shard]
        _check_sha(directory / file["file"], file["sha256"])
        labels = reference / ref["file"]
        _check_sha(labels, ref["sha256"])
        truth = align_predictions(_csv(directory / file["file"]), _csv(labels))
        if len(truth) != ref["count"] or len(truth) != contract["test_key_shards"]["m5"][shard]["n"]:
            raise ValueError(f"{shard}: incomplete frozen test coverage")
        scores = sufficient(truth).sum()
        totals.append(scores)
        by_store.append({"store": shard, **summarize(scores, **normalizers)})
    test = summarize(pd.concat(totals, axis=1).sum(axis=1), **normalizers)
    frozen = contract["lock"]["frozen_test_keys"]["m5"]
    if test["n"] != frozen["count"] or set(manifest) != set(contract["m5_stores"]):
        raise ValueError("Step 2 did not verify every frozen M5 test key")
    pd.DataFrame(by_store).to_csv(output / "m5_test_by_store.csv", index=False)
    pd.DataFrame([test]).to_csv(output / "m5_test.csv", index=False)
    record = {"status": "complete", "full_m5_series": len(saved["scalers"]), "stores": len(manifest),
              "seed": FULL_SEED, "fit_scope": "train_only", "fit": saved["fit"], "settings": saved["parameters"],
              "test_keys_sha256": frozen["sha256"], "checkpoint_sha256": checkpoint_sha, "metrics": test,
              "prediction_files": manifest}
    training._write_json(output / "m5_test_audit.json", record)
    print(f"[Step 2 audit] {test['n']} frozen test keys, RMSE {test['rmse']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=PHASE5 / "data_cache")
    parser.add_argument("--reference-root", type=Path, default=PHASE5 / "audit" / "m5" / "reference")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stage", choices=("all", "screen", "full"), default="all")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = training.resolve_device(args.device)
    torch.set_num_threads(THREADS)
    torch.set_float32_matmul_precision("highest")
    contract = load_contract(args.phase1_root, LOCK_PATH)
    root = contract["root"]
    identity = {"phase1_sha256": contract["phase1_sha256"], "lock_sha256": contract["lock_sha256"],
                "implementation_sha256": _sha256(Path(__file__)),
                "training_sha256": _sha256(Path(training.__file__)),
                "model_sha256": _sha256(Path(__file__).with_name("song_forecast.py")),
                "cuda_graph_sha256": _sha256(Path(__file__).with_name("phase5_cuda_graph.py")),
                "torch_version": str(torch.__version__), "device": str(device),
                "execution_mode": MODE if device.type == "cuda" else "cpu",
                "attention_backend": "math", "threads": THREADS,
                "search": {"lookbacks": list(LOOKBACKS), "learning_rates": list(RATES),
                           "screen_seeds": list(SCREEN_SEEDS), "epoch_cap": EPOCHS, "batch": BATCH}}
    identity_file = output / "run_identity.json"
    if identity_file.is_file() and training._read_json(identity_file) != identity:
        raise ValueError("The existing tune output belongs to another configuration or runtime")
    training._write_json(identity_file, identity)
    if args.stage in {"all", "screen"}:
        screen(root, output, contract, identity, device)
    if args.stage in {"all", "full"}:
        selected = training._read_json(output / "selection.json")
        if selected["identity"] != identity or selected["selection_split"] != "validation":
            raise ValueError("Full fit requires the locked validation selection")
        checkpoint = full_fit(root, output, args.cache_dir, contract, identity, device, selected["selected"])
        forecast_and_audit(root, output, args.cache_dir, args.reference_root, contract, device, checkpoint)


if __name__ == "__main__":
    main()
