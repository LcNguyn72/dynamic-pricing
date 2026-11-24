"""Global Phase 5 refits with bounded M5 batches and resumable neural training."""

from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn

from .forecast_torch import SequenceForecaster, feature_arrays as sequence_features
from .phase1_finalize import _observed, _sha256
from .phase5_data import iter_windows, load_contract, load_scalers
from .song_forecast import (
    CURRENT_FEATURES, HISTORY_FEATURES, LOOKBACK, SongTransformer, _scalers,
    feature_arrays as song_features,
)

STORE_BATCH = 32_768
PROGRESS_BATCH = 1_024
INFERENCE_BATCH = 512
SRC = Path(__file__).resolve().parent


def _write_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(f"{path.suffix}.tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_torch(path: Path, value: dict) -> None:
    temp = path.with_suffix(".pt.tmp")
    torch.save(value, temp)
    temp.replace(path)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch runtime has no available CUDA GPU")
    return torch.device(requested)


def _feature_identity(dataset: str, variant: str, contract: dict, upstream: str) -> dict:
    return {
        "dataset": dataset, "variant": variant, "phase1_sha256": contract["phase1_sha256"],
        "protocol_sha256": contract["protocol_sha256"], "preprocessing": "train_only",
        "features_code_sha256": _sha256(SRC / upstream),
        "runner_code_sha256": _sha256(Path(__file__)),
    }


def _cache_directory(root: Path, identity: dict) -> Path:
    cache = root / "features" / (f"{identity['dataset']}_{identity['variant']}_"
                                 f"{identity['phase1_sha256'][:12]}_{identity['runner_code_sha256'][:12]}")
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / "identity.json"
    if marker.exists():
        if _read_json(marker) != identity:
            raise ValueError(f"Feature cache identity mismatch: {cache}")
    else:
        _write_json(marker, identity)
    return cache


def _fit_rows(windows: pd.DataFrame) -> pd.DataFrame:
    return windows.loc[windows["split"].isin(("train", "validation")) & _observed(windows)]


def _scope(dataset: str, root: Path, contract: dict) -> tuple[dict, list[str], tuple[str, ...]]:
    scalers = load_scalers(root, dataset, contract)
    shards = ["uci"] if dataset == "uci" else list(contract["protocol"]["datasets"]["m5"]["store_ids"])
    covariates = () if dataset == "uci" else ("event_days", "snap_days")
    return scalers, shards, covariates


def _neural_encoding(model_name: str, settings: dict, scalers: dict, covariates: tuple) -> tuple:
    if model_name == "song_transformer":
        config = {"history_dim": len(HISTORY_FEATURES), "current_dim": len(CURRENT_FEATURES)}

        def encode(rows):
            history, aux = song_features(rows, scalers)
            return history, aux, np.zeros(len(rows), dtype=np.int64)

        return config, encode
    mapping = {sku: i for i, sku in enumerate(sorted(scalers))}
    embedding = len(mapping) > 256
    with_price = model_name.endswith("_price")
    declared = settings["model_config"]
    if declared["lookback"] != LOOKBACK or declared["history_dim"] != (2 if with_price else 1):
        raise ValueError(f"{model_name}: locked history features differ")
    config = {
        "architecture": declared["architecture"], "lookback": LOOKBACK,
        "history_dim": declared["history_dim"],
        "auxiliary_dim": 4 + len(covariates) + int(with_price) + (0 if embedding else len(mapping)),
        "sku_count": len(mapping) if embedding else 0,
        "sku_embedding_dim": min(32, max(8, round(math.sqrt(len(mapping))))) if embedding else 0,
    }

    def encode(rows):
        arrays = sequence_features(rows, mapping, scalers, LOOKBACK, with_price, covariates,
                                   "embedding" if embedding else "onehot")
        return arrays if embedding else (*arrays, np.zeros(len(rows), dtype=np.int64))

    return config, encode


def _raw_features(rows: pd.DataFrame, encode, scalers: dict, *, song: bool) -> tuple:
    history, aux, indexes = encode(rows)
    quantity = rows["qty"].to_numpy(dtype=np.float32)
    if song:
        mean, std, _, _ = _scalers(rows, scalers)
        quantity = (quantity - mean) / std
    else:
        quantity = np.log1p(quantity)
    return history, aux, indexes, quantity


TENSOR_NAMES = ("history", "auxiliary", "sku", "target")


def _materialize_tensors(cache: Path, shard: str, rows: pd.DataFrame, encode, scalers: dict, song: bool) -> dict:
    if rows.empty:
        raise ValueError(f"{shard}: no observed training or validation rows")
    files = {name: cache / f"{shard}_{name}.npy" for name in TENSOR_NAMES}
    data = _raw_features(rows.iloc[:min(STORE_BATCH, len(rows))], encode, scalers, song=song)
    blocks = {name: np.lib.format.open_memmap(path, mode="w+", dtype=part.dtype,
                                             shape=(len(rows), *part.shape[1:]))
              for (name, path), part in zip(files.items(), data, strict=True)}
    try:
        for start in range(0, len(rows), STORE_BATCH):
            end = min(start + STORE_BATCH, len(rows))
            if start:
                data = _raw_features(rows.iloc[start:end], encode, scalers, song=song)
            for name, part in zip(TENSOR_NAMES, data, strict=True):
                blocks[name][start:end] = part
        for block in blocks.values():
            block.flush()
        del block
    finally:
        blocks.clear()
    record = {"n": len(rows), "files": {name: _sha256(path) for name, path in files.items()}}
    _write_json(cache / f"{shard}.json", record)
    return record


def _tensor_shards(root: Path, dataset: str, contract: dict, base_cache: Path, shards: list[str],
                   model: str, encode, scalers: dict) -> tuple[Path, dict]:
    song = model == "song_transformer"
    variant = "song" if song else ("seq_price" if model.endswith("_price") else "seq")
    identity = _feature_identity(dataset, variant, contract, "song_forecast.py" if song else "forecast_torch.py")
    cache = _cache_directory(base_cache, identity)
    records = {}
    for shard in shards:
        marker = cache / f"{shard}.json"
        if marker.exists():
            record = _read_json(marker)
            if all((path := cache / f"{shard}_{name}.npy").is_file() and _sha256(path) == checksum
                   for name, checksum in record["files"].items()):
                records[shard] = record
        if shard not in records:
            for _, windows, _ in iter_windows(root, dataset, contract, shards=[shard], cache_root=base_cache):
                records[shard] = _materialize_tensors(cache, shard, _fit_rows(windows), encode, scalers, song)
            print(f"[Phase 5 features] {dataset} {variant} {shard}: {records[shard]['n']} fit rows", flush=True)
    return cache, records


def _open_tensors(cache: Path, shard: str, device: torch.device) -> tuple:
    arrays = tuple(np.load(cache / f"{shard}_{name}.npy", allow_pickle=False, mmap_mode="c")
                   for name in TENSOR_NAMES)
    if device.type == "cuda":
        return tuple(torch.from_numpy(value).to(device) for value in arrays)
    return arrays


def _batch_tensors(arrays: tuple, order, start: int, stop: int, device: torch.device) -> tuple:
    if device.type == "cuda":
        indices = order[start:stop]
        return tuple(values[indices] for values in arrays)
    return tuple(torch.from_numpy(values[order[start:stop]]) for values in arrays)


def _train_step(network, optimizer, batch: tuple, *, song: bool) -> torch.Tensor:
    history, auxiliary, idx, target = batch
    optimizer.zero_grad(set_to_none=True)
    prediction = network(history, auxiliary) if song else network(history, auxiliary, idx)
    loss = nn.functional.mse_loss(prediction, target)
    loss.backward()
    if not song:
        nn.utils.clip_grad_norm_(network.parameters(), 1.0)
    optimizer.step()
    return loss.detach()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_eval(network, rows, encode, scalers, device, *, song: bool) -> np.ndarray:
    results = []
    network.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), INFERENCE_BATCH):
            part = rows.iloc[start:start + INFERENCE_BATCH]
            h, a, i = (torch.from_numpy(value).to(device) for value in encode(part))
            output = (network(h, a) if song else network(h, a, i)).cpu().numpy()
            if song:
                mean, std, _, _ = _scalers(part, scalers)
                output = output * std + mean
            else:
                output = np.expm1(output)
            results.append(np.maximum(output, 0.0))
    return np.concatenate(results) if results else np.empty(0)


def _make_network(model: str, config: dict, seed: int, device: torch.device):
    torch.manual_seed(seed)
    network = SongTransformer(**config) if model == "song_transformer" else SequenceForecaster(**config)
    return network.to(device)


def _neural_progress(path: Path, network, optimizer, device: torch.device, version: dict,
                     epoch: int, shard_index: int, batches: int, seconds: float, steps: int) -> None:
    _save_torch(path, {"version": version, "model": network.state_dict(),
                      "optimizer": optimizer.state_dict(), "rng": torch.random.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                      "epoch": epoch, "shard_index": shard_index, "batches": batches,
                      "seconds": seconds, "steps": steps})


def _fit_neural(model, seed, dir_out, settings, device, config, cache, shards, identity):
    network = _make_network(model, config, seed, device)
    optimizer = torch.optim.Adam(network.parameters(), lr=settings["learning_rate"])
    path = dir_out / "train_progress.pt"
    epochs, batch_size = int(settings["best_epoch"]), int(settings["batch_size"])
    cursor = {"epoch": 0, "shard_index": 0, "batches": 0, "steps": 0, "seconds": 0.0}
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["version"] != identity:
            raise ValueError("Neural progress checkpoint belongs to a different run")
        network.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.random.set_rng_state(saved["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        cursor.update({key: saved[key] for key in cursor})
        print(f"[Phase 5 resume] epoch {cursor['epoch']+1}, shard {cursor['shard_index']+1}, "
              f"batch {cursor['batches']}", flush=True)
        del saved
    steps = cursor["steps"]
    started = perf_counter()
    network.train()
    for epoch in range(cursor["epoch"], epochs):
        order = np.random.default_rng([seed, epoch]).permutation(len(shards))
        shard_list = list(shards)
        for index, number in enumerate(order):
            if epoch == cursor["epoch"] and index < cursor["shard_index"]:
                continue
            shard = shard_list[int(number)]
            n = shards[shard]["n"]
            shuffle = np.random.default_rng([seed, epoch, int(number)]).permutation(n)
            if device.type == "cuda":
                shuffle = torch.from_numpy(shuffle).to(device)
            data = _open_tensors(cache, shard, device)
            skipped = cursor["batches"] if (epoch, index) == (cursor["epoch"], cursor["shard_index"]) else 0
            total = math.ceil(n / batch_size)
            for batch_id in range(skipped, total):
                start = batch_id * batch_size
                batch = _batch_tensors(data, shuffle, start, min(start + batch_size, n), device)
                loss = _train_step(network, optimizer, batch, song=model == "song_transformer")
                steps += 1
                if (batch_id + 1) % PROGRESS_BATCH == 0 or batch_id + 1 == total:
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("Refit diverged before checkpointing")
                    seconds = cursor["seconds"] + perf_counter() - started
                    at_boundary = batch_id + 1 == total
                    # The cursor points to the first batch that still needs work.
                    next_epoch = epoch + 1 if at_boundary and index + 1 == len(shards) else epoch
                    next_shard = (index + 1) % len(shards) if at_boundary else index
                    _neural_progress(path, network, optimizer, device, identity, next_epoch, next_shard,
                                     0 if at_boundary else batch_id + 1, seconds, steps)
                    print(f"[Phase 5 fit] {model} seed={seed} epoch={epoch+1}/{epochs} "
                          f"{shard} batch={batch_id+1}/{total} loss={float(loss):.5f}", flush=True)
            del data
    return network, {"optimizer_steps": steps, "fit_seconds": cursor["seconds"] + perf_counter() - started,
                     "epochs": epochs, "batch_size": batch_size}


def _train_neural(dataset, model, seed, root, contract, settings, out, cache_root, device, cpu_threads):
    torch.set_num_threads(cpu_threads)
    scalers, shards, covariates = _scope(dataset, root, contract)
    config, encode = _neural_encoding(model, settings, scalers, covariates)
    checkpoint = out / "model.pt"
    version = {"lock_sha256": contract["lock_sha256"], "phase1_sha256": contract["phase1_sha256"],
               "dataset": dataset, "model": model, "seed": seed, "device": str(device),
               "torch_version": str(torch.__version__), "trainer_sha256": _sha256(Path(__file__)),
               "architecture": config, "threads": cpu_threads}
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["version"] != version or saved["scalers"] != scalers:
            raise ValueError("Existing forecast weights differ from this run identity")
        network = _make_network(model, config, seed, device)
        network.load_state_dict(saved["state_dict"])
        fit = saved["fit"]
    else:
        cache, records = _tensor_shards(root, dataset, contract, cache_root, shards, model, encode, scalers)
        network, fit = _fit_neural(model, seed, out, settings, device, config, cache, records, version)
        fit["fit_rows"] = sum(record["n"] for record in records.values())
        _save_torch(checkpoint, {"version": version, "state_dict": network.state_dict(),
                                "scalers": scalers, "fit": fit, "learning_rate": settings["learning_rate"]})

    def predict(rows):
        return _model_eval(network, rows, encode, scalers, device, song=model == "song_transformer")

    return predict, {"path": checkpoint.name, "sha256": _sha256(checkpoint), "fit": fit}


def _train_xgboost(dataset, model, seed, root, contract, settings, out, cache_root):
    import xgboost as xgb
    from scipy import sparse
    from .forecast_benchmark import _tabular_features

    scalers, shards, covariates = _scope(dataset, root, contract)
    skus = sorted(scalers)
    price = model.endswith("_price")
    parameters = dict(settings["parameters"])
    rounds = int(parameters.pop("n_estimators"))
    nthread = int(parameters.pop("n_jobs"))
    parameters.pop("random_state")
    parameters.update(seed=seed, nthread=nthread, verbosity=0)
    identity = _feature_identity(dataset, model, contract, "forecast_benchmark.py")
    checkpoint, meta = out / "model.ubj", out / "model_meta.json"
    version = {**identity, "lock_sha256": contract["lock_sha256"], "seed": seed,
               "xgboost": xgb.__version__, "parameters": parameters, "rounds": rounds}

    def encode(rows):
        return _tabular_features(rows, skus, scalers, LOOKBACK, price, covariates)

    if checkpoint.exists() and meta.exists():
        metadata = _read_json(meta)
        if metadata["version"] != version or metadata["model_sha256"] != _sha256(checkpoint):
            raise ValueError("Existing XGBoost model differs from this run identity")
        booster = xgb.Booster(params={"nthread": nthread}, model_file=checkpoint)
    else:
        cache = _cache_directory(cache_root, identity)
        n = 0
        for shard in shards:
            features, labels, marker = cache / f"{shard}.npz", cache / f"{shard}_y.npy", cache / f"{shard}.json"
            record = _read_json(marker) if marker.exists() else {}
            if not features.exists() or not labels.exists() or record.get("features") != _sha256(features) or record.get("labels") != _sha256(labels):
                for _, windows, _ in iter_windows(root, dataset, contract, shards=[shard], cache_root=cache_root):
                    fit = _fit_rows(windows)
                    sparse.save_npz(features, sparse.csr_matrix(encode(fit)), compressed=False)
                    np.save(labels, np.log1p(fit["qty"].to_numpy(dtype=np.float32)), allow_pickle=False)
                    record = {"n": len(fit), "features": _sha256(features), "labels": _sha256(labels)}
                    _write_json(marker, record)
                    del fit
                print(f"[Phase 5 features] {dataset} {model} {shard}: {record['n']} fit rows", flush=True)
            n += record["n"]

        class StoreIterator(xgb.DataIter):
            def __init__(self, prefix: Path):
                super().__init__(cache_prefix=str(prefix), release_data=True)
                self.index = 0

            def reset(self):
                self.index = 0

            def next(self, input_data):
                if self.index == len(shards):
                    return False
                shard = shards[self.index]
                input_data(data=sparse.load_npz(cache / f"{shard}.npz"),
                           label=np.load(cache / f"{shard}_y.npy", allow_pickle=False))
                self.index += 1
                return True

        class Progress(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                if (epoch + 1) % 50 == 0 or epoch + 1 == rounds:
                    print(f"[Phase 5 fit] {model_name} seed={seed} boosting {epoch+1}/{rounds}", flush=True)
                return False

        model_name = model
        start = perf_counter()
        with TemporaryDirectory(prefix="xgb-pages-", dir=out) as temporary:
            # One pooled quantile matrix, even when each input page comes from a different store.
            training = xgb.ExtMemQuantileDMatrix(StoreIterator(Path(temporary) / "page"), nthread=nthread)
            booster = xgb.train(parameters, training, num_boost_round=rounds, callbacks=[Progress()])
            del training
        temporary = checkpoint.with_suffix(".tmp.ubj")
        booster.save_model(temporary)
        temporary.replace(checkpoint)
        metadata = {"version": version, "model_sha256": _sha256(checkpoint), "train_scalers": scalers,
                    "sku_order": skus, "covariates": covariates, "fit_rows": n,
                    "fit_seconds": perf_counter() - start}
        _write_json(meta, metadata)

    def predict(rows):
        return np.concatenate([np.expm1(booster.inplace_predict(encode(rows.iloc[i:i + STORE_BATCH])))
                               for i in range(0, len(rows), STORE_BATCH)]).clip(min=0)

    return predict, {"path": checkpoint.name, "sha256": _sha256(checkpoint),
                     "metadata_path": meta.name, "metadata_sha256": _sha256(meta),
                     "fit": {key: metadata[key] for key in ("fit_rows", "fit_seconds")},
                     "effective_parameters": parameters, "rounds": rounds}


def fit_or_load(dataset: str, model: str, seed: int, root: Path, contract: dict, settings: dict,
                output: Path, cache_root: Path, device: str, cpu_threads: int):
    if model.startswith("xgboost"):
        return _train_xgboost(dataset, model, seed, root, contract, settings, output, cache_root)
    return _train_neural(dataset, model, seed, root, contract, settings, output, cache_root,
                         torch.device(device), cpu_threads)


def calibrate_neural(model, seed, root, lock_path, cache_root, shard, batches, requested_device, cpu_threads):
    if model not in ("lstm", "lstm_price", "transformer", "transformer_price", "song_transformer"):
        raise ValueError("Calibration requires a neural model")
    if batches < 1:
        raise ValueError("Calibration batch count must be positive")
    setup_start = perf_counter()
    torch.set_num_threads(cpu_threads)
    device = resolve_device(requested_device)
    contract = load_contract(root, lock_path)
    settings = contract["lock"]["forecast"]["m5"]["settings_by_model"][model]
    scalers, _, covariates = _scope("m5", root, contract)
    config, encode = _neural_encoding(model, settings, scalers, covariates)
    cache, records = _tensor_shards(root, "m5", contract, cache_root, [shard], model, encode, scalers)
    n, batch_size = records[shard]["n"], settings["batch_size"]
    network = _make_network(model, config, seed, device)
    optimizer = torch.optim.Adam(network.parameters(), lr=settings["learning_rate"])
    data = _open_tensors(cache, shard, device)
    order = np.random.default_rng(seed).permutation(n)
    if device.type == "cuda":
        order = torch.from_numpy(order).to(device)
    count = min(math.ceil(n / batch_size), batches)
    warmup = min(8, count - 1)
    for i in range(warmup):
        _train_step(network, optimizer, _batch_tensors(data, order, i*batch_size, (i+1)*batch_size, device),
                    song=model == "song_transformer")
    _sync(device)
    start = perf_counter()
    for i in range(warmup, count):
        _train_step(network, optimizer, _batch_tensors(data, order, i*batch_size, (i+1)*batch_size, device),
                    song=model == "song_transformer")
    _sync(device)
    seconds = perf_counter() - start
    rows = min(count * batch_size, n) - warmup * batch_size
    return {"purpose": "throughput_only", "dataset": "m5", "store": shard, "model": model,
            "device": str(device), "torch_version": str(torch.__version__), "batch_size": batch_size,
            "timed_batches": count - warmup, "timed_rows": rows, "training_seconds": seconds,
            "elapsed_seconds_including_setup": perf_counter() - setup_start, "fit_rows_in_store": n,
            "rows_per_second": rows / seconds, "estimated_store_epoch_seconds": n * seconds / rows,
            "locked_full_fit_epochs": settings["best_epoch"]}
