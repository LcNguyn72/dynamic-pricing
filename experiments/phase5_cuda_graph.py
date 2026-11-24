"""Optional CUDA Graph execution of the locked M5 neural refits."""

from __future__ import annotations

from contextlib import nullcontext
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .phase1_finalize import _sha256
from .phase5_data import load_contract
from . import phase5_training as eager

MODE = "cuda_graph_fused_adam_fp32"
PROGRESS_BATCH = eager.PROGRESS_BATCH
NEURAL_MODELS = ("lstm", "lstm_price", "transformer", "transformer_price", "song_transformer")
ATTENTION_MODELS = frozenset(("transformer", "transformer_price", "song_transformer"))


def _attention_scope(model: str):
    # The efficient SDPA backward returned NaN on a finite CA_3 training batch
    # in the Windows CUDA runtime. Mathematical SDPA passed on the same inputs.
    return sdpa_kernel(SDPBackend.MATH) if model in ATTENTION_MODELS else nullcontext()


def _attention_name(model: str) -> str:
    return "math" if model in ATTENTION_MODELS else "not_applicable"


class GraphStep:
    """Replay full batches and use the same fused optimizer on smaller final batches."""

    def __init__(self, network, optimizer, batch_size: int, device: torch.device, song: bool):
        self.network = network
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.device = device
        self.song = song
        self.graph = None

    def _capture(self, batch: tuple) -> None:
        self.static = tuple(value.detach().clone() for value in batch)
        # Warm up the CUDA kernels on a side stream, then restore original
        # optimizer, weights and RNG so warmup is invisible to the refit.
        values = list(self.network.state_dict().values())
        values.extend(tensor for state in self.optimizer.state.values() for tensor in state.values())
        stored = [value.detach().clone() for value in values]
        device_id = self.device.index if self.device.index is not None else torch.cuda.current_device()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.random.fork_rng(devices=[device_id]):
            with torch.cuda.stream(stream):
                for _ in range(2):
                    eager._train_step(self.network, self.optimizer, self.static, song=self.song)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
        with torch.no_grad():
            for value, original in zip(values, stored, strict=True):
                value.copy_(original)
        self.optimizer.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss = eager._train_step(self.network, self.optimizer, self.static, song=self.song)

    def step(self, batch: tuple) -> torch.Tensor:
        if len(batch[0]) != self.batch_size:
            return eager._train_step(self.network, self.optimizer, batch, song=self.song)
        if self.graph is None:
            self._capture(batch)
        else:
            for destination, value in zip(self.static, batch, strict=True):
                destination.copy_(value)
        self.graph.replay()
        return self.loss


def _fused_optimizer(network, lr: float) -> torch.optim.Adam:
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, fused=True, capturable=True)
    # Allocate the Adam state before capture so all replays and checkpoint
    # restores refer to the same update tensors.
    for parameter in network.parameters():
        optimizer.state[parameter] = {
            "step": torch.zeros((), dtype=torch.float32, device=parameter.device),
            "exp_avg": torch.zeros_like(parameter, memory_format=torch.preserve_format),
            "exp_avg_sq": torch.zeros_like(parameter, memory_format=torch.preserve_format),
        }
    return optimizer


def fit_graph(model, seed, output, settings, device, config, cache, shards, identity):
    with _attention_scope(model):
        return _fit_graph(model, seed, output, settings, device, config, cache, shards, identity)


def _fit_graph(model, seed, output, settings, device, config, cache, shards, identity):
    network = eager._make_network(model, config, seed, device)
    optimizer = _fused_optimizer(network, settings["learning_rate"])
    path = output / "train_progress.pt"
    epochs, batch_size = int(settings["best_epoch"]), int(settings["batch_size"])
    cursor = {"epoch": 0, "shard_index": 0, "batches": 0, "seconds": 0.0, "steps": 0}
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["version"] != identity:
            raise ValueError("CUDA Graph progress belongs to a different run")
        network.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.random.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"], device)
        cursor.update({key: saved[key] for key in cursor})
        print(f"[Phase 5 graph resume] epoch={cursor['epoch']+1} shard={cursor['shard_index']+1} "
              f"batch={cursor['batches']}", flush=True)
        del saved
    steps = cursor["steps"]
    started = perf_counter()
    network.train()
    executor = GraphStep(network, optimizer, batch_size, device, model == "song_transformer")
    shard_names = list(shards)
    for epoch in range(cursor["epoch"], epochs):
        order = np.random.default_rng([seed, epoch]).permutation(len(shards))
        for position, number in enumerate(order):
            if epoch == cursor["epoch"] and position < cursor["shard_index"]:
                continue
            shard = shard_names[int(number)]
            n = shards[shard]["n"]
            shuffle = torch.from_numpy(np.random.default_rng([seed, epoch, int(number)]).permutation(n)).to(device)
            data = eager._open_tensors(cache, shard, device)
            first = cursor["batches"] if (epoch, position) == (cursor["epoch"], cursor["shard_index"]) else 0
            total = math.ceil(n / batch_size)
            for batch_id in range(first, total):
                start = batch_id * batch_size
                loss = executor.step(eager._batch_tensors(data, shuffle, start, min(start + batch_size, n), device))
                steps += 1
                boundary = batch_id + 1 == total
                if (batch_id + 1) % PROGRESS_BATCH == 0 or boundary:
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("CUDA Graph refit diverged before checkpointing")
                    next_epoch = epoch + 1 if boundary and position + 1 == len(shards) else epoch
                    next_shard = (position + 1) % len(shards) if boundary else position
                    seconds = cursor["seconds"] + perf_counter() - started
                    eager._neural_progress(path, network, optimizer, device, identity, next_epoch, next_shard,
                                           0 if boundary else batch_id + 1, seconds, steps)
                    print(f"[Phase 5 graph fit] {model} seed={seed} epoch={epoch+1}/{epochs} "
                          f"{shard} batch={batch_id+1}/{total} loss={float(loss):.5f}", flush=True)
            del data, shuffle
    return network, {"optimizer_steps": steps, "fit_seconds": cursor["seconds"] + perf_counter() - started,
                     "epochs": epochs, "batch_size": batch_size}


def fit_or_load_graph(dataset, model, seed, root, contract, settings, output, cache_root, device, cpu_threads):
    torch.set_num_threads(cpu_threads)
    torch.set_float32_matmul_precision("highest")
    device = eager.resolve_device(device)
    if dataset != "m5" or model not in NEURAL_MODELS or device.type != "cuda":
        raise ValueError("CUDA Graph requires a Phase 5 M5 neural model and CUDA")
    scalers, shards, covariates = eager._scope(dataset, root, contract)
    config, encode = eager._neural_encoding(model, settings, scalers, covariates)
    checkpoint = output / "model.pt"
    identity = {"lock_sha256": contract["lock_sha256"], "phase1_sha256": contract["phase1_sha256"],
                "dataset": dataset, "model": model, "seed": seed, "device": str(device),
                "torch_version": str(torch.__version__), "execution_mode": MODE,
                "attention_backend": _attention_name(model),
                "graph_sha256": _sha256(Path(__file__)), "trainer_sha256": _sha256(Path(eager.__file__)),
                "architecture": config, "threads": cpu_threads,
                "matmul_precision": torch.get_float32_matmul_precision(),
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["version"] != identity or saved["scalers"] != scalers:
            raise ValueError("Existing CUDA Graph weights differ from this run identity")
        network = eager._make_network(model, config, seed, device)
        network.load_state_dict(saved["state_dict"])
        fit = saved["fit"]
    else:
        cache, records = eager._tensor_shards(root, dataset, contract, cache_root, shards, model, encode, scalers)
        network, fit = fit_graph(model, seed, output, settings, device, config, cache, records, identity)
        fit["fit_rows"] = sum(record["n"] for record in records.values())
        eager._save_torch(checkpoint, {"version": identity, "state_dict": network.state_dict(),
                                     "scalers": scalers, "fit": fit, "learning_rate": settings["learning_rate"]})

    def predict(rows):
        with _attention_scope(model):
            return eager._model_eval(network, rows, encode, scalers, device, song=model == "song_transformer")

    return predict, {"path": checkpoint.name, "sha256": _sha256(checkpoint),
                     "fit": fit, "execution_mode": MODE, "attention_backend": _attention_name(model)}


def calibrate_graph(model, seed, root, lock_path, cache_root, shard, batches, requested_device, cpu_threads):
    with _attention_scope(model):
        return _calibrate_graph(model, seed, root, lock_path, cache_root, shard, batches,
                                requested_device, cpu_threads)


def _calibrate_graph(model, seed, root, lock_path, cache_root, shard, batches, requested_device, cpu_threads):
    if model not in NEURAL_MODELS or batches < 1:
        raise ValueError("Calibration needs a supported neural model and positive batch count")
    device = eager.resolve_device(requested_device)
    if device.type != "cuda":
        raise ValueError("CUDA Graph calibration requires CUDA")
    started = perf_counter()
    torch.set_num_threads(cpu_threads)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats(device)
    contract = load_contract(root, lock_path)
    settings = contract["lock"]["forecast"]["m5"]["settings_by_model"][model]
    scalers, _, covariates = eager._scope("m5", root, contract)
    config, encode = eager._neural_encoding(model, settings, scalers, covariates)
    cache, records = eager._tensor_shards(root, "m5", contract, cache_root, [shard], model, encode, scalers)
    n, batch_size = records[shard]["n"], settings["batch_size"]
    network = eager._make_network(model, config, seed, device)
    optimizer = _fused_optimizer(network, settings["learning_rate"])
    graph = GraphStep(network, optimizer, batch_size, device, model == "song_transformer")
    data = eager._open_tensors(cache, shard, device)
    order = torch.from_numpy(np.random.default_rng(seed).permutation(n)).to(device)
    count = min(math.ceil(n / batch_size), batches)
    warm = min(8, count - 1)
    for i in range(warm):
        graph.step(eager._batch_tensors(data, order, i*batch_size, (i+1)*batch_size, device))
    eager._sync(device)
    timed = perf_counter()
    for i in range(warm, count):
        graph.step(eager._batch_tensors(data, order, i*batch_size, (i+1)*batch_size, device))
    eager._sync(device)
    seconds = perf_counter() - timed
    rows = min(count * batch_size, n) - warm * batch_size
    return {"purpose": "throughput_only", "dataset": "m5", "store": shard, "model": model,
            "device": str(device), "torch_version": str(torch.__version__), "execution_mode": MODE,
            "attention_backend": _attention_name(model),
            "batch_size": batch_size, "timed_batches": count-warm, "timed_rows": rows,
            "training_seconds": seconds, "elapsed_seconds_including_setup": perf_counter() - started,
            "fit_rows_in_store": n, "rows_per_second": rows / seconds,
            "estimated_store_epoch_seconds": n * seconds / rows, "locked_full_fit_epochs": settings["best_epoch"],
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
