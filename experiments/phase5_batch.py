"""Run or inspect the resumable Phase 5 forecast matrix from its frozen settings."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

from .phase5_data import load_contract
from .phase5_verify import LOCK_PATH, OUTPUT_ROOT, PHASE1_ROOT, audit_forecast_runs, matrix


def _selected(lock: dict, dataset: str, models, seeds):
    known_models = set(lock["forecast"][dataset]["settings_by_model"])
    known_seeds = set(lock["phase5_seeds"])
    if (models and not set(models).issubset(known_models)) or (seeds and not set(seeds).issubset(known_seeds)):
        raise ValueError("The requested model or seed is absent from the locked configuration")
    return [(model, seed) for model, seed in matrix(lock, dataset)
            if (not models or model in models) and (not seeds or seed in seeds)]


def _default_run_path(output: Path, dataset: str, model: str, seed: int) -> Path:
    return output / "forecast" / dataset / model / f"seed{seed}"


def _uses_cuda(model: str, cuda_python: Path | None) -> bool:
    return cuda_python is not None and model.startswith(("lstm", "transformer", "song_"))


def _launch_jobs(args, jobs: list[tuple[str, int]], lane: str, identity: dict) -> None:
    for index, (model, seed) in enumerate(jobs, 1):
        target = _default_run_path(args.output_root, args.dataset, model, seed)
        marker = target / "run.json"
        if marker.is_file():
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if saved.get("status") == "complete":
                wanted = {**identity, "dataset": args.dataset, "model": model, "seed": seed}
                if any(saved.get(key) != value for key, value in wanted.items()):
                    raise ValueError(f"{target}: completed output does not match the scheduled job")
                print(f"[Phase 5 batch] {lane} {index}/{len(jobs)} {args.dataset}/{model}/{seed}: already finished",
                      flush=True)
                continue
        cuda = _uses_cuda(model, args.cuda_python)
        python = str(args.cuda_python.resolve()) if cuda else sys.executable
        device = "cuda" if cuda else "cpu"
        cmd = [python, "-u", "-m", "experiments.phase5_forecast", "--dataset", args.dataset,
               "--model", model, "--seed", str(seed), "--phase1-root", str(args.phase1_root),
               "--lock-path", str(args.lock_path), "--output-dir", str(target),
               "--device", device, "--cpu-threads", str(args.cpu_threads),
               "--arima-workers", str(args.arima_workers)]
        if args.cache_dir is not None:
            cmd.extend(["--cache-dir", str(args.cache_dir)])
        if args.cuda_graph and cuda:
            cmd.append("--cuda-graph")
        print(f"[Phase 5 batch] {lane} {index}/{len(jobs)} {args.dataset}/{model}/{seed} ({device})", flush=True)
        subprocess.run(cmd, check=True)


def show_status(output: Path, lock: dict) -> None:
    result = {}
    for dataset in ("uci", "m5"):
        jobs = list(matrix(lock, dataset))
        published = []
        working = []
        for name, seed in jobs:
            marker = _default_run_path(output, dataset, name, seed) / "run.json"
            if not marker.is_file():
                continue
            record = json.loads(marker.read_text(encoding="utf-8"))
            if record.get("status") == "complete":
                published.append((name, seed))
            else:
                working.append({"model": name, "seed": seed, "manifest_status": record.get("status"),
                                "scored_shards": len(record.get("shards", {}))})
        audit = output / "audit" / dataset / "audit.json"
        verified = 0
        if audit.is_file():
            verified = json.loads(audit.read_text(encoding="utf-8"))["verified_model_seed_runs"]
        result[dataset] = {"expected": len(jobs), "published": len(published), "audited": verified,
                           "published_runs": published, "incomplete_runs": working}
    pricing = output / "pricing"
    markers = [pricing / name for name in ("pricing_run.json", "pricing_progress.json")]
    for marker in markers:
        if marker.is_file():
            details = json.loads(marker.read_text(encoding="utf-8"))
            result["pricing"] = {k: details[k] for k in
                                 ("status", "seeds", "n_rows", "completed_units", "planned_units",
                                  "completed_decision_rows", "planned_decision_rows", "policies_by_seed")
                                 if k in details}
            break
    else:
        result["pricing"] = {"status": "pending"}
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("uci", "m5"))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--phase1-root", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--lock-path", type=Path, default=LOCK_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cuda-python", type=Path,
                        help="Run LSTM and Transformers with the specified CUDA-enabled Python")
    parser.add_argument("--parallel-cpu-cuda", action="store_true",
                        help="Run one CPU lane and one CUDA lane in the same batch")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="Use tested CUDA Graph execution for M5's neural lane")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--arima-workers", type=int, default=4)
    parser.add_argument("--status", action="store_true", help="Show manifests without launching workloads")
    parser.add_argument("--no-audit", action="store_true",
                        help="Defer the independent audit to a separate command")
    args = parser.parse_args()
    if args.status:
        lock = json.loads(args.lock_path.read_text(encoding="utf-8"))
        show_status(args.output_root, lock)
        return
    if args.dataset is None:
        parser.error("choose --dataset for a batch")
    if args.cpu_threads < 1 or args.arima_workers < 1:
        parser.error("thread and worker counts must be positive")
    if args.cuda_python is not None and not args.cuda_python.is_file():
        parser.error("the selected CUDA Python does not exist")
    if args.parallel_cpu_cuda and args.cuda_python is None:
        parser.error("--parallel-cpu-cuda requires --cuda-python")
    if args.cuda_graph and (args.dataset != "m5" or args.cuda_python is None):
        parser.error("--cuda-graph requires M5 and --cuda-python")
    contract = load_contract(args.phase1_root, args.lock_path)
    identity = {key: contract[key] for key in ("lock_sha256", "phase1_sha256")}
    chosen = _selected(contract["lock"], args.dataset, args.models, args.seeds)
    if not chosen:
        parser.error("the requested filters leave no locked jobs")
    errors = []
    if args.parallel_cpu_cuda:
        gpu = [job for job in chosen if _uses_cuda(job[0], args.cuda_python)]
        cpu = [job for job in chosen if job not in gpu]
        with ThreadPoolExecutor(max_workers=2) as lanes:
            pending = {lanes.submit(_launch_jobs, args, jobs, lane, identity): lane
                       for lane, jobs in (("cpu", cpu), ("cuda", gpu)) if jobs}
            for result in as_completed(pending):
                try:
                    result.result()
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    print(f"[Phase 5 batch] {pending[result]} lane stopped: {error}", file=sys.stderr, flush=True)
                    errors.append(error)
    else:
        _launch_jobs(args, chosen, "serial", identity)
    if not args.no_audit:
        report = audit_forecast_runs(args.phase1_root, args.output_root, args.dataset, contract)
        print(f"[Phase 5 batch] {args.dataset}: {report['verified_model_seed_runs']}/"
              f"{report['expected_model_seed_runs']} runs audited.", flush=True)
    if errors:
        raise errors[0]


if __name__ == "__main__":
    main()
