"""Persist and audit the Transformer state paired with Phase 4 online PPO.

Phase 4 saved PPO ZIP files but not the forecaster after simulated feedback
during PPO training. Re-run only that policy, save its paired forecaster and
the per-SKU Torch RNG state needed for online test replay, then require every
saved Phase 4 test decision to match before publishing a Phase 5C manifest.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import stable_baselines3
import torch

from .pricing_simulator import LagAdjustedKernelSimulator
from .song_forecast import CURRENT_FEATURES, HISTORY_FEATURES
from .song_pricing_pilot import _environment, _ppo, load_forecaster


DEFAULT_PILOT = Path(__file__).resolve().parent / "artifacts" / "song_m5_pilot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _test_rows(windows: pd.DataFrame, skus: list[str], weeks_per_sku: int) -> pd.DataFrame:
    rows = windows.loc[windows["split"].eq("test")
                       & windows["observed_sales"].astype(bool)
                       & windows["sku"].isin(skus)].copy()
    rows = (rows.sort_values(["sku", "week"])
            .groupby("sku", group_keys=False).head(weeks_per_sku))
    counts = rows.groupby("sku").size().to_dict()
    if counts != {sku: weeks_per_sku for sku in skus}:
        raise ValueError(f"Phase 4 test cohort does not match: {counts}")
    return rows


def _replay(model, forecaster, test_rows: pd.DataFrame, simulator,
            seed: int) -> tuple[pd.DataFrame, dict[str, torch.Tensor]]:
    env = _environment(test_rows, simulator, "test", forecaster,
                       gradient=True, feedback=True)
    initial_forecaster = deepcopy(env.forecaster)
    episodes = []
    rng_by_sku = {}
    for sku_index, sku in enumerate(env.available_skus):
        env.forecaster = deepcopy(initial_forecaster)
        env.feedback.clear()
        env.interactions = 0
        env.feedback_updates = 0
        rng_by_sku[sku] = torch.get_rng_state().clone()
        observation, _ = env.reset(seed=seed + sku_index, options={"sku": sku})
        done = False
        while not done:
            action = model.predict(observation, deterministic=True)[0]
            observation, _, done, _, info = env.step(action)
            episodes.append({"policy": "ppo_feedback_gradient", "seed": seed, **info})
    return pd.DataFrame(episodes), rng_by_sku


def _compare(actual: pd.DataFrame, saved: pd.DataFrame) -> dict:
    keys = ["policy", "seed", "sku", "week"]
    actual = actual.sort_values(keys).reset_index(drop=True)
    saved = saved.sort_values(keys).reset_index(drop=True)
    if not actual[keys].equals(saved[keys]):
        raise AssertionError("Replayed and Phase 4 SKU-week keys differ")
    columns = ["chosen_price", "simulated_units", "simulated_revenue",
               "expected_revenue", "oracle_revenue", "expected_regret",
               "reference_price", "forecast_update_count"]
    differences = {name: float(np.max(np.abs(actual[name].to_numpy(dtype=float)
                                              - saved[name].to_numpy(dtype=float))))
                   for name in columns}
    if any(value > 1e-5 for value in differences.values()):
        raise AssertionError(f"Phase 4 replay diverged: {differences}")
    return {"rows": len(actual), "max_absolute_difference": differences,
            "online_updates_per_sku": actual.groupby("sku")["forecast_update_count"].max().to_dict()}


def run(pilot_dir: Path = DEFAULT_PILOT, pricing_dir: Path | None = None,
        output_dir: Path | None = None, seeds: tuple[int, ...] | None = None) -> dict:
    pilot_dir = Path(pilot_dir)
    pricing_dir = Path(pricing_dir) if pricing_dir else pilot_dir / "pricing_lag_512_5seeds"
    output_dir = Path(output_dir) if output_dir else pilot_dir / "phase5c_online_state"
    pricing_manifest_path = pricing_dir / "pricing_manifest.json"
    pricing_manifest = json.loads(pricing_manifest_path.read_text(encoding="utf-8"))
    if pricing_manifest["simulator_variant"] != "lag_adjusted_kernel":
        raise ValueError("Phase 5C requires the selected lag-adjusted simulator")
    prepared_manifest_path = pilot_dir / "manifest.json"
    simulator_config_path = Path(pricing_manifest["simulator_config"])
    forecast_path = Path(pricing_manifest["forecast_checkpoint"])
    weekly_path, windows_path = pilot_dir / "weekly.csv", pilot_dir / "windows.csv"
    if sha256(prepared_manifest_path) != pricing_manifest["source_manifest_sha256"]:
        raise ValueError("Prepared panel differs from Phase 4")
    if sha256(forecast_path) != pricing_manifest["forecast_checkpoint_sha256"]:
        raise ValueError("Forecast checkpoint differs from Phase 4")
    if sha256(simulator_config_path) != pricing_manifest["simulator_config_sha256"]:
        raise ValueError("Simulator selection differs from Phase 4")
    simulator_config = json.loads(simulator_config_path.read_text(encoding="utf-8"))
    if sha256(weekly_path) != simulator_config["prepared_weekly_sha256"]:
        raise ValueError("Weekly data differs from simulator selection")
    selected_seeds = tuple(seeds) if seeds else tuple(pricing_manifest["seeds"])
    if not selected_seeds or not set(selected_seeds).issubset(pricing_manifest["seeds"]):
        raise ValueError("Seed must belong to Phase 4")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    weekly = pd.read_csv(weekly_path, dtype={"sku": str}, parse_dates=["week"])
    windows = pd.read_csv(windows_path, dtype={"sku": str}, parse_dates=["week"])
    simulator = LagAdjustedKernelSimulator(
        float(simulator_config["recent_weight"]),
        ratio_limit=float(simulator_config["ratio_limit"])).fit(weekly)
    weeks_per_sku = pricing_manifest["evaluation_cycles"] // len(pricing_manifest["evaluation_skus"])
    test_rows = _test_rows(windows, pricing_manifest["evaluation_skus"], weeks_per_sku)
    saved_episodes = pd.read_csv(pricing_dir / "policy_episodes.csv", dtype={"sku": str})
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in selected_seeds:
        forecaster = load_forecaster(forecast_path)
        base_state = {key: value.clone() for key, value in forecaster.model.state_dict().items()}
        model_path = output_dir / "models" / f"ppo_feedback_gradient_seed{seed}"
        model, training_stats = _ppo(windows, simulator, forecaster, seed,
                                     int(pricing_manifest["training_timesteps_requested"]),
                                     True, True, model_path)
        trained_forecaster = deepcopy(forecaster)
        parameter_change = max(float(torch.max(torch.abs(value - base_state[key])))
                               for key, value in trained_forecaster.model.state_dict().items())
        if parameter_change <= 0:
            raise AssertionError(f"Seed {seed} had no online forecaster training update")
        replayed, rng_by_sku = _replay(model, forecaster, test_rows, simulator, seed)
        saved = saved_episodes.loc[saved_episodes["policy"].eq("ppo_feedback_gradient")
                                  & saved_episodes["seed"].eq(seed)]
        audit = _compare(replayed, saved)
        state_path = output_dir / "forecasters" / f"ppo_feedback_gradient_seed{seed}.pt"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": trained_forecaster.model.state_dict(),
            "scalers": trained_forecaster.scalers,
            "history_features": HISTORY_FEATURES,
            "current_features": CURRENT_FEATURES,
            "rng_by_sku": rng_by_sku,
            "seed": seed,
            "source_forecast_sha256": sha256(forecast_path),
        }, state_path)
        record = {
            "seed": seed, "policy_zip": str(model_path.with_suffix(".zip").resolve()),
            "policy_zip_sha256": sha256(model_path.with_suffix(".zip")),
            "forecaster_checkpoint": str(state_path.resolve()),
            "forecaster_checkpoint_sha256": sha256(state_path),
            "phase4_policy_zip_sha256": sha256(pricing_dir / "models" / f"ppo_feedback_gradient_seed{seed}.zip"),
            "max_forecaster_parameter_change": parameter_change,
            "training": training_stats, "replay_audit": audit,
        }
        records.append(record)
        print(f"[Phase 5C] seed={seed}: {audit['rows']} test decisions matched; "
              f"max price difference={audit['max_absolute_difference']['chosen_price']:.9g}", flush=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "M5 CA_1 pilot; paired post-training Transformer and PPO for online feedback replay",
        "pricing_manifest_sha256": sha256(pricing_manifest_path),
        "prepared_manifest_sha256": sha256(prepared_manifest_path),
        "simulator_config_sha256": sha256(simulator_config_path),
        "weekly_sha256": sha256(weekly_path),
        "windows_sha256": sha256(windows_path),
        "phase4_episodes_sha256": sha256(pricing_dir / "policy_episodes.csv"),
        "torch_version": torch.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "records": records,
        "interpretation": "Full online test replay uses simulated sales feedback, not observed business outcomes.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--pricing-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    run(args.pilot_dir, args.pricing_dir, args.output_dir,
        tuple(args.seeds) if args.seeds else None)


if __name__ == "__main__":
    main()
