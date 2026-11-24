"""Run the paper's pricing-policy structure on a bounded real-M5 pilot.

This checks the Transformer -> gradient -> PPO -> simulated demand -> online
forecast update loop. It is deliberately labelled a pilot: Song's exact M5
selection, normalized revenue scale and 200-cycle sampling code are not public.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .pricing_benchmark import PricingEnvironment, evaluate_policy, validate_simulator
from .pricing_simulator import KernelDemandSimulator, LagAdjustedKernelSimulator
from .song_forecast import SongDemandForecaster, SongTransformer


PAPER_SEEDS = (42, 123, 456, 789, 1024)
DEFAULT_PILOT = Path(__file__).resolve().parent / "artifacts" / "song_m5_pilot"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_forecaster(checkpoint: Path) -> SongDemandForecaster:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = SongTransformer()
    model.load_state_dict(saved["state_dict"])
    return SongDemandForecaster(model, saved["scalers"])


def _environment(windows: pd.DataFrame, simulator: KernelDemandSimulator, split: str,
                 forecaster: SongDemandForecaster | None = None,
                 *, gradient: bool = True, feedback: bool = False) -> PricingEnvironment:
    return PricingEnvironment(
        windows, simulator, split, forecaster=forecaster, with_gradient=gradient,
        online_forecast_update=feedback, action_mode="absolute",
        include_sku_one_hot=False, update_every=10, feedback_capacity=200,
        update_batch=32,
        max_episode_steps=12 if split == "train" else None,
        random_start=split == "train",
    )


def _ppo(windows: pd.DataFrame, simulator: KernelDemandSimulator,
         forecaster: SongDemandForecaster, seed: int, timesteps: int,
         gradient: bool, feedback: bool, output: Path):
    from stable_baselines3 import PPO

    env = _environment(windows, simulator, "train", forecaster, gradient=gradient, feedback=feedback)
    model = PPO(
        "MlpPolicy", env, seed=seed, verbose=0, device="cpu",
        policy_kwargs={"net_arch": {"pi": [256, 256, 256], "vf": [256, 256, 256]},
                       "activation_fn": nn.ReLU},
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        batch_size=64, n_steps=64,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    return model, {"training_episodes_started": env.episodes_started,
                   "training_distinct_skus_seen": len(env.seen_skus)}


def _offpolicy(name: str, windows: pd.DataFrame, simulator: KernelDemandSimulator,
               seed: int, timesteps: int, output: Path):
    from stable_baselines3 import DDPG, SAC
    from stable_baselines3.common.noise import NormalActionNoise

    env = _environment(windows, simulator, "train")
    cls = {"ddpg": DDPG, "sac": SAC}[name]
    kwargs = {"action_noise": NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1))} if name == "ddpg" else {}
    model = cls(
        "MlpPolicy", env, seed=seed, verbose=0, device="cpu",
        policy_kwargs={"net_arch": [256, 256, 256], "activation_fn": nn.ReLU},
        learning_rate=3e-4, gamma=0.99, batch_size=64,
        learning_starts=min(100, timesteps // 10), buffer_size=max(1000, timesteps + 1),
        **kwargs,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    return model, {"training_episodes_started": env.episodes_started,
                   "training_distinct_skus_seen": len(env.seen_skus)}


def run(pilot_dir: str | Path = DEFAULT_PILOT, output_dir: str | Path | None = None,
        *, forecast_checkpoint: str | Path | None = None,
        simulator_config: str | Path | None = None,
        seeds: tuple[int, ...] = PAPER_SEEDS, timesteps: int = 512,
        eval_cycles: int = 200, cycles_per_sku: int = 20,
        policies: tuple[str, ...] = (
            "fixed_mean", "heuristic", "ddpg", "sac", "ppo_frozen",
            "ppo_feedback_gradient", "ppo_feedback_no_gradient", "ppo_frozen_no_gradient",
        )) -> pd.DataFrame:
    pilot_dir = Path(pilot_dir)
    output_dir = Path(output_dir) if output_dir is not None else pilot_dir / "song_pricing_pilot"
    checkpoint = Path(forecast_checkpoint) if forecast_checkpoint is not None else pilot_dir / "song_forecast" / "song_transformer_seed42.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if timesteps < 64 or eval_cycles < 1 or cycles_per_sku < 1 or not seeds:
        raise ValueError("timesteps >= 64, eval_cycles > 0, cycles_per_sku > 0 and seeds are required")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    weekly = pd.read_csv(pilot_dir / "weekly.csv", dtype={"sku": str}, parse_dates=["week"])
    windows = pd.read_csv(pilot_dir / "windows.csv", dtype={"sku": str}, parse_dates=["week"])
    source_manifest = json.loads((pilot_dir / "manifest.json").read_text(encoding="utf-8"))
    audit_path = pilot_dir / "simulator_audit" / "audit_manifest.json"
    simulator_gate = None
    if audit_path.is_file():
        audit_record = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_record.get("prepared_manifest_sha256") == _sha256(pilot_dir / "manifest.json"):
            simulator_gate = audit_record.get("necessary_validation_gate_passed")
    simulator_config_path = Path(simulator_config) if simulator_config is not None else None
    if simulator_config_path is None:
        simulator = KernelDemandSimulator()
        simulator_variant = "song_kernel"
        simulator_config_hash = None
        recent_weight = None
    else:
        selected_config = json.loads(simulator_config_path.read_text(encoding="utf-8"))
        if selected_config.get("prepared_manifest_sha256") != _sha256(pilot_dir / "manifest.json"):
            raise ValueError("Simulator selection belongs to a different M5 panel")
        if selected_config.get("prepared_weekly_sha256") != _sha256(pilot_dir / "weekly.csv"):
            raise ValueError("Simulator selection belongs to different M5 weekly data")
        if selected_config.get("simulator_variant") != "lag_adjusted_kernel":
            raise ValueError("Unsupported simulator variant")
        recent_weight = float(selected_config["recent_weight"])
        simulator = LagAdjustedKernelSimulator(
            recent_weight, ratio_limit=float(selected_config["ratio_limit"]))
        simulator_variant = "lag_adjusted_kernel"
        simulator_config_hash = _sha256(simulator_config_path)
        simulator_gate = bool(selected_config["necessary_validation_gate_passed"])
    simulator.fit(weekly)
    test_rows = windows.loc[windows["split"].eq("test") & windows["observed_sales"].astype(bool)].copy()
    if eval_cycles % cycles_per_sku:
        raise ValueError("For a balanced pilot, eval_cycles must be divisible by cycles_per_sku")
    n_eval_skus = eval_cycles // cycles_per_sku
    eligible_counts = test_rows.groupby("sku").size()
    train_ranked = [f"{item}@{source_manifest['pilot_store']}" for item in source_manifest["pilot_items"]]
    skus = [sku for sku in train_ranked if sku in simulator.surfaces
            and eligible_counts.get(sku, 0) >= cycles_per_sku][:n_eval_skus]
    if len(skus) != n_eval_skus:
        raise ValueError(f"Only {len(skus)} of {n_eval_skus} evaluation series have enough test weeks")
    test_rows = (test_rows.loc[test_rows["sku"].isin(skus)]
                 .sort_values(["sku", "week"]).groupby("sku", group_keys=False).head(cycles_per_sku))
    if len(test_rows) != eval_cycles:
        raise ValueError(f"Only {len(test_rows)} balanced test cycles are available")
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_simulator(simulator, weekly).to_csv(output_dir / "simulator_observed_price_check.csv", index=False)
    simulator.diagnostics().to_csv(output_dir / "simulator_diagnostics.csv", index=False)
    summaries = []
    episodes = []
    for seed in seeds:
        for name in policies:
            forecaster = None
            training_stats = {}
            if name.startswith("ppo"):
                forecaster = load_forecaster(checkpoint)
                gradient = name in {"ppo_frozen", "ppo_feedback_gradient"}
                feedback = name in {"ppo_feedback_gradient", "ppo_feedback_no_gradient"}
                model, training_stats = _ppo(windows, simulator, forecaster, seed, timesteps,
                                             gradient, feedback,
                                             output_dir / "models" / f"{name}_seed{seed}")
                env = _environment(test_rows, simulator, "test", forecaster, gradient=gradient, feedback=feedback)
                policy = lambda observation, _: model.predict(observation, deterministic=True)[0]
            elif name in {"ddpg", "sac"}:
                model, training_stats = _offpolicy(name, windows, simulator, seed, timesteps,
                                                   output_dir / "models" / f"{name}_seed{seed}")
                env = _environment(test_rows, simulator, "test")
                policy = lambda observation, _: model.predict(observation, deterministic=True)[0]
            elif name in {"fixed_mean", "heuristic", "kernel_grid"}:
                env = _environment(test_rows, simulator, "test")
                policy = None
            else:
                raise ValueError(f"Unknown policy {name}")
            frame, summary = evaluate_policy(env, name, policy, seed=seed)
            if len(frame) != eval_cycles:
                raise AssertionError(f"{name} evaluated {len(frame)} instead of {eval_cycles} cycles")
            summaries.append(summary)
            summaries[-1].update(training_stats)
            episodes.append(frame)
            print(f"[M5 pilot pricing] seed={seed} {name}: revenue={summary['simulated_revenue']:.2f}", flush=True)
    runs = pd.DataFrame(summaries)
    fixed = runs.loc[runs["policy"].eq("fixed_mean"), ["seed", "simulated_revenue"]].rename(
        columns={"simulated_revenue": "fixed_revenue"})
    if not fixed.empty:
        runs = runs.merge(fixed, on="seed", how="left", validate="many_to_one")
        runs["uplift_vs_fixed_pct"] = 100 * (runs["simulated_revenue"] / runs["fixed_revenue"] - 1)
    runs.to_csv(output_dir / "policy_runs.csv", index=False)
    pd.concat(episodes, ignore_index=True).to_csv(output_dir / "policy_episodes.csv", index=False)
    aggregate = runs.groupby("policy", as_index=False).agg(
        seeds=("seed", "nunique"), revenue_mean=("simulated_revenue", "mean"),
        revenue_sd=("simulated_revenue", "std"), regret_mean=("expected_regret", "mean"),
    )
    aggregate.to_csv(output_dir / "policy_aggregate.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "PILOT only; 20 train-ranked item-store series at one store",
        "source_manifest": str((pilot_dir / "manifest.json").resolve()),
        "source_manifest_sha256": _sha256(pilot_dir / "manifest.json"),
        "source_dataset": source_manifest["dataset"],
        "forecast_checkpoint": str(checkpoint.resolve()),
        "forecast_checkpoint_sha256": _sha256(checkpoint),
        "simulator_validation_gate_passed": simulator_gate,
        "scientific_status": (
            "functional pilot only; observed-price validation gate failed"
            if simulator_gate is False else
            "simulator audit gate not available" if simulator_gate is None else
            "observed-price gate passed; counterfactual demand remains unverified"
        ),
        "simulator": ("Train-only Gaussian kernel with Silverman price bandwidth; "
                      "optional validation-tuned last-demand level adjustment"),
        "simulator_variant": simulator_variant,
        "simulator_config": str(simulator_config_path.resolve()) if simulator_config_path else None,
        "simulator_config_sha256": simulator_config_hash,
        "simulator_recent_weight": recent_weight,
        "action_space": "Full observed training price support, affine mapping from [-1, 1]",
        "policy_network": "Three 256-unit ReLU layers; PPO clip=.2, gamma=.99, GAE=.95, lr=3e-4, batch=64",
        "training_timesteps_requested": timesteps,
        "training_episode_weeks": 12,
        "training_episode_start": "Random train SKU and random chronological 12-week segment; test always uses the fixed 20-week segment",
        "training_note": "SB3 PPO rounds total timesteps up to a multiple of n_steps=64",
        "evaluation_cycles": eval_cycles,
        "evaluation_selection": f"First {cycles_per_sku} chronological observed test weeks for each of {len(skus)} train-ranked series",
        "evaluation_skus": skus,
        "seeds": seeds, "policies": policies,
        "feedback": "Every 10 simulated decisions, a batch of up to 32 from the most recent 200",
        "reward_note": "Simulated price*units divided by one global train revenue constant; reported totals are unscaled simulated price*units, not the paper's normalized M5 units",
        "comparison_limit": "The author's exact M5 simulator/series selection/normalization code is unavailable; pilot revenues are not numerically comparable to Table 3",
    }
    (output_dir / "pricing_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--forecast-checkpoint", type=Path)
    parser.add_argument("--simulator-config", type=Path,
                        help="Optional selected_simulator.json; omit for the Song-style kernel")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(PAPER_SEEDS))
    parser.add_argument("--timesteps", type=int, default=512)
    parser.add_argument("--eval-cycles", type=int, default=200)
    parser.add_argument("--cycles-per-sku", type=int, default=20)
    parser.add_argument("--policies", nargs="+", default=["fixed_mean", "heuristic", "ddpg", "sac",
                                                    "ppo_frozen", "ppo_feedback_gradient",
                                                    "ppo_feedback_no_gradient", "ppo_frozen_no_gradient"])
    args = parser.parse_args()
    aggregate = run(args.pilot_dir, args.output_dir, forecast_checkpoint=args.forecast_checkpoint,
                    simulator_config=args.simulator_config,
                    seeds=tuple(args.seeds), timesteps=args.timesteps,
                    eval_cycles=args.eval_cycles, cycles_per_sku=args.cycles_per_sku,
                    policies=tuple(args.policies))
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
