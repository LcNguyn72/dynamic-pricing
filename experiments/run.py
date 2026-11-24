"""Reproducible research pipeline alongside the existing 20-product demo.

Examples:
    python -m experiments.run --stage forecast
    python -m experiments.run --stage pricing --seeds 42 --timesteps 512
    python -m experiments.run --stage all --split-profile retail2_80_20
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .data import DEFAULT_SELECTED_PATH, DEFAULT_WEEKLY_PATH, SPLIT_PROFILES, make_windows, prepare_experiment
from .forecast_benchmark import DEFAULT_MODELS, run_forecast_benchmark
from .pricing_benchmark import PricingEnvironment, evaluate_policy, train_ppo, validate_simulator
from .pricing_simulator import KernelDemandSimulator


DEFAULT_ROOT = Path(__file__).resolve().parent / "artifacts" / "online_retail_ii_20"
DEFAULT_M5_ROOT = Path(__file__).resolve().parent / "artifacts" / "m5"
PAPER_SEEDS = (42, 123, 456, 789, 1024)


def _forecast_loader(model_dir: Path):
    from .forecast_torch import load_price_forecaster

    return load_price_forecaster(model_dir, train_only=True)


def _stable_baselines_policy(algorithm, windows, simulator, seed, timesteps, model_path):
    from stable_baselines3 import DDPG, SAC

    env = PricingEnvironment(windows, simulator, "train", forecaster=None, with_gradient=False)
    cls = {"ddpg": DDPG, "sac": SAC}[algorithm]
    model = cls("MlpPolicy", env, seed=seed, verbose=0, device="cpu",
                learning_rate=3e-4, gamma=0.99,
                policy_kwargs={"net_arch": [256, 256]},
                learning_starts=min(100, max(1, timesteps // 10)),
                buffer_size=max(1000, timesteps + 1), batch_size=64)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    return model


def run_pricing_benchmark(
    weekly: pd.DataFrame,
    windows: pd.DataFrame,
    output_dir: str | Path,
    *,
    forecast_model_dir: str | Path | None = None,
    seeds: tuple[int, ...] = PAPER_SEEDS,
    timesteps: int = 2048,
    learned_policies: bool = True,
    ablations: bool = False,
    dataset_name: str = "Online Retail II selected 20 SKU-weeks",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare policies under the *same* fitted simulator and held-out weeks.

    The simulator is fitted only to train. Baseline methods and agents share
    bounds, test calendar states and seeded noise. Its validation error is
    reported, but no observational data identify counterfactual price effects.
    """
    if not seeds:
        raise ValueError("At least one seed is required")
    if timesteps < 1:
        raise ValueError("timesteps must be positive")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    simulator = KernelDemandSimulator().fit(weekly)
    simulator.diagnostics().to_csv(out / "simulator_diagnostics.csv", index=False)
    validation = validate_simulator(simulator, weekly)
    validation.to_csv(out / "simulator_observed_price_check.csv", index=False)
    source = Path(forecast_model_dir) if forecast_model_dir is not None else None
    if learned_policies and (source is None or not (source / "transformer_price_train_only.pt").exists()):
        raise FileNotFoundError("Pricing PPO needs a train-only Transformer+price checkpoint from the same split")

    episodes: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for seed in seeds:
        for name in ("fixed_mean", "heuristic"):
            env = PricingEnvironment(windows, simulator, "test")
            frame, summary = evaluate_policy(env, name, seed=seed)
            episodes.append(frame)
            summaries.append(summary)
            print(f"[pricing] seed={seed} {name}: simulated revenue {summary['simulated_revenue']:.2f}")
        if not learned_policies:
            continue
        assert source is not None
        for algorithm in ("ddpg", "sac"):
            model = _stable_baselines_policy(algorithm, windows, simulator, seed, timesteps,
                                             out / "models" / f"{algorithm}_seed{seed}")
            env = PricingEnvironment(windows, simulator, "test")
            frame, summary = evaluate_policy(
                env, algorithm, lambda obs, _: model.predict(obs, deterministic=True)[0], seed=seed
            )
            episodes.append(frame)
            summaries.append(summary)
            print(f"[pricing] seed={seed} {algorithm}: simulated revenue {summary['simulated_revenue']:.2f}")
        configs = [
            ("ppo_frozen", True, False),
            ("ppo_feedback_gradient", True, True),
        ]
        if ablations:
            configs.extend([
                ("ppo_feedback_no_gradient", False, True),
                ("ppo_frozen_no_gradient", False, False),
            ])
        for name, with_gradient, online_update in configs:
            forecaster = _forecast_loader(source)
            model = train_ppo(
                windows, simulator, forecaster, seed=seed, total_timesteps=timesteps,
                with_gradient=with_gradient, online_forecast_update=online_update,
                output_path=out / "models" / f"{name}_seed{seed}",
            )
            # Keep the forecast weights reached during policy training. The
            # test only reveals simulated outcomes sequentially during replay.
            env = PricingEnvironment(
                windows, simulator, "test", forecaster,
                with_gradient=with_gradient, online_forecast_update=online_update,
            )
            frame, summary = evaluate_policy(
                env, name, lambda obs, _: model.predict(obs, deterministic=True)[0], seed=seed
            )
            episodes.append(frame)
            summaries.append(summary)
            print(f"[pricing] seed={seed} {name}: simulated revenue {summary['simulated_revenue']:.2f}")
    episode_df = pd.concat(episodes, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    baseline = summary_df.loc[summary_df["policy"].eq("fixed_mean"), ["seed", "simulated_revenue"]].rename(
        columns={"simulated_revenue": "fixed_revenue"}
    )
    summary_df = summary_df.merge(baseline, on="seed", how="left", validate="many_to_one")
    summary_df["uplift_vs_fixed_pct"] = 100 * (summary_df["simulated_revenue"] / summary_df["fixed_revenue"] - 1)
    aggregate = summary_df.groupby("policy", as_index=False).agg(
        runs=("seed", "count"),
        revenue_mean=("simulated_revenue", "mean"),
        revenue_sd=("simulated_revenue", "std"),
        uplift_mean_pct=("uplift_vs_fixed_pct", "mean"),
        regret_mean=("expected_regret", "mean"),
    )
    summary_df.to_csv(out / "policy_runs.csv", index=False)
    aggregate.to_csv(out / "policy_aggregate.csv", index=False)
    episode_df.to_csv(out / "policy_episodes.csv", index=False)
    check = validation.groupby("split").agg(
        n=("absolute_error", "size"), mae=("absolute_error", "mean")
    ).to_dict(orient="index")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "simulator": "Train-only Gaussian kernel regression of observed qty on price, week of year, known M5 event/SNAP context when present",
        "simulator_check_at_observed_prices": check,
        "simulator_limitation": "Observed-price error does not validate counterfactual demand; simulated revenue is not realized revenue",
        "policy_training_timesteps": timesteps if learned_policies else 0,
        "seeds": seeds,
        "price_action": "Within observed train price min/max and ±10% of current price",
        "initial_price": "Train-period mean price for all policies",
        "reward_for_rl": "Simulated price*units divided by one global train mean revenue constant; proportional to reported total revenue",
        "reporting_metric": "Sum of unscaled simulated price*units over identical SKU-test weeks",
        "oracle_gap": "Expected revenue gap to the best of 100 prices across full training price support; this oracle may not be reachable in one ±10% step",
        "evaluation_order": "Independent SKU test episodes; forecast weights and feedback reset before each SKU to avoid cross-SKU future leakage",
        "forecaster_checkpoint": str(source.resolve()) if source is not None else None,
        "forecaster_training_scope": "Train-only checkpoint; validation may select epoch, but policy/simulator never learn validation labels",
        "feedback": "Forecast fine-tuned every 10 simulated decisions on a batch of at most 32 from the latest 200",
        "ablations": ablations,
    }
    with (out / "pricing_run.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, default=str)
    return summary_df, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("data", "forecast", "pricing", "all"), default="all")
    parser.add_argument("--dataset", choices=("retail2", "m5"), default="retail2")
    parser.add_argument("--split-profile", choices=SPLIT_PROFILES, default="paper_70_15_15")
    parser.add_argument("--weekly-path", type=Path, default=DEFAULT_WEEKLY_PATH)
    parser.add_argument("--selected-path", type=Path, default=DEFAULT_SELECTED_PATH)
    parser.add_argument("--m5-dir", type=Path)
    parser.add_argument("--m5-sales-file", default="sales_train_validation.csv")
    parser.add_argument("--m5-item-id", action="append")
    parser.add_argument("--m5-store-id", action="append")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument("--torch-epochs", type=int, default=25)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--no-learned-policies", action="store_true")
    parser.add_argument("--ablations", action="store_true")
    args = parser.parse_args()
    default_root = DEFAULT_M5_ROOT if args.dataset == "m5" else DEFAULT_ROOT
    base_root = args.output_root or default_root
    root = base_root if args.split_profile == "paper_70_15_15" else base_root / args.split_profile
    if args.dataset == "retail2":
        weekly, windows, manifest = prepare_experiment(
            root, args.weekly_path, args.selected_path, 12, args.split_profile
        )
    else:
        if args.m5_dir is None:
            parser.error("--m5-dir is required for --dataset m5")
        if not args.m5_item_id and not args.m5_store_id:
            parser.error("Specify at least one --m5-item-id or --m5-store-id to bound memory use")
        from .m5 import load_m5_weekly

        weekly, manifest = load_m5_weekly(
            args.m5_dir, sales_file=args.m5_sales_file,
            item_ids=args.m5_item_id, store_ids=args.m5_store_id,
            split_profile=args.split_profile,
        )
        manifest["lookback_weeks"] = 12
        windows = make_windows(weekly, lookback=12)
        root.mkdir(parents=True, exist_ok=True)
        weekly.to_csv(root / "weekly_panel.csv", index=False)
        windows.to_csv(root / "forecast_windows.csv", index=False)
        with (root / "data_manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, default=str)
    if args.stage in {"forecast", "all"}:
        run_forecast_benchmark(
            windows, root / "forecast", manifest,
            models=args.models, torch_epochs=args.torch_epochs, strict=True,
        )
    if args.stage in {"pricing", "all"}:
        run_pricing_benchmark(
            weekly, windows, root / "pricing_verified",
            forecast_model_dir=root / "forecast" / "models",
            seeds=tuple(args.seeds), timesteps=args.timesteps,
            learned_policies=not args.no_learned_policies,
            ablations=args.ablations,
            dataset_name=manifest["dataset"],
        )


if __name__ == "__main__":
    main()
