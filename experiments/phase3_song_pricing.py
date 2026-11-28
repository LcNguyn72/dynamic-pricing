"""Run the Song PPO pricing experiment on the frozen M5 V2 validation cohort."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import stable_baselines3
import torch
from stable_baselines3 import PPO

from .data import make_windows
from .phase1_finalize import _observed
from .phase2_song import (
    M5_COLUMNS, PHASE1_ROOT, PROTOCOL, OUTPUT_ROOT as PHASE2_ROOT,
    _read_json, _verify, _write_json,
)
from .pricing_benchmark import PricingEnvironment, evaluate_policy, validate_simulator
from .pricing_simulator import KernelDemandSimulator
from .song_forecast import (
    CURRENT_FEATURES, HISTORY_FEATURES, LOOKBACK, SongDemandForecaster,
    SongTransformer, _sha256,
)
from .song_pricing_pilot import _offpolicy, _ppo

OUTPUT_ROOT = PHASE1_ROOT.with_name("phase3_v2")
BASELINES = ("fixed_mean", "heuristic", "ddpg", "sac")
MAIN_POLICY = "ppo_feedback_gradient"
PPO_SETTINGS = {
    "ppo_frozen": (True, False),
    MAIN_POLICY: (True, True),
    "ppo_feedback_no_gradient": (False, True),
    "ppo_frozen_no_gradient": (False, False),
}
FEEDBACK = {"update_every": 10, "feedback_capacity": 200, "update_batch": 32}
FEEDBACK_TORCH_SEED_OFFSET = 2_000_003
PPO_HYPERPARAMETERS = {"actor_critic_layers": [256, 256, 256], "activation": "relu",
                       "learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95,
                       "clip_range": 0.2, "batch_size": 64, "n_steps": 64}
TRAIN_HISTORY = [f"{field}_lag_{lag}" for lag in range(1, LOOKBACK + 1)
                 for field in ("qty", "price")]
CALENDAR = ["event_days", "snap_days", *(f"{field}_lag_{lag}" for lag in range(1, LOOKBACK + 1)
                                         for field in ("event_days", "snap_days"))]
KEYS = ["sku", "week"]


def validation_inputs(windows: pd.DataFrame, skus: list[str], start: pd.Timestamp,
                      weeks: int) -> pd.DataFrame:
    """Keep calendar context and pre-episode observed history, withholding episode sales."""
    schedule = pd.date_range(start=start, periods=weeks, freq="7D")
    columns = [*KEYS, "split", "observed_sales", *CALENDAR, *TRAIN_HISTORY]
    selected = windows.loc[
        windows["sku"].isin(skus) & windows["split"].eq("validation")
        & windows["week"].isin(schedule) & _observed(windows), columns,
    ].copy()
    selected = selected.sort_values(KEYS).reset_index(drop=True)
    for sku in skus:
        dates = pd.DatetimeIndex(selected.loc[selected["sku"].eq(sku), "week"])
        if not dates.equals(schedule):
            raise ValueError(f"{sku}: incomplete validation calendar or unpriced week in fixed window")
    # Subsequent history is built by PricingEnvironment from simulated outcomes.
    selected[TRAIN_HISTORY] = selected[TRAIN_HISTORY].astype(float)
    selected.loc[selected["week"].ne(start), TRAIN_HISTORY] = np.nan
    return selected


def _load_cohort(root: Path, phase1: dict, protocol: dict) -> tuple[pd.DataFrame, list[str]]:
    report = phase1["m5"]
    cohort_path = root / "m5_pricing_cohort.csv"
    _verify(cohort_path, report["pricing_cohort_sha256"])
    cohort = pd.read_csv(cohort_path, dtype=str)
    stores = protocol["datasets"]["m5"]["store_ids"]
    skus = protocol["pricing"]["cohort_skus"]
    if list(cohort["store"]) != stores or list(cohort["sku"]) != skus:
        raise ValueError("Phase 1 pricing cohort differs from the V2 protocol")
    source = {entry["store"]: entry for entry in report["stores"]}
    parts = []
    for store, sku in zip(stores, skus, strict=True):
        weekly_path = root / "m5" / f"weekly_{store}.csv"
        _verify(weekly_path, source[store]["weekly_sha256"])
        part = pd.read_csv(weekly_path, usecols=M5_COLUMNS, dtype={"sku": str}, parse_dates=["week"])
        selected = part.loc[part["sku"].eq(sku) & part["split"].isin(("train", "validation"))]
        if selected.empty or selected.duplicated(KEYS).any():
            raise ValueError(f"{sku}: missing or duplicate weekly rows")
        parts.append(selected)
    return pd.concat(parts, ignore_index=True), skus


class FeedbackForecaster(SongDemandForecaster):
    """Keep dropout and shuffling independent from PPO's exploration draws."""

    def __init__(self, model, scalers, *, seed: int, feedback_rng_state: torch.Tensor | None = None):
        super().__init__(model, scalers)
        if feedback_rng_state is None:
            self.feedback_rng_state = torch.Generator().manual_seed(
                seed + FEEDBACK_TORCH_SEED_OFFSET).get_state()
        else:
            self.feedback_rng_state = feedback_rng_state.clone()

    def fine_tune(self, samples, epochs: int = 1, lr: float = 1e-4, batch_size: int = 32) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.feedback_rng_state)
            super().fine_tune(samples, epochs=epochs, lr=lr, batch_size=batch_size)
            self.feedback_rng_state = torch.get_rng_state()


def _forecaster(saved: dict, skus: list[str], seed: int) -> FeedbackForecaster:
    if (saved["history_features"] != HISTORY_FEATURES or saved["current_features"] != CURRENT_FEATURES
            or saved["lookback"] != LOOKBACK or not set(skus).issubset(saved["scalers"])):
        raise ValueError("Transformer checkpoint is incompatible with the pricing cohort or features")
    model = SongTransformer()
    model.load_state_dict(saved["state_dict"])
    return FeedbackForecaster(model, {sku: saved["scalers"][sku] for sku in skus},
                              seed=seed, feedback_rng_state=saved.get("feedback_rng_state"))


class ValidationEnvironment(PricingEnvironment):
    """Log demand at the selected price before updating the forecaster."""

    def __init__(self, inputs: pd.DataFrame, simulator: KernelDemandSimulator,
                 diagnostic: SongDemandForecaster | None, forecaster: SongDemandForecaster | None,
                 *, gradient: bool, feedback: bool):
        if forecaster is None and diagnostic is None:
            raise ValueError("Validation requires a forecaster for reporting")
        super().__init__(inputs, simulator, "validation", forecaster, with_gradient=gradient,
                         online_forecast_update=feedback, action_mode="absolute",
                         include_sku_one_hot=False, **FEEDBACK)
        self.diagnostic = diagnostic

    def step(self, action: np.ndarray):
        agent = self.forecaster if self.forecaster is not None else self.diagnostic
        price = self.price_from_action(float(np.asarray(action).reshape(-1)[0]))
        prediction = agent.predict_q(self._current_row(), price)
        observation, reward, terminated, truncated, info = super().step(action)
        info["predicted_qty"] = prediction
        return observation, reward, terminated, truncated, info


def _evaluate(name: str, model, inputs: pd.DataFrame, simulator: KernelDemandSimulator,
              diagnostic: SongDemandForecaster | None, forecaster: SongDemandForecaster | None,
              seed: int, *, action_policy=None) -> tuple[pd.DataFrame, dict]:
    gradient, feedback = PPO_SETTINGS.get(name, (False, False))
    env = ValidationEnvironment(inputs, simulator, diagnostic, forecaster,
                                gradient=gradient, feedback=feedback)

    def choose(observation, _):
        return model.predict(observation, deterministic=True)[0]

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        policy = action_policy if model is None else choose
        rows, summary = evaluate_policy(env, name, policy, seed)
    expected_keys = inputs.sort_values(KEYS)[KEYS].copy().reset_index(drop=True)
    expected_keys["week"] = expected_keys["week"].dt.strftime("%Y-%m-%d")
    if not rows[KEYS].equals(expected_keys):
        raise ValueError(f"{name}: policy decisions differ from the fixed validation keys")
    diagnostics = simulator.diagnostics().set_index("sku")
    rows["train_price_min"] = rows["sku"].map(diagnostics["price_min"])
    rows["train_price_max"] = rows["sku"].map(diagnostics["price_max"])
    if not (rows["chosen_price"].ge(rows["train_price_min"] - 1e-9)
            & rows["chosen_price"].le(rows["train_price_max"] + 1e-9)).all():
        raise ValueError(f"{name}: action outside the training price support")
    step = rows.groupby("sku").cumcount() + 1
    expected_updates = step // FEEDBACK["update_every"] if feedback else step * 0
    if not rows["forecast_update_count"].eq(expected_updates).all():
        raise ValueError(f"{name}: unexpected validation feedback update schedule")
    if not np.isfinite(rows.select_dtypes(include="number").to_numpy()).all():
        raise ValueError(f"{name}: non-finite policy output")
    rows["cumulative_simulated_revenue"] = rows["simulated_revenue"].cumsum()
    summary["validation_forecast_updates"] = int(rows.groupby("sku")["forecast_update_count"].max().sum())
    return rows, summary


def _audit_simulator(simulator: KernelDemandSimulator, weekly: pd.DataFrame,
                     windows: pd.DataFrame, out: Path) -> dict:
    checked = validate_simulator(simulator, weekly, splits=("validation",))
    checked = checked.merge(windows[[*KEYS, "qty_lag_1"]], on=KEYS, how="left", validate="one_to_one")
    if checked["qty_lag_1"].isna().any():
        raise ValueError("An observed-price validation row lacks the previous week's quantity")
    bounds = simulator.diagnostics()
    bounds.to_csv(out / "simulator_train_support.csv", index=False)
    checked = checked.merge(bounds[["sku", "price_min", "price_max"]], on="sku", validate="many_to_one")
    checked["price_in_train_support"] = (checked["price"].ge(checked["price_min"] - 1e-9)
                                          & checked["price"].le(checked["price_max"] + 1e-9))
    checked.to_csv(out / "simulator_observed_price_predictions.csv", index=False)
    in_support = checked.loc[checked["price_in_train_support"]]
    if in_support.empty:
        raise ValueError("No comparable validation rows at observed prices within train support")
    scores = []
    for label, rows in (("all", checked), ("in_train_support", in_support)):
        for model, column in (("kernel_simulator", "simulator_pred_qty"), ("last_week", "qty_lag_1")):
            error = rows[column] - rows["qty"]
            scores.append({"segment": label, "model": model, "n": len(rows),
                           "rmse": float(np.sqrt((error ** 2).mean())), "mae": float(error.abs().mean())})
    pd.DataFrame(scores).to_csv(out / "simulator_observed_price_metrics.csv", index=False)
    metrics = {row["model"]: row["rmse"] for row in scores if row["segment"] == "in_train_support"}
    return {"validation_rows": len(checked), "rows_in_train_support": len(in_support),
            "kernel_rmse_in_support": metrics["kernel_simulator"],
            "last_week_rmse_in_support": metrics["last_week"],
            "historical_price_gate_passed": metrics["kernel_simulator"] <= metrics["last_week"]}


def _save_pair(out: Path, name: str, seed: int, forecaster: FeedbackForecaster,
               saved: dict, source_sha: str) -> dict:
    policy_file = out / "models" / f"{name}.zip"
    forecaster_file = out / "forecasters" / f"{name}.pt"
    forecaster_file.parent.mkdir(exist_ok=True)
    weights = forecaster.model.state_dict()
    change = max(float((param - saved["state_dict"][key]).abs().max()) for key, param in weights.items())
    if (change > 0) != PPO_SETTINGS[name][1]:
        raise ValueError(f"{name}: unexpected change in Transformer weights during training")
    policy_hash = _sha256(policy_file)
    torch.save({
        "state_dict": weights, "scalers": forecaster.scalers, "seed": seed, "policy": name,
        "feedback_rng_state": forecaster.feedback_rng_state,
        "history_features": HISTORY_FEATURES, "current_features": CURRENT_FEATURES, "lookback": LOOKBACK,
        "phase2_checkpoint_sha256": source_sha, "matched_ppo_sha256": policy_hash,
    }, forecaster_file)
    return {"policy_file": policy_file.relative_to(out).as_posix(), "policy_sha256": policy_hash,
            "forecaster_file": forecaster_file.relative_to(out).as_posix(),
            "forecaster_sha256": _sha256(forecaster_file), "transformer_max_weight_change": change}


def replay(out: Path) -> dict:
    """Replay the main policy after loading both checkpoint files and the frozen inputs."""
    torch.set_num_threads(1)
    manifest = _read_json(out / "run.json")
    _verify(out / "simulator_config.json", manifest["simulator_config_sha256"])
    config = _read_json(out / "simulator_config.json")
    _verify(out / "simulator_train.csv", config["train_sha256"])
    _verify(out / "validation_inputs.csv", manifest["validation_inputs_sha256"])
    _verify(out / "policy_episodes.csv", manifest["policy_episodes_sha256"])
    pair = manifest["pairs"][MAIN_POLICY]
    _verify(out / pair["policy_file"], pair["policy_sha256"])
    _verify(out / pair["forecaster_file"], pair["forecaster_sha256"])
    saved = torch.load(out / pair["forecaster_file"], map_location="cpu", weights_only=True)
    if (saved["matched_ppo_sha256"] != pair["policy_sha256"] or saved["policy"] != MAIN_POLICY
            or saved["seed"] != manifest["seed"]
            or saved["phase2_checkpoint_sha256"] != manifest["phase2_checkpoint_sha256"]):
        raise ValueError("The two checkpoint files do not belong to the same trained pair")
    csv_options = {"dtype": {"sku": str}, "parse_dates": ["week"]}
    train = pd.read_csv(out / "simulator_train.csv", **csv_options)
    inputs = pd.read_csv(out / "validation_inputs.csv", **csv_options)
    simulator = KernelDemandSimulator(config["seasonal_bandwidth_weeks"]).fit(train)
    model = PPO.load(out / pair["policy_file"], device="cpu")
    actual, _ = _evaluate(MAIN_POLICY, model, inputs, simulator, None,
                          _forecaster(saved, manifest["skus"], manifest["seed"]), manifest["seed"])
    reference = pd.read_csv(out / "policy_episodes.csv", dtype={"sku": str})
    reference = reference.loc[reference["policy"].eq(MAIN_POLICY)].reset_index(drop=True)
    if not actual[KEYS].equals(reference[KEYS]):
        raise ValueError("Replayed validation keys differ from the recorded run")
    fields = ["chosen_price", "predicted_qty", "simulated_units", "simulated_revenue",
              "expected_regret", "forecast_update_count"]
    differences = (actual[fields] - reference[fields]).abs().max().to_dict()
    if any(not np.isfinite(value) or value > 1e-5 for value in differences.values()):
        raise ValueError(f"Validation replay differs: {differences}")
    return {"n_rows": len(actual), "max_absolute_difference": differences}


def _compare_revenue(runs: pd.DataFrame) -> pd.DataFrame:
    for policy, field in (("fixed_mean", "uplift_vs_fixed_pct"),
                          ("ppo_frozen", "uplift_vs_ppo_frozen_pct")):
        base = float(runs.loc[runs["policy"].eq(policy), "simulated_revenue"].iloc[0])
        runs[field] = 100 * (runs["simulated_revenue"] / base - 1) if base > 0 else np.nan
    return runs


def run(root: Path, phase2: Path, output: Path, *, seed: int, timesteps: int, weeks: int) -> dict:
    if output.exists():
        raise FileExistsError(f"Run directory already exists: {output}")
    if timesteps < 64 or timesteps % 64 or weeks < FEEDBACK["update_every"]:
        raise ValueError("Use a positive multiple of 64 timesteps and at least 10 validation weeks")
    torch.set_num_threads(1)
    started = perf_counter()
    phase1_path = root / "phase1_manifest.json"
    phase1 = _read_json(phase1_path)
    protocol = _read_json(PROTOCOL)
    if _sha256(PROTOCOL) != phase1["protocol_sha256"]:
        raise ValueError("Phase 1 data differs from the locked V2 protocol")
    cfg = protocol["datasets"]["m5"]
    pricing = protocol["pricing"]
    spec_feedback = (pricing["forecast_feedback_every_interactions"],
                     pricing["feedback_recent_capacity"], pricing["feedback_batch_size"])
    if spec_feedback != tuple(FEEDBACK.values()) or weeks > cfg["validation_weeks"]:
        raise ValueError("The feedback schedule or validation duration differs from the locked protocol")
    phase2_run = _read_json(phase2 / "run.json")
    checkpoint = phase2 / "song_transformer.pt"
    source_sha = phase2_run["checkpoint_sha256"]
    _verify(checkpoint, source_sha)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved["dataset"] != "m5" or saved["fit_scope"] != "train_only"
            or saved["phase1_manifest_sha256"] != _sha256(phase1_path)):
        raise ValueError("Phase 3 requires the Phase 2 M5 train-only Transformer on identical Phase 1 data")
    weekly, skus = _load_cohort(root, phase1, protocol)
    windows = make_windows(weekly, LOOKBACK)
    history = windows.loc[windows["split"].eq("train") & _observed(windows)]
    if set(history["sku"]) != set(skus):
        raise ValueError("Missing usable training history for a locked Phase 1 pricing series")
    inputs = validation_inputs(windows, skus, pd.Timestamp(cfg["validation_start"]), weeks)
    sim_train = weekly.loc[weekly["split"].eq("train") & _observed(weekly),
                           [*KEYS, "split", "qty", "price", "observed_sales", "event_days", "snap_days"]]
    simulator = KernelDemandSimulator().fit(sim_train)
    if set(simulator.surfaces) != set(skus):
        raise ValueError("Simulator could not fit all ten frozen series")
    diagnostic = _forecaster(saved, skus, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".phase3-", dir=output.parent) as work:
        out = Path(work) / "run"
        out.mkdir()
        print(f"[Phase 3] Seed {seed}: {len(skus)} series, {len(inputs)} validation decisions per policy", flush=True)
        sim_train.to_csv(out / "simulator_train.csv", index=False)
        inputs.to_csv(out / "validation_inputs.csv", index=False)
        _write_json(out / "simulator_config.json", {
            "name": "gaussian_kernel_silverman", "seasonal_bandwidth_weeks": simulator.seasonal_bandwidth_weeks,
            "fit_scope": "observed train only", "train_sha256": _sha256(out / "simulator_train.csv"),
            "reward_scale": max(float((sim_train["price"] * sim_train["qty"]).mean()), 1.0),
        })
        simulator_scores = _audit_simulator(simulator, weekly, windows, out)
        print(f"[Phase 3] Simulator validation: {simulator_scores}", flush=True)
        episodes, summaries, pairs = [], [], {}
        for name in (*BASELINES, *PPO_SETTINGS):
            print(f"[Phase 3] Running {name}", flush=True)
            forecaster, model, stats = None, None, {}
            gradient, feedback = PPO_SETTINGS.get(name, (False, False))
            model_file = out / "models" / name
            if name in PPO_SETTINGS:
                forecaster = _forecaster(saved, skus, seed)
                model, stats = _ppo(history, simulator, forecaster, seed, timesteps,
                                    gradient, feedback, model_file)
                pairs[name] = _save_pair(out, name, seed, forecaster, saved, source_sha)
            elif name in {"ddpg", "sac"}:
                model, stats = _offpolicy(name, history, simulator, seed, timesteps, model_file)
            used = int(model.num_timesteps) if model is not None else 0
            updates = int(model.get_env().get_attr("feedback_updates")[0]) if model is not None else 0
            if (model is not None and used != timesteps) or updates != (used // FEEDBACK["update_every"] if feedback else 0):
                raise ValueError(f"{name}: training budget or feedback schedule differs")
            frame, summary = _evaluate(name, model, inputs, simulator, diagnostic, forecaster, seed)
            summaries.append({**summary, **stats, "training_timesteps": used,
                              "training_forecast_updates": updates})
            episodes.append(frame)
            print(f"[Phase 3] {name}: simulated validation revenue {summary['simulated_revenue']:.2f}", flush=True)
        runs = _compare_revenue(pd.DataFrame(summaries))
        runs.to_csv(out / "policy_summary.csv", index=False)
        pd.concat(episodes, ignore_index=True).to_csv(out / "policy_episodes.csv", index=False)
        result = {
            "protocol_id": protocol["protocol_id"], "dataset": "m5", "split": "validation",
            "seed": seed, "phase2_checkpoint_seed": saved["seed"], "skus": skus,
            "phase1_manifest_sha256": _sha256(phase1_path), "protocol_sha256": phase1["protocol_sha256"],
            "pricing_cohort_sha256": phase1["m5"]["pricing_cohort_sha256"],
            "phase2_checkpoint_sha256": source_sha, "training_timesteps_per_learned_policy": timesteps,
            "ppo_hyperparameters": PPO_HYPERPARAMETERS, "feedback": FEEDBACK,
            "feedback_torch_seed_offset": FEEDBACK_TORCH_SEED_OFFSET,
            "validation_weeks_per_sku": weeks, "decisions_per_policy": len(inputs),
            "first_validation_week": inputs["week"].min().date().isoformat(),
            "last_validation_week": inputs["week"].max().date().isoformat(),
            "validation_inputs_sha256": _sha256(out / "validation_inputs.csv"),
            "policy_episodes_sha256": _sha256(out / "policy_episodes.csv"),
            "simulator_config_sha256": _sha256(out / "simulator_config.json"),
            "simulator_validation": simulator_scores, "pairs": pairs,
            "torch_version": str(torch.__version__), "sb3_version": stable_baselines3.__version__,
            "interpretation": "Single-seed simulated validation; revenue is raw action price times simulated units.",
        }
        _write_json(out / "run.json", result)
        result["replay"] = replay(out)
        result["seconds"] = round(perf_counter() - started, 2)
        _write_json(out / "run.json", result)
        out.rename(output.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("run")
    train.add_argument("--phase1-root", type=Path, default=PHASE1_ROOT)
    train.add_argument("--phase2-run-dir", type=Path, default=PHASE2_ROOT / "m5_seed42")
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--timesteps", type=int, default=512)
    train.add_argument("--validation-weeks", type=int, default=20)
    check = commands.add_parser("replay")
    check.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "replay":
        print(replay(args.run_dir))
    else:
        output = args.output_dir or OUTPUT_ROOT / f"m5_seed{args.seed}"
        result = run(args.phase1_root, args.phase2_run_dir, output, seed=args.seed,
                     timesteps=args.timesteps, weeks=args.validation_weeks)
        print(f"[Phase 3] Saved {output.resolve()} ({result['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
