"""Compare V2 pricing policies on validation and write the Phase 5 handoff."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DDPG, PPO, SAC
from xgboost import XGBRegressor

from .data import make_windows
from .forecast_benchmark import _tabular_features
from .phase1_finalize import PROJECT_ROOT
from .phase2_song import (
    PHASE1_ROOT, OUTPUT_ROOT as PHASE2_ROOT, _read_json, _verify, _write_json,
)
from .phase3_song_pricing import (
    MAIN_POLICY, OUTPUT_ROOT as PHASE3_ROOT, _evaluate, _forecaster, _load_cohort, validation_inputs,
)
from .phase4_forecast import (
    OUTPUT_ROOT, calendar_features, checked_m5_checkpoint, checked_protocol, only_validation,
)
from .pricing_simulator import KernelDemandSimulator, LagAdjustedKernelSimulator
from .select_simulator import CANDIDATE_WEIGHTS, RATIO_LIMIT, _scores, _scoring_rows
from .song_forecast import LOOKBACK, _sha256, feature_arrays as song_arrays

NOISE_SEEDS = (42, 123, 456)
GRIDS = ("grid_song", "grid_xgboost_price")
PPO_POLICIES = ("ppo_frozen", MAIN_POLICY, "ppo_feedback_no_gradient", "ppo_frozen_no_gradient")
COMPARISONS = ("fixed_mean", "heuristic", "ddpg", "sac", *PPO_POLICIES, *GRIDS)
GRID_POINTS = 41
CALIBRATION_WEEKS = 20


def grid_price(prices: np.ndarray, quantities: np.ndarray, reference: float) -> float:
    if prices.shape != quantities.shape or not prices.size:
        raise ValueError("Each candidate price must have one demand prediction")
    revenue = prices * quantities
    if not np.isfinite(revenue).all():
        raise ValueError("Non-finite forecast revenue on the price grid")
    ties = np.flatnonzero(np.isclose(revenue, revenue.max(), rtol=0, atol=1e-8))
    return float(prices[min(ties, key=lambda i: (abs(prices[i] - reference), prices[i]))])


class XGBDemand:
    def __init__(self, model_path: Path, metadata: dict):
        if not metadata["with_price"] or metadata["fit_scope"] != "train_only":
            raise ValueError("Grid pricing requires train-only XGBoost with a candidate-price feature")
        self.model = XGBRegressor(n_jobs=2)
        self.model.load_model(model_path)
        self.model.set_params(n_jobs=2)
        self.metadata = metadata

    def predict_prices(self, row, prices: np.ndarray) -> np.ndarray:
        frame = calendar_features(pd.DataFrame([dict(row)] * len(prices)).assign(price=prices))
        m = self.metadata
        features = _tabular_features(frame, m["skus"], m["scalers"], LOOKBACK, True,
                                     m["calendar_covariates"])
        return np.expm1(self.model.predict(features)).clip(0)

    def predict_q(self, row, price: float) -> float:
        return float(self.predict_prices(row, np.array([price]))[0])


def _song_quantities(forecaster, row, prices: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame([dict(row)] * len(prices))
    h, c = song_arrays(frame, forecaster.scalers, prices)
    with torch.no_grad():
        z = forecaster.model(torch.from_numpy(h), torch.from_numpy(c)).numpy()
    scaler = forecaster.scalers[str(row["sku"])]
    return np.maximum(0, z * scaler["qty_std"] + scaler["qty_mean"])


def _candidate_prices(simulator, sku: str, reference: float) -> np.ndarray:
    surface = simulator.surfaces[sku]
    return np.unique(np.append(np.linspace(surface.price_min, surface.price_max, GRID_POINTS),
                               np.clip(reference, surface.price_min, surface.price_max)))


def _grid_policy(predict):
    def choose(_, env):
        row = env._current_row()
        prices = _candidate_prices(env.simulator, env.sku, env.current_price)
        best = grid_price(prices, predict(row, prices), env.current_price)
        return env.action_for_price(best)
    return choose


def _simulators(weekly: pd.DataFrame, start: pd.Timestamp, out: Path):
    only_validation(weekly)
    boundary = start + pd.Timedelta(weeks=CALIBRATION_WEEKS)
    base = KernelDemandSimulator().fit(weekly)
    base_rows = _scoring_rows(base, weekly, "validation")
    candidates = {"gaussian_song": base}
    for weight in CANDIDATE_WEIGHTS:
        if weight > 0:
            candidates[f"lag_{weight:g}"] = LagAdjustedKernelSimulator(weight, ratio_limit=RATIO_LIMIT).fit(weekly)
    rows_by_name = {"last_week": base_rows, **{
        name: base_rows if name == "gaussian_song" else _scoring_rows(sim, weekly, "validation")
        for name, sim in candidates.items()
    }}
    metrics = []
    for name, rows in rows_by_name.items():
        for fold, selection in (("calibration", rows["week"].lt(boundary)),
                                ("later_validation", rows["week"].ge(boundary))):
            for score in _scores(rows.loc[selection], "validation",
                                 "naive_last" if name == "last_week" else name):
                metrics.append({**score, "model": name, "validation_block": fold})
    board = pd.DataFrame(metrics)
    board.to_csv(out / "simulator_validation.csv", index=False)
    calibration = board.loc[board["validation_block"].eq("calibration")
                            & board["segment"].eq("in_support") & board["model"].str.startswith("lag_")]
    lag_name = calibration.sort_values(["rmse", "model"]).iloc[0]["model"]
    later = board.loc[board["validation_block"].eq("later_validation")
                      & board["segment"].eq("in_support")].set_index("model")
    names = ("gaussian_song", lag_name)
    sims = {name: candidates[name] for name in names}
    for name, sim in sims.items():
        if set(sim.surfaces) != set(weekly["sku"]):
            raise ValueError(f"{name}: incomplete simulator SKU coverage")
    base.diagnostics().to_csv(out / "simulator_support.csv", index=False)
    diagnostics = {
        "first_calibration_week": start.date().isoformat(), "first_later_validation_week": boundary.date().isoformat(),
        "gaussian_seasonal_bandwidth_weeks": base.seasonal_bandwidth_weeks,
        "lag_weights": list(CANDIDATE_WEIGHTS), "lag_ratio_limit": RATIO_LIMIT,
        "selected_lag_kernel": lag_name, "selected_lag_weight": sims[lag_name].recent_weight,
        "later_in_support_rows": int(later.loc["last_week", "n"]),
        "later_last_week_rmse": float(later.loc["last_week", "rmse"]),
        "later_validation": {name: {"rmse": float(later.loc[name, "rmse"]),
                                    "mae": float(later.loc[name, "mae"]),
                                    "passes_gate": bool(later.loc[name, "rmse"] <= later.loc["last_week", "rmse"])}
                             for name in names},
    }
    return sims, boundary, diagnostics


def _read_forecast(path: Path, dataset: str, phase1_hash: str) -> dict:
    summary = _read_json(path / "forecast_run.json")
    _verify(path / "validation_keys.csv", summary["validation_keys_sha256"])
    if (summary["dataset"] != dataset or summary["split"] != "validation"
            or summary["phase1_manifest_sha256"] != phase1_hash):
        raise ValueError(f"{dataset}: Phase 4 forecast scope differs")
    return summary


def _read_policies(phase3: Path, phase1_hash: str, phase2_hash: str, skus: list[str]):
    source = _read_json(phase3 / "run.json")
    if (source["skus"] != skus or source["phase1_manifest_sha256"] != phase1_hash
            or source["phase2_checkpoint_sha256"] != phase2_hash or source["split"] != "validation"):
        raise ValueError("Phase 3 policy provenance differs from the Phase 4 pricing cohort")
    models, forecasters, hashes = {}, {}, {}
    for name, cls in (("ddpg", DDPG), ("sac", SAC)):
        path = phase3 / "models" / f"{name}.zip"
        hashes[name] = _sha256(path)
        models[name] = cls.load(path, device="cpu")
    for name in PPO_POLICIES:
        pair = source["pairs"][name]
        model_path, forecast_path = phase3 / pair["policy_file"], phase3 / pair["forecaster_file"]
        _verify(model_path, pair["policy_sha256"])
        _verify(forecast_path, pair["forecaster_sha256"])
        saved = torch.load(forecast_path, map_location="cpu", weights_only=True)
        if (saved["matched_ppo_sha256"] != pair["policy_sha256"] or saved["policy"] != name
                or saved["phase2_checkpoint_sha256"] != phase2_hash or saved["seed"] != source["seed"]):
            raise ValueError(f"Invalid checkpoint pair for {name}")
        models[name] = PPO.load(model_path, device="cpu")
        forecasters[name] = _forecaster(saved, skus, source["seed"])
        hashes[name] = {"policy_sha256": pair["policy_sha256"], "forecaster_sha256": pair["forecaster_sha256"]}
    return models, forecasters, hashes, source


def _responses(inputs: pd.DataFrame, simulators, song, xgb, out: Path) -> None:
    rows, simulator_rows = [], []
    for _, row in inputs.groupby("sku", sort=True).head(1).iterrows():
        reference = float(row["price_lag_1"])
        sku, week = row["sku"], row["week"]
        prices = _candidate_prices(simulators["gaussian_song"], sku, reference)
        for name, pred in (("song_transformer", _song_quantities(song, row, prices)),
                           ("xgboost_price", xgb.predict_prices(row, prices))):
            rows.extend({"model": name, "sku": sku, "week": week,
                         "candidate_price": price, "predicted_qty": qty,
                         "forecast_revenue": price * qty} for price, qty in zip(prices, pred))
        context = {field: row[field] for field in ("event_days", "snap_days", "qty_lag_1", "price_lag_1")}
        for name, simulator in simulators.items():
            simulator_rows.extend({
                "simulator": name, "sku": sku, "week": week, "candidate_price": price,
                "expected_units": simulator.predict_mean(sku, float(price), week, context),
            } for price in prices)
    pd.DataFrame(rows).to_csv(out / "demand_price_response.csv", index=False)
    pd.DataFrame(simulator_rows).to_csv(out / "simulator_price_response.csv", index=False)


def _policy_results(sims, inputs, song, xgb, models, forecasters, out: Path):
    rows, summaries = [], []
    grid = {"grid_song": _grid_policy(lambda row, prices: _song_quantities(song, row, prices)),
            "grid_xgboost_price": _grid_policy(xgb.predict_prices)}
    for sim_name, sim in sims.items():
        for seed in NOISE_SEEDS:
            for name in COMPARISONS:
                diagnostic = xgb if name == "grid_xgboost_price" else song
                frame, summary = _evaluate(name, models.get(name), inputs, sim, diagnostic,
                                           forecasters.get(name), seed, action_policy=grid.get(name))
                edge = (np.isclose(frame["chosen_price"], frame["train_price_min"], rtol=0, atol=1e-6)
                        | np.isclose(frame["chosen_price"], frame["train_price_max"], rtol=0, atol=1e-6))
                frame["at_price_boundary"] = edge
                rows.append(frame.assign(simulator=sim_name))
                summaries.append({**summary, "simulator": sim_name, "price_boundary_pct": float(edge.mean() * 100)})
            print(f"[Phase 4] {sim_name}: {len(COMPARISONS)} policies at simulation noise seed {seed}", flush=True)
    decisions = pd.concat(rows, ignore_index=True)
    decisions.to_csv(out / "pricing_validation_decisions.csv", index=False)
    by_sku = decisions.groupby(["simulator", "policy", "seed", "sku"], as_index=False).agg(
        decisions=("chosen_price", "size"), simulated_revenue=("simulated_revenue", "sum"),
        mean_price=("chosen_price", "mean"), expected_regret=("expected_regret", "sum"),
    )
    by_sku.to_csv(out / "pricing_validation_by_sku.csv", index=False)
    runs = pd.DataFrame(summaries)
    indices = ["simulator", "seed"]
    for comparator, field in (("fixed_mean", "uplift_vs_fixed_pct"), ("ppo_frozen", "uplift_vs_ppo_frozen_pct")):
        baseline = runs.loc[runs["policy"].eq(comparator), indices + ["simulated_revenue"]].rename(
            columns={"simulated_revenue": "baseline_revenue"})
        runs = runs.merge(baseline, on=indices, validate="many_to_one")
        positive = runs["baseline_revenue"].gt(0)
        runs[field] = np.where(positive, 100 * (runs["simulated_revenue"] /
                                                runs["baseline_revenue"].where(positive) - 1), np.nan)
        runs = runs.drop(columns="baseline_revenue")
    runs.to_csv(out / "pricing_validation_runs.csv", index=False)
    board = runs.groupby(["simulator", "policy"], as_index=False).agg(
        seeds=("seed", "size"), mean_revenue=("simulated_revenue", "mean"),
        revenue_std=("simulated_revenue", "std"), revenue_min=("simulated_revenue", "min"),
        revenue_max=("simulated_revenue", "max"), uplift_vs_fixed_pct=("uplift_vs_fixed_pct", "mean"),
        uplift_vs_ppo_frozen_pct=("uplift_vs_ppo_frozen_pct", "mean"),
        mean_regret=("expected_regret", "mean"), price_boundary_pct=("price_boundary_pct", "mean"),
        seeds_above_fixed=("uplift_vs_fixed_pct", lambda values: int(values.gt(0).sum())),
    ).sort_values(["simulator", "mean_revenue", "policy"], ascending=[True, False, True])
    board.to_csv(out / "pricing_validation.csv", index=False)
    return board, decisions


def _pricing_choice(board: pd.DataFrame, diagnostics: dict, decisions: pd.DataFrame) -> dict:
    winners = board.groupby("simulator", sort=False).head(1).set_index("simulator")
    winner_names = winners["policy"].to_dict()
    eligible = [name for name, row in diagnostics["later_validation"].items() if row["passes_gate"]]
    same_top = len(set(winner_names.values())) == 1
    candidate = None
    if (eligible and same_top and winners.loc[eligible, "uplift_vs_fixed_pct"].gt(0).all()
            and winners.loc[eligible, "seeds_above_fixed"].ge(2).all()):
        candidate = next(iter(winner_names.values()))
    upper_bound_hits = {}
    for simulator, winner in winner_names.items():
        rows = decisions.loc[decisions["simulator"].eq(simulator) & decisions["policy"].eq(winner)]
        upper_bound_hits[simulator] = {
            "exact_upper_bound": int(rows["chosen_price"].eq(rows["train_price_max"]).sum()),
            "decisions": len(rows),
        }
    repeats_fixed_max = bool(candidate and all(row["exact_upper_bound"] == row["decisions"]
                                               for row in upper_bound_hits.values()))
    return {"top_by_mean_revenue_per_simulator": winner_names, "simulators_passing_gate": eligible,
            "validation_candidate": candidate,
            "winning_policy_upper_bound_actions": upper_bound_hits,
            "candidate_repeats_fixed_upper_bound_on_validation": repeats_fixed_max,
            "additional_phase5_control": "fixed_max_observed_train_price" if repeats_fixed_max else None,
            "decision_rule": "At least one simulator passes the observed-price check; same top policy "
                             "across simulators, positive mean uplift and at least two wins against fixed "
                             "among three noise seeds in each simulator passing the check.",
            "fallback_if_no_candidate": "Keep fixed price as reference and the shortlist for simulation only"}


def _artifact_ref(path: Path) -> dict:
    resolved = path.resolve()
    display = resolved.relative_to(PROJECT_ROOT) if resolved.is_relative_to(PROJECT_ROOT) else resolved
    return {"path": display.as_posix(), "sha256": _sha256(path)}


def run(root: Path, phase2: Path, phase3: Path, forecast_uci: Path, forecast_m5: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    started = perf_counter()
    torch.set_num_threads(1)
    phase1, protocol = checked_protocol(root)
    phase1_hash = _sha256(root / "phase1_manifest.json")
    saved, phase2_run = checked_m5_checkpoint(root, phase2)
    forecast_runs = {dataset: _read_forecast(path, dataset, phase1_hash)
                     for dataset, path in (("uci", forecast_uci), ("m5", forecast_m5))}
    fm5 = forecast_runs["m5"]
    if fm5["settings"]["song_transformer"]["reused_phase2_checkpoint_sha256"] != phase2_run["checkpoint_sha256"]:
        raise ValueError("Phase 4 forecasts and pricing refer to different M5 checkpoints")
    xgb_model = forecast_m5 / "models" / "xgboost_price.json"
    xgb_meta = forecast_m5 / "models" / "xgboost_price_meta.json"
    _verify(xgb_model, fm5["xgboost_price_model_sha256"])
    _verify(xgb_meta, fm5["xgboost_price_metadata_sha256"])
    xgb = XGBDemand(xgb_model, _read_json(xgb_meta))
    weekly, skus = _load_cohort(root, phase1, protocol)
    song = _forecaster(saved, skus, NOISE_SEEDS[0])
    models, forecasters, model_hashes, source = _read_policies(
        phase3, phase1_hash, phase2_run["checkpoint_sha256"], skus)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".phase4-pricing-", dir=output.parent) as temp:
        out = Path(temp) / "run"
        out.mkdir()
        start = pd.Timestamp(protocol["datasets"]["m5"]["validation_start"])
        sims, rollout_start, diagnostics = _simulators(weekly, start, out)
        print(f"[Phase 4] Simulator selection and later validation: {diagnostics}", flush=True)
        weeks = protocol["pricing"]["test_weeks_per_series"]
        inputs = validation_inputs(make_windows(weekly, LOOKBACK), skus, rollout_start, weeks)
        inputs.to_csv(out / "pricing_validation_inputs.csv", index=False)
        _responses(inputs, sims, song, xgb, out)
        board, decisions = _policy_results(sims, inputs, song, xgb, models, forecasters, out)
        choice = _pricing_choice(board, diagnostics, decisions)
        sources = {name: _artifact_ref(path) for name, path in (
            ("phase1", root / "phase1_manifest.json"), ("phase2_m5", phase2 / "run.json"),
            ("phase3_m5", phase3 / "run.json"), ("phase4_forecast_uci", forecast_uci / "forecast_run.json"),
            ("phase4_forecast_m5", forecast_m5 / "forecast_run.json"),
        )}
        handoff = {
            "protocol_id": protocol["protocol_id"], "protocol_sha256": phase1["protocol_sha256"],
            "selection_evidence": sources,
            "selection_split": "validation", "fit_scope_during_selection": "train_only",
            "test_forecast_fit_scope": protocol["forecast"]["test_forecast_checkpoint_fit_scope"],
            "test_pricing_forecaster_fit_scope": protocol["forecast"]["pricing_forecaster_checkpoint_fit_scope"],
            "full_scope_forecast_feature_rules": {
                "sequence_sku_encoding": "one-hot through 256 SKUs, learned embedding above 256",
                "sequence_embedding_size": "min(32, max(8, round(sqrt(n_skus))))",
                "xgboost_sku_encoding": "dense one-hot through 256 SKUs, sparse one-hot above 256",
                "model_config_dimensions": "Recompute from the full-scope inputs; preserve chosen training settings",
            },
            "forecast": {dataset: {"preferred": summary["best_by_validation_rmse"],
                                   "selection_metric": "micro RMSE on identical validation keys",
                                   "settings_by_model": summary["settings"],
                                   "screening_rows": summary["validation_rows"],
                                   "screening_skus": len(summary["skus"]),
                                   "scored_screening_skus": summary["validation_window_skus"],
                                   "screening_validation_keys_sha256": summary["validation_keys_sha256"],
                                   "deferred_models": summary["deferred_models"]}
                         for dataset, summary in forecast_runs.items()},
            "frozen_test_keys": {
                dataset: {"count": protocol["datasets"][ds]["primary_test_key_count"],
                          "sha256": protocol["datasets"][ds]["primary_test_key_sha256"]}
                for dataset, ds in (("uci", "uci_online_retail"), ("m5", "m5"))},
            "phase5_seeds": protocol["forecast"]["stochastic_seeds"],
            "pricing": {"cohort_skus": skus, "test_decisions_per_seed": protocol["pricing"]["evaluation_cycles"],
                        "primary_song_simulator": "gaussian_song", "sensitivity_simulator": diagnostics["selected_lag_kernel"],
                        "gaussian_seasonal_bandwidth_weeks": diagnostics["gaussian_seasonal_bandwidth_weeks"],
                        "sensitivity_weight": diagnostics["selected_lag_weight"], "ratio_limit": RATIO_LIMIT,
                        "grid_points": GRID_POINTS, "training_timesteps_per_policy": source["training_timesteps_per_learned_policy"],
                        "grid_rule": "Uniform grid in observed train support plus current price; maximize "
                                     "price times predicted demand, then closest to current price, then lower price.",
                        "ppo_hyperparameters": source["ppo_hyperparameters"],
                        "feedback_config": source["feedback"],
                        "feedback_torch_seed_offset": source["feedback_torch_seed_offset"],
                        "pricing_demand_training_skus": fm5["skus"],
                        "phase5_methods": list(COMPARISONS),
                        **choice},
        }
        _write_json(out / "phase5_settings.json", handoff)
        result = {"dataset": "m5", "split": "validation", "phase1_manifest_sha256": phase1_hash,
                  "sources": sources,
                  "phase3_run_sha256": _sha256(phase3 / "run.json"),
                  "phase2_checkpoint_sha256": phase2_run["checkpoint_sha256"],
                  "xgboost_price_sha256": fm5["xgboost_price_model_sha256"],
                  "pricing_inputs_sha256": _sha256(out / "pricing_validation_inputs.csv"),
                  "simulator_validation": diagnostics, "policies": list(COMPARISONS),
                  "policy_checkpoint_hashes": model_hashes, "policy_training_seed": source["seed"],
                  "validation_noise_seeds": list(NOISE_SEEDS),
                  "validation_first_week": rollout_start.date().isoformat(),
                  "validation_weeks_per_sku": weeks, "decisions_per_policy_per_noise_seed": len(inputs),
                  "simulation_scope_note": "Phase 3 policies trained in Gaussian are held fixed across simulators. "
                                           "Validation noise seeds are not independent training seeds.",
                  "policy_selection": choice, "seconds": round(perf_counter() - started, 2)}
        _write_json(out / "pricing_run.json", result)
        out.rename(output.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--phase2-dir", type=Path, default=PHASE2_ROOT / "m5_seed42")
    parser.add_argument("--phase3-dir", type=Path, default=PHASE3_ROOT / "m5_seed42")
    parser.add_argument("--forecast-uci", type=Path, default=OUTPUT_ROOT / "forecast_uci")
    parser.add_argument("--forecast-m5", type=Path, default=OUTPUT_ROOT / "forecast_m5")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "pricing")
    args = parser.parse_args()
    result = run(args.phase1_root, args.phase2_dir, args.phase3_dir,
                 args.forecast_uci, args.forecast_m5, args.output_dir)
    print(f"[Phase 4] Saved pricing validation in {result['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
