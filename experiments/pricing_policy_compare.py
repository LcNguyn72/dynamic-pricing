"""Step 4: compare tuned Transformer and ARIMAX pricing in the frozen M5 simulators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import stable_baselines3
import torch
from stable_baselines3 import PPO

from .phase1_finalize import _observed
from .phase5_data import PHASE1_ROOT, iter_weekly, load_contract
from .phase5_pricing import (
    FIXED_MAX, CachedGaussian, CachedLag, FastFeedbackForecaster, _atomic_json,
    _contract, _data, _evaluate_test, _unit, test_inputs,
)
from .phase5_verify import check_pricing_noise
from .pricing_arimax import _write_csv
from .song_forecast import CURRENT_FEATURES, HISTORY_FEATURES, LOOKBACK, SongTransformer, _sha256
from .song_pricing_pilot import _ppo


ARTIFACTS = PHASE1_ROOT.parent
OUTPUT = ARTIFACTS / "pricing_policy_compare_v2"
STEP1 = ARTIFACTS / "pricing_baseline_v2" / "pricing"
STEP2 = ARTIFACTS / "pricing_transformer_tune_v2"
STEP3 = ARTIFACTS / "pricing_arimax_v2"
CACHE = ARTIFACTS / "phase5_v2" / "data_cache"
KEYS = ["sku", "week"]
POLICY_KEYS = ["simulator", "policy", "seed"]
OLD = (FIXED_MAX, "fixed_mean", "grid_song", "ppo_feedback_gradient", "ppo_frozen", "ddpg")
NEW_PPO = {"ppo_tuned_feedback": "ppo_feedback_gradient", "ppo_tuned_frozen": "ppo_frozen"}
NEW = ("grid_song_tuned", "grid_arimax", *NEW_PPO)
EXPANSION_PER_STORE = 5


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verified(path, expected):
    if _sha256(path) != expected:
        raise ValueError(f"Unexpected artifact hash: {path}")
    return path


class TunedForecaster(FastFeedbackForecaster):
    """Keep the Phase 5 policy state and encode the checkpoint's own history length."""

    def _arrays(self, row, price):
        history, current = super()._arrays(row, price)
        return history[:, -self.model.lookback:], current


class ArimaxDemand:
    """ARIMAX(1,0,0) conditional on last week's quantity and price."""

    def __init__(self, scalers):
        self.scalers = scalers

    def predict_prices(self, row, prices):
        state = self.scalers[str(row["sku"])]
        prices = np.asarray(prices, dtype=float)
        last_qty, last_price = float(row["qty_lag_1"]), float(row["price_lag_1"])
        if (not np.isfinite(prices).all() or not np.isfinite([last_qty, last_price]).all()
                or last_qty < 0 or last_price <= 0
                or np.any(prices < state["price_min"] - 1e-6) or np.any(prices > state["price_max"] + 1e-6)):
            raise ValueError("ARIMAX needs finite train-supported candidate prices and observed prior history")
        intercept, beta, phi, _ = state["params"]
        residual = ((last_qty - state["qty_mean"]) / state["qty_sd"]
                    - beta * (last_price - state["price_mean"]) / state["price_sd"])
        z = intercept + phi * residual + beta * (prices - state["price_mean"]) / state["price_sd"]
        return np.maximum(0.0, state["qty_mean"] + z * state["qty_sd"])

    def predict_q(self, row, price):
        return float(self.predict_prices(row, [price])[0])


def forecaster(saved, skus, seed):
    if (saved["history_features"] != HISTORY_FEATURES or saved["current_features"] != CURRENT_FEATURES
            or not 1 <= saved["lookback"] <= LOOKBACK or not set(skus).issubset(saved["scalers"])):
        raise ValueError("Unexpected Transformer input layout or SKU scalers")
    model = SongTransformer(lookback=saved["lookback"])
    model.load_state_dict(saved["state_dict"])
    return TunedForecaster(model, {sku: saved["scalers"][sku] for sku in skus}, seed=seed,
                           feedback_rng_state=saved.get("feedback_rng_state"))


def volume_band_skus(volumes, count):
    ranked = volumes.sort_index().sort_values(ascending=False, kind="stable")
    if len(ranked) < count:
        raise ValueError("Insufficient eligible series for volume-band sampling")
    return [str(group[len(group) // 2]) for group in np.array_split(ranked.index.to_numpy(), count)]


def expanded_cohort(root, cache, contract, params, exclude):
    start = pd.Timestamp(contract["protocol"]["datasets"]["m5"]["test_start"])
    weeks = contract["protocol"]["pricing"]["test_weeks_per_series"]
    end = start + pd.Timedelta(weeks=weeks - 1)
    first_history = start - pd.Timedelta(weeks=LOOKBACK)
    inputs, train, info = [], [], []
    for store, weekly, _ in iter_weekly(root, "m5", contract, cache_root=cache):
        observed = _observed(weekly)
        original = weekly.loc[weekly["split"].eq("train") & observed & weekly["sku"].isin(params)]
        volume = original.groupby("sku", sort=False)["qty"].sum()
        levels = original.groupby("sku", sort=False)["price"].nunique()
        covered = weekly.loc[weekly["week"].between(first_history, end) & observed]
        contiguous = covered.groupby("sku", sort=False).size().eq(LOOKBACK + weeks)
        eligible = levels.index[levels.ge(3)].intersection(contiguous.index[contiguous])
        eligible = eligible.difference(list(exclude))
        picks = volume_band_skus(volume.loc[eligible], EXPANSION_PER_STORE)
        source = weekly.loc[weekly["sku"].isin(picks)]
        inputs.append(test_inputs(source, picks, start, weeks))
        train.append(original.loc[original["sku"].isin(picks)])
        info.extend({"store": store, "sku": sku, "train_volume_band": band,
                     "train_units": float(volume[sku]), "train_price_levels": int(levels[sku])}
                    for band, sku in enumerate(picks, start=1))
    data = pd.concat(inputs, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    return pd.concat(train, ignore_index=True), data, pd.DataFrame(info)


def samplers(train, config):
    band = config["gaussian_seasonal_bandwidth_weeks"]
    return {
        config["primary_song_simulator"]: CachedGaussian(band).fit(train),
        config["sensitivity_simulator"]: CachedLag(config["sensitivity_weight"], band, config["ratio_limit"]).fit(train),
    }


def train_pair(output, saved, inputs, simulator, name, seed, budget, identity):
    skus = list(simulator.surfaces)
    feedback = name == "ppo_tuned_feedback"

    def build(stage):
        demand = forecaster(saved, skus, seed)
        policy, stats = _ppo(inputs, simulator, demand, seed, budget, True, feedback, stage / "policy")
        updates = int(policy.get_env().get_attr("feedback_updates")[0])
        if policy.num_timesteps != budget or updates != (budget // 10 if feedback else 0):
            raise ValueError(f"{name}: unexpected PPO training budget or feedback count")
        weights = demand.model.state_dict()
        change = max(float((value - saved["state_dict"][key]).abs().max()) for key, value in weights.items())
        if not np.isfinite(change) or (change > 0) != feedback:
            raise ValueError(f"{name}: unexpected change in pretrained weights")
        torch.save({"state_dict": weights, "lookback": saved["lookback"],
                    "history_features": HISTORY_FEATURES, "current_features": CURRENT_FEATURES,
                    "feedback_rng_state": demand.feedback_rng_state,
                    "policy_sha256": _sha256(stage / "policy.zip"),
                    "source_transformer_sha256": identity["transformer_sha256"],
                    "seed": seed, "policy": name}, stage / "transformer.pt")
        return {**stats, "training_timesteps": budget, "training_forecast_updates": updates,
                "transformer_max_weight_change": change}

    _unit(output, {**identity, "seed": seed, "policy": name, "budget": budget}, build)
    return output


def load_pair(path, saved, skus, name, seed):
    updated = torch.load(path / "transformer.pt", map_location="cpu", weights_only=True)
    policy_file = verified(path / "policy.zip", updated["policy_sha256"])
    if updated["seed"] != seed or updated["policy"] != name or updated["lookback"] != saved["lookback"]:
        raise ValueError("PPO policy and Transformer checkpoint are not a matched pair")
    return PPO.load(policy_file, device="cpu"), forecaster({**saved, **updated}, skus, seed)


def simulate(output, scope, inputs, saved, arimax, available, pairs, name, seed, config, identity):
    def build(stage):
        skus = inputs["sku"].unique().tolist()
        blocks = []
        for sim, sampler in available.items():
            if set(sampler.surfaces) != set(skus):
                raise ValueError(f"Unexpected extra or missing simulator series in {scope}")
            model, feedback = None, None
            if name in NEW_PPO:
                model, feedback = load_pair(pairs[name, seed], saved, skus, name, seed)
                demand = feedback
            else:
                demand = arimax if name == "grid_arimax" else forecaster(saved, skus, seed)
            rows, _ = _evaluate_test(NEW_PPO.get(name, name), seed, model, sampler, inputs,
                                     demand, feedback, grid_points=config["grid_points"])
            blocks.append(rows.assign(policy=name, simulator=sim))
        decisions = pd.concat(blocks, ignore_index=True)
        decisions.to_csv(stage / "decisions.csv", index=False)
        return {"decision_rows": len(decisions)}

    _unit(output, {**identity, "scope": scope, "policy": name, "seed": seed}, build)
    return pd.read_csv(output / "decisions.csv", dtype={"sku": str, "week": str})


def audit_decisions(frame, inputs, methods, seeds, sims):
    key_table = inputs[KEYS].assign(week=inputs["week"].dt.strftime("%Y-%m-%d"))
    expected_keys = set(key_table.itertuples(index=False, name=None))
    expected_groups = {(sim, method, seed) for sim in sims for method in methods for seed in seeds}
    groups = frame.groupby(POLICY_KEYS)
    if set(groups.groups) != expected_groups:
        raise ValueError("Missing or extra groups in pricing audit")
    for group, decisions in groups:
        if len(decisions) != len(expected_keys) or set(decisions[KEYS].itertuples(index=False, name=None)) != expected_keys:
            raise ValueError(f"Unexpected pricing test keys: {group}")
        days = decisions.sort_values(KEYS).groupby("sku").cumcount() + 1
        expected = days // 10 if group[1] in ("ppo_tuned_feedback", "ppo_feedback_gradient") else days * 0
        if not decisions.sort_values(KEYS)["forecast_update_count"].eq(expected).all():
            raise ValueError(f"Incorrect online feedback schedule: {group}")
    price = frame["chosen_price"]
    # Phase 5 already recorded the meaning of these train-support boundaries.
    if (price.lt(frame["train_price_min"] - 1e-9) | price.gt(frame["train_price_max"] + 1e-9)).any():
        raise ValueError("A price decision violates train support")
    if not np.allclose(price * frame["simulated_units"], frame["simulated_revenue"], rtol=1e-10, atol=1e-8):
        raise ValueError("Revenue must equal simulated units times price")
    if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Partially invalid pricing metrics")
    check_pricing_noise(frame)


def summarize(frame, summary_path):
    grouped = frame.groupby(POLICY_KEYS)
    runs = grouped.agg(decisions=("chosen_price", "size"), revenue=("simulated_revenue", "sum"),
                       expected_revenue=("expected_revenue", "sum"), regret=("expected_regret", "sum"),
                       upper=("at_upper_bound", "mean"), lower=("at_lower_bound", "mean"))
    spans = frame.groupby([*POLICY_KEYS, "sku"])["chosen_price"].agg(lambda p: p.max() - p.min())
    runs["share_skus_with_weekly_price_change"] = spans.gt(0.01).groupby(level=POLICY_KEYS).mean()
    runs["boundary_pct"] = (runs["upper"] + runs["lower"]) * 100
    cap = runs["revenue"].xs(FIXED_MAX, level="policy").rename("fixed_cap")
    runs = runs.reset_index().merge(cap, on=["simulator", "seed"], validate="many_to_one")
    runs["delta_vs_fixed_cap"] = runs["revenue"] - runs["fixed_cap"]
    runs["uplift_vs_fixed_cap_pct"] = 100 * runs["delta_vs_fixed_cap"] / runs["fixed_cap"].where(runs["fixed_cap"].gt(0))
    scores = runs.groupby(["simulator", "policy"], as_index=False).agg(
        n_seeds=("seed", "size"), mean_revenue=("revenue", "mean"), std_revenue=("revenue", "std"),
        min_revenue=("revenue", "min"), max_revenue=("revenue", "max"), mean_regret=("regret", "mean"),
        mean_paired_uplift_vs_fixed_cap_pct=("uplift_vs_fixed_cap_pct", "mean"),
        seed_wins_vs_fixed_cap=("delta_vs_fixed_cap", lambda delta: int(delta.gt(0.01).sum())),
        upper_pct=("upper", lambda f: 100 * f.mean()), lower_pct=("lower", lambda f: 100 * f.mean()),
        boundary_pct=("boundary_pct", "mean"),
        sku_seed_weekly_price_change_pct=("share_skus_with_weekly_price_change", lambda f: 100 * f.mean()),
    ).sort_values(["simulator", "mean_revenue"], ascending=[True, False])
    _write_csv(summary_path / "runs.csv", runs)
    _write_csv(summary_path / "summary.csv", scores)
    paired = []
    choices = [
        ("grid_song_tuned", "grid_arimax"), ("grid_song_tuned", "grid_song"),
        ("ppo_tuned_feedback", "ppo_feedback_gradient"), ("ppo_tuned_feedback", "ppo_tuned_frozen"),
        ("ppo_tuned_feedback", "grid_song_tuned"), ("ppo_tuned_feedback", "grid_arimax"),
    ]
    for sim, part in runs.groupby("simulator"):
        wide = part.pivot(index="seed", columns="policy", values="revenue")
        for candidate, reference in choices:
            if {candidate, reference}.issubset(wide.columns):
                delta = wide[candidate] - wide[reference]
                paired.append({"simulator": sim, "candidate": candidate, "reference": reference,
                               "mean_revenue_delta": float(delta.mean()), "seed_wins": int(delta.gt(.01).sum()),
                               "mean_paired_pct": float((100 * delta / wide[reference].where(wide[reference].gt(0))).mean())})
    _write_csv(summary_path / "paired_comparisons.csv", pd.DataFrame(paired))
    return scores


def evaluate_scope(output, scope, inputs, saved, states, available, pairs, config, identity, seeds, baseline=None):
    directory = output / scope
    directory.mkdir(exist_ok=True)
    _write_csv(directory / "test_keys.csv", inputs[KEYS])
    methods = NEW if baseline is not None else (*NEW, "fixed_mean", FIXED_MAX)
    frames = [] if baseline is None else [baseline]
    arimax = ArimaxDemand(states)
    for seed in seeds:
        for name in methods:
            frames.append(simulate(directory / "episodes" / f"seed_{seed}" / name, scope, inputs,
                                   saved, arimax, available, pairs, name, seed, config, identity))
            print(f"[Step 4 {scope}] {seed} {name}", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    all_methods = [*OLD, *methods] if baseline is not None else list(methods)
    audit_decisions(frame, inputs, all_methods, seeds, list(available))
    _write_csv(directory / "decisions.csv", frame)
    summarize(frame, directory)
    report = {"status": "complete", "scope": scope, "n_decisions": len(frame),
              "series": int(inputs["sku"].nunique()), "n_weeks": int(inputs["week"].nunique()),
              "n_unique_keys": len(inputs), "methods": all_methods, "seeds": seeds,
              "simulators": list(available), "source": identity,
              "files": {file.name: _sha256(file) for file in directory.glob("*.csv")}}
    _atomic_json(directory / "audit.json", report)
    print(f"[Step 4 {scope}] audited {len(frame)} decisions", flush=True)
    return report


def run(root, output, cache, step1, step2, step3, lock_path, scope):
    started = perf_counter()
    torch.set_num_threads(1)
    provenance = load_contract(root, lock_path)
    phase1, protocol, lock = _contract(root, lock_path)
    config = lock["pricing"]
    receipts = [read_json(step1 / "pricing_run.json"), read_json(step2 / "m5_test_audit.json"),
                read_json(step3 / "run.json")]
    baseline_metadata, tuned_metadata, arimax_metadata = receipts
    if (any(receipt["status"] != "complete" for receipt in receipts)
            or baseline_metadata["lock_sha256"] != provenance["lock_sha256"]
            or tuned_metadata["test_keys_sha256"] != lock["frozen_test_keys"]["m5"]["sha256"]
            or tuned_metadata["fit_scope"] != "train_only"
            or arimax_metadata["selected"] != "arimax_price"
            or arimax_metadata["identity"]["fit_scope"] != "train_only"
            or arimax_metadata["identity"]["transformer_checkpoint_sha256"] != tuned_metadata["checkpoint_sha256"]):
        raise ValueError("Steps 1–3 artifacts do not match the frozen pricing run")
    checkpoint = verified(step2 / "full_m5" / "song_transformer.pt", tuned_metadata["checkpoint_sha256"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved["version"]["phase1_sha256"] != provenance["phase1_sha256"]
            or saved["fit_scope"] != "train_only" or saved["lookback"] != tuned_metadata["settings"]["lookback"]):
        raise ValueError("The tuned Transformer does not match the train-only M5 dataset")
    states = {}
    for store, meta in arimax_metadata["models_by_store"].items():
        models = pd.read_csv(verified(step3 / meta["models"], meta["models_sha256"]), keep_default_na=False)
        chosen = models.loc[models["model"].eq(arimax_metadata["selected"]) & models["status"].eq("ok")]
        states.update({entry.sku: json.loads(entry.parameters) for entry in chosen.itertuples(index=False)})
    if not set(config["cohort_skus"]).issubset(states):
        raise ValueError("ARIMAX misses a locked pricing SKU")
    identity = {"phase1_sha256": provenance["phase1_sha256"], "lock_sha256": provenance["lock_sha256"],
                "scope_rules_sha256": _sha256(Path(__file__)), "transformer_sha256": _sha256(checkpoint),
                "step1_run_sha256": _sha256(step1 / "pricing_run.json"),
                "step3_run_sha256": _sha256(step3 / "run.json"), "torch": str(torch.__version__),
                "stable_baselines3": stable_baselines3.__version__}
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.is_file() and read_json(identity_path) != identity:
        raise ValueError("Use a separate output directory for new settings or a different runtime")
    _atomic_json(identity_path, identity)
    train, inputs, _, _, history = _data(root, phase1, protocol, lock)
    sims = samplers(train, config)
    policy_seed = lock["phase5_seeds"]
    pairs = {}
    for seed in policy_seed:
        for name in NEW_PPO:
            pairs[name, seed] = train_pair(output / "checkpoints" / f"seed_{seed}" / name, saved, history,
                                           sims["gaussian_song"], name, seed,
                                           config["training_timesteps_per_policy"], identity)
            print(f"[Step 4 train] {seed} {name}: saved or verified pair", flush=True)

    core = None
    expanded = None
    if scope in ("core", "all"):
        decisions = verified(step1 / "pricing_test_decisions.csv", baseline_metadata["decisions_sha256"])
        baseline = pd.read_csv(decisions, dtype={"sku": str, "week": str})
        baseline = baseline.loc[baseline["policy"].isin(OLD)]
        core = evaluate_scope(output, "core", inputs, saved, states, sims, pairs, config, identity,
                              policy_seed, baseline)
    if scope in ("expanded", "all"):
        more_train, more_inputs, cohort = expanded_cohort(root, cache, provenance, states, config["cohort_skus"])
        _write_csv(output / "expanded_cohort.csv", cohort)
        expanded = evaluate_scope(output, "expanded", more_inputs, saved, states, samplers(more_train, config),
                                  pairs, config, identity, policy_seed)
    report = {"status": "complete", "scope": scope, "trained_policy_instances": len(pairs),
              "ppo_train_timesteps_per_seed": config["training_timesteps_per_policy"],
              "transformer_pretraining_seed": saved["seed"], "transformer_lookback": saved["lookback"],
              "ppo_observation_lookback": LOOKBACK, "seconds": perf_counter() - started,
              "core": core, "expanded": expanded}
    _atomic_json(output / "run.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("phase5_settings_v2.json"))
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--step1-dir", type=Path, default=STEP1)
    parser.add_argument("--step2-dir", type=Path, default=STEP2)
    parser.add_argument("--step3-dir", type=Path, default=STEP3)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--scope", choices=("core", "expanded", "all"), default="all")
    args = parser.parse_args()
    run(args.phase1_root, args.output_dir, args.cache_dir, args.step1_dir, args.step2_dir,
        args.step3_dir, args.lock, args.scope)


if __name__ == "__main__":
    main()
