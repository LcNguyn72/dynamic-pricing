"""Compare the same pricing policies across two explicitly different simulators.

Revenue and uplift here are simulated. The cross-simulator comparison is a
sensitivity check, not evidence of a real-world pricing effect.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_run(path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((path / "pricing_manifest.json").read_text(encoding="utf-8"))
    runs = pd.read_csv(path / "policy_runs.csv")
    episodes = pd.read_csv(path / "policy_episodes.csv", dtype={"sku": str})
    bounds = pd.read_csv(path / "simulator_diagnostics.csv", dtype={"sku": str})[
        ["sku", "price_min", "price_max"]]
    return manifest, runs, episodes, bounds


def _check_run(manifest: dict, runs: pd.DataFrame, episodes: pd.DataFrame) -> None:
    seeds = {int(seed) for seed in manifest["seeds"]}
    policies = set(manifest["policies"])
    expected = {(seed, policy) for seed in seeds for policy in policies}
    actual = set(zip(runs["seed"].astype(int), runs["policy"].astype(str)))
    if actual != expected or len(runs) != len(expected):
        raise ValueError("Policy runs do not cover the declared seed × policy grid exactly")
    if not np.isfinite(runs["simulated_revenue"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite simulated revenue")
    if "fixed_mean" not in policies:
        raise ValueError("A fixed_mean baseline is required")
    cycles = int(manifest["evaluation_cycles"])
    if not (runs["n_sku_weeks"].astype(int) == cycles).all():
        raise ValueError("A policy did not complete all evaluation cycles")
    counts = episodes.groupby(["seed", "policy"]).size()
    if set(counts.index) != expected or not (counts == cycles).all():
        raise ValueError("Episode log does not contain the declared evaluation cycles")
    if episodes.duplicated(["seed", "policy", "sku", "week"]).any():
        raise ValueError("Duplicate SKU-week evaluation row")
    reference = None
    for (_, _), frame in episodes.groupby(["seed", "policy"]):
        keys = set(zip(frame["sku"].astype(str), frame["week"].astype(str)))
        if reference is None:
            reference = keys
        elif keys != reference:
            raise ValueError("Policies or seeds used different SKU-week evaluation rows")


def compare(lag_dir: str | Path, song_dir: str | Path, output_dir: str | Path) -> pd.DataFrame:
    lag_manifest, lag_runs, lag_episodes, lag_bounds = _read_run(Path(lag_dir))
    song_manifest, song_runs, song_episodes, song_bounds = _read_run(Path(song_dir))
    if lag_manifest["simulator_variant"] != "lag_adjusted_kernel":
        raise ValueError("lag_dir must contain the validation-selected simulator")
    if song_manifest["simulator_variant"] != "song_kernel":
        raise ValueError("song_dir must contain the original Song-style kernel")
    matching_fields = (
        "source_manifest_sha256", "forecast_checkpoint_sha256", "seeds", "policies",
        "training_timesteps_requested", "evaluation_cycles", "evaluation_skus",
        "evaluation_selection", "training_episode_weeks",
    )
    for field in matching_fields:
        if lag_manifest[field] != song_manifest[field]:
            raise ValueError(f"Pricing experiments differ in {field}")
    _check_run(lag_manifest, lag_runs, lag_episodes)
    _check_run(song_manifest, song_runs, song_episodes)
    lag_keys = set(zip(lag_episodes["sku"].astype(str), lag_episodes["week"].astype(str)))
    song_keys = set(zip(song_episodes["sku"].astype(str), song_episodes["week"].astype(str)))
    if lag_keys != song_keys:
        raise ValueError("Simulators used different SKU-week evaluation rows")
    lag_bounds = lag_bounds.sort_values("sku").reset_index(drop=True)
    song_bounds = song_bounds.sort_values("sku").reset_index(drop=True)
    if (lag_bounds["sku"].tolist() != song_bounds["sku"].tolist()
            or not np.allclose(lag_bounds[["price_min", "price_max"]],
                               song_bounds[["price_min", "price_max"]])):
        raise ValueError("Simulators used different training price support")

    records = []
    for label, frame in (("lag_adjusted_kernel", lag_runs), ("song_kernel", song_runs)):
        frame = frame.copy()
        frame["simulator_variant"] = label
        fixed = frame.loc[frame["policy"].eq("fixed_mean"), ["seed", "simulated_revenue"]]
        fixed = fixed.rename(columns={"simulated_revenue": "checked_fixed_revenue"})
        frame = frame.merge(fixed, on="seed", validate="many_to_one")
        frame["checked_uplift_pct"] = 100 * (
            frame["simulated_revenue"] / frame["checked_fixed_revenue"] - 1
        )
        if not np.allclose(frame["checked_uplift_pct"], frame["uplift_vs_fixed_pct"], atol=1e-8):
            raise ValueError("Stored uplift does not match the paired fixed-price baseline")
        records.append(frame)
    all_runs = pd.concat(records, ignore_index=True)
    boundary_records = []
    for label, episodes, bounds in (("lag_adjusted_kernel", lag_episodes, lag_bounds),
                                    ("song_kernel", song_episodes, song_bounds)):
        priced = episodes.merge(bounds, on="sku", how="left", validate="many_to_one")
        if priced[["price_min", "price_max"]].isna().any().any():
            raise ValueError("A priced SKU has no train price bounds")
        priced["at_train_min"] = priced["chosen_price"].le(priced["price_min"] + 1e-6)
        priced["at_train_max"] = priced["chosen_price"].ge(priced["price_max"] - 1e-6)
        grouped = priced.groupby(["seed", "policy"], as_index=False).agg(
            at_train_min_pct=("at_train_min", "mean"),
            at_train_max_pct=("at_train_max", "mean"),
        )
        grouped[["at_train_min_pct", "at_train_max_pct"]] *= 100
        grouped["simulator_variant"] = label
        boundary_records.append(grouped)
    boundaries = pd.concat(boundary_records, ignore_index=True)
    all_runs = all_runs.merge(boundaries, on=["simulator_variant", "seed", "policy"],
                              how="left", validate="one_to_one")
    summary = all_runs.groupby(["simulator_variant", "policy"], as_index=False).agg(
        seeds=("seed", "nunique"),
        simulated_revenue_mean=("simulated_revenue", "mean"),
        simulated_revenue_sd=("simulated_revenue", "std"),
        uplift_vs_fixed_mean_pct=("checked_uplift_pct", "mean"),
        uplift_vs_fixed_sd_pct=("checked_uplift_pct", "std"),
        training_skus_seen_min=("training_distinct_skus_seen", "min"),
        training_skus_seen_mean=("training_distinct_skus_seen", "mean"),
        at_train_min_mean_pct=("at_train_min_pct", "mean"),
        at_train_max_mean_pct=("at_train_max_pct", "mean"),
    )
    summary = summary.sort_values(["simulator_variant", "uplift_vs_fixed_mean_pct"],
                                  ascending=[True, False]).reset_index(drop=True)

    comparisons = (
        ("ppo_feedback_gradient", "ppo_frozen", "online_feedback"),
        ("ppo_feedback_gradient", "ppo_feedback_no_gradient", "price_gradient"),
        ("ppo_feedback_gradient", "ppo_frozen_no_gradient", "feedback_and_gradient"),
        ("ppo_feedback_gradient", "ddpg", "versus_ddpg"),
        ("ppo_feedback_gradient", "sac", "versus_sac"),
    )
    ablations = []
    for simulator, frame in all_runs.groupby("simulator_variant"):
        pivot = frame.pivot(index="seed", columns="policy", values="checked_uplift_pct")
        for full, reference, component in comparisons:
            if {full, reference}.issubset(pivot.columns):
                diff = pivot[full] - pivot[reference]
                ablations.append({
                    "simulator_variant": simulator, "comparison": component,
                    "full_policy": full, "reference_policy": reference,
                    "seeds": len(diff), "paired_uplift_difference_mean_pp": diff.mean(),
                    "paired_uplift_difference_sd_pp": diff.std(ddof=1),
                })
    sensitivity = summary.pivot(index="policy", columns="simulator_variant",
                                values="uplift_vs_fixed_mean_pct").reset_index()
    sensitivity["lag_minus_song_uplift_pp"] = (
        sensitivity["lag_adjusted_kernel"] - sensitivity["song_kernel"])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(output / "paired_policy_runs.csv", index=False)
    summary.to_csv(output / "policy_summary.csv", index=False)
    pd.DataFrame(ablations).to_csv(output / "ppo_ablations.csv", index=False)
    sensitivity.to_csv(output / "simulator_sensitivity.csv", index=False)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lag_dir": str(Path(lag_dir).resolve()),
        "song_dir": str(Path(song_dir).resolve()),
        "seeds": lag_manifest["seeds"],
        "training_timesteps_requested": lag_manifest["training_timesteps_requested"],
        "evaluation_cycles_per_seed_policy": lag_manifest["evaluation_cycles"],
        "evaluation_skus": lag_manifest["evaluation_skus"],
        "validation_selected_simulator_config_sha256": lag_manifest["simulator_config_sha256"],
        "scientific_status": "Exploratory simulator sensitivity, not a causal or measured revenue effect",
        "interpretation": (
            "Compare uplift versus fixed price within each simulator and seed. "
            "Absolute revenue across simulators is not directly compared. "
            "The Song-style kernel failed observed-price validation; the selected "
            "lag-adjusted kernel passed validation but did not beat the naive "
            "observed-price baseline on test. Counterfactual price response is unverified."
        ),
    }
    (output / "comparison_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lag-dir", type=Path, required=True)
    parser.add_argument("--song-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(compare(args.lag_dir, args.song_dir, args.output_dir).to_string(index=False))


if __name__ == "__main__":
    main()
