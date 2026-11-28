"""Evaluate train-only same-item price transfer for the shared M5 demand model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from final_system.bundle import DEFAULT_MODEL_DIR, load_bundle

from .forecast_feature_ablation import ARTIFACTS, KEYS, _json, _sha, metric_totals, scores
from .phase1_finalize import _observed
from .phase5_data import PHASE1_ROOT, iter_weekly, load_contract, verify_shard_test_keys


FACTORS = (0.0, 0.25, 0.5, 1.0)
MIN_DONORS = 2
PRICE_LIMIT = 0.05
PRICE_TOL = 1e-6
ELASTICITY_LIMIT = 5.0


def borrow_price_state(reference, qty_mean, donors):
    """Estimate a local slope and require two donor stores to support every tested price."""
    usable = [state for state in donors if state["qty_mean"] > 0
              and state["price_min"] - PRICE_TOL <= reference <= state["price_max"] + PRICE_TOL]
    if len(usable) < MIN_DONORS:
        return None
    low = max(reference * (1 - PRICE_LIMIT),
              sorted(state["price_min"] for state in usable)[MIN_DONORS - 1])
    high = min(reference * (1 + PRICE_LIMIT),
               sorted((state["price_max"] for state in usable), reverse=True)[MIN_DONORS - 1])
    elasticities = [state["price_slope"] * state["price_mean"] / state["qty_mean"] for state in usable]
    elasticity = float(np.median(np.clip(elasticities, -ELASTICITY_LIMIT, ELASTICITY_LIMIT)))
    if elasticity >= 0 or high - low <= PRICE_TOL:
        return None
    return {"low": min(low, reference), "high": max(high, reference),
            "slope": elasticity * qty_mean / reference, "donors": len(usable)}


def price_effect(price, last_price, reference, phi, slope):
    """The price contribution in the same ARIMAX(1,0,0) quantity equation."""
    return slope * ((price - reference) - phi * (last_price - reference))


def _store_records(weekly, series, donor_states):
    train = weekly["split"].eq("train") & _observed(weekly)
    anchors = weekly.loc[train].groupby("sku", sort=False)["price"].median()
    records = []
    for key, reference in anchors.items():
        fits = series[key]
        ident = {"sku": key, "store": key.rsplit("@", 1)[1], "train_price_reference": reference}
        if fits["arimax"] is not None:
            records.append({**ident, "status": "native_arimax"})
            continue
        ar1 = fits["ar1"]
        if ar1 is None:
            records.append({**ident, "status": "insufficient_local_dynamics"})
            continue
        item = key.rsplit("@", 1)[0]
        donors = [state for other, state in donor_states[item].items() if other != key]
        transfer = borrow_price_state(reference, ar1["qty_mean"], donors)
        intercept, phi, _ = ar1["params"]
        local = {**ident, "phi": phi, "qty_mean": ar1["qty_mean"], "center": intercept * ar1["qty_sd"]}
        if transfer is None:
            records.append({**local, "status": "fixed_price_only"})
        else:
            records.append({**local, "status": "transfer_candidate", **transfer})
    return pd.DataFrame(records)


def _store_scores(weekly, records, totals, key_counts):
    target = records.loc[records["status"].isin(("fixed_price_only", "transfer_candidate"))]
    if target.empty:
        return
    observed = _observed(weekly)
    panel = weekly.copy()
    for name in ("qty", "price"):
        panel[f"{name}_lag"] = panel[name].where(observed).groupby(panel["sku"], sort=False).shift(1)
    good = observed & panel["split"].isin(("validation", "test"))
    rows = panel.loc[good, KEYS + ["split", "qty", "price", "qty_lag", "price_lag"]]
    joined = rows.merge(target, on="sku", validate="many_to_one")
    joined = joined.loc[joined["qty_lag"].notna()].copy()
    joined["baseline"] = joined["qty_mean"] + joined["center"] + joined["phi"] * (joined["qty_lag"] - joined["qty_mean"])
    at_reference = ((joined["price"] - joined["train_price_reference"]).abs().le(PRICE_TOL)
                    & (joined["price_lag"] - joined["train_price_reference"]).abs().le(PRICE_TOL))
    scope = "fixed_price_at_train_reference"
    for split, samples in joined.loc[at_reference].groupby("split", sort=False):
        key_counts[(split, scope)].update(samples["sku"])
        totals[(split, scope, 0.0)] += metric_totals(samples["qty"], samples["baseline"].clip(lower=0))
    joined = joined.loc[joined["status"].eq("transfer_candidate")]
    if joined.empty:
        return
    previous_supported = (joined["price_lag"].ge(joined["low"] - PRICE_TOL)
                          & joined["price_lag"].le(joined["high"] + PRICE_TOL))
    current_supported = (joined["price"].ge(joined["low"] - PRICE_TOL)
                         & joined["price"].le(joined["high"] + PRICE_TOL))
    joined = joined.loc[previous_supported & current_supported].copy()
    joined["effect"] = price_effect(joined["price"], joined["price_lag"], joined["train_price_reference"],
                                    joined["phi"], joined["slope"])
    for split in ("validation", "test"):
        full = joined.loc[joined["split"].eq(split)]
        changes = full.loc[(full["price"] - full["price_lag"]).abs().gt(PRICE_TOL)]
        for scope, samples in (("within_transfer_band", full), ("price_changed_within_band", changes)):
            key_counts[(split, scope)].update(samples["sku"])
            if not samples.empty:
                for factor in FACTORS:
                    y = np.maximum(0, samples["baseline"] + factor * samples["effect"])
                    totals[(split, scope, factor)] += metric_totals(samples["qty"], y)


def run(root, output, model_dir, cache):
    contract = load_contract(root)
    bundle = load_bundle(model_dir)
    manifest = bundle.manifest
    if (manifest["provenance"]["phase1_sha256"] != contract["phase1_sha256"]
            or manifest["provenance"]["fit_scope"] != "train_only"):
        raise ValueError("Use a train-only model bundle from the same verified M5 panel")
    all_states = bundle.data["m5"]
    donor_states = defaultdict(dict)
    for key, fits in all_states.items():
        if fits["arimax"] is not None:
            donor_states[key.rsplit("@", 1)[0]][key] = fits["arimax"]
    parts = []
    totals = defaultdict(lambda: np.zeros(5))
    key_counts = defaultdict(set)
    for store, weekly, _ in iter_weekly(root, "m5", contract, cache_root=cache):
        verify_shard_test_keys(root, "m5", contract, store, weekly)
        records = _store_records(weekly, all_states, donor_states)
        _store_scores(weekly, records, totals, key_counts)
        parts.append(records)
        print(f"{store}: {records['status'].value_counts().to_dict()}", flush=True)
    coverage = pd.concat(parts, ignore_index=True)
    if coverage["sku"].duplicated().any() or set(coverage["sku"]) != set(all_states):
        raise ValueError("M5 coverage must enumerate the full model roster")
    results = pd.DataFrame([
        {"split": split, "scope": scope, "factor": factor, "n_sku_stores": len(key_counts[(split, scope)]), **scores(values)}
        for (split, scope, factor), values in totals.items()
    ])
    primary = results.loc[results["split"].eq("validation") & results["scope"].eq("price_changed_within_band")]
    selected = float(primary.sort_values(["rmse", "factor"]).iloc[0]["factor"]) if not primary.empty else 0.0
    output.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output / "series_coverage.csv", index=False)
    results.to_csv(output / "transfer_scores.csv", index=False)
    _json(output / "run.json", {
        "model_version": manifest["version"], "model_sha256": manifest["file_sha256"],
        "source_sha256": _sha(Path(__file__)), "phase1_sha256": contract["phase1_sha256"],
        "m5_test_keys_sha256": contract["phase1"]["m5"]["test_keys"]["sha256"],
        "min_donors": MIN_DONORS, "max_price_change_from_training_median": PRICE_LIMIT,
        "coverage": coverage["status"].value_counts().to_dict(),
        "validation_selected_factor": selected, "criterion": "RMSE on observed price changes within donor support",
    })
    print(coverage["status"].value_counts().to_string(), flush=True)
    print(results.to_string(index=False), flush=True)
    print(f"Validation-selected transfer factor: {selected}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cache-dir", type=Path, default=ARTIFACTS / "phase5_v2" / "data_cache")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "m5_demand_coverage_v2")
    args = parser.parse_args()
    run(args.prepared_root, args.output_dir, args.model_dir, args.cache_dir)


if __name__ == "__main__":
    main()
