"""Publish a model bundle by refitting fixed estimators or importing verified parameters."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels

from .bundle import (
    DEFAULT_MODEL_DIR, DEFAULT_PREPARED_ROOT, PACKAGE_ROOT, PROJECT_ROOT, RUNTIME,
    file_sha256, read_json, verify_file, write_bundle,
)
from .pipeline.estimators import ARIMAX, BASE, MAXITER, MIN_PAIRS, _limit_threads, fit_series
from .pipeline.panel_audit import _observed
from .pipeline.reference_demand import KernelDemandSimulator, LagAdjustedKernelSimulator
from .pipeline.verified_weekly import iter_weekly, load_contract, load_scalers


CONFIG = PACKAGE_ROOT / "configs" / "serving_v2.json"
PHASE1_ROOT = PROJECT_ROOT / "experiments" / "artifacts" / "phase1_v2"
STEP3 = PHASE1_ROOT.with_name("pricing_arimax_v2")
FOLLOWUP = PHASE1_ROOT.with_name("pricing_arimax_feedback_v2")
IMPORT_LOCK_SHA256 = "b56504bd1dfddc6acd60b182d2ecdaf983543d36c4b5c6d5df46ef9b2df075d1"
MODEL_KEYS = {BASE: "ar1", ARIMAX: "arimax"}
FIT_CONFIG = {"scope": "train_only", "order": [1, 0, 0], "min_train_pairs": MIN_PAIRS, "maxiter": MAXITER}
FIT_IDENTITY = {"fit_scope": FIT_CONFIG["scope"], "model_order": FIT_CONFIG["order"],
                "min_train_pairs": MIN_PAIRS, "maxiter": MAXITER}


def _inputs(root):
    root = Path(root).resolve()
    config = read_json(CONFIG)
    contract = load_contract(root)
    protocol = contract["protocol"]
    if config["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Serving configuration uses a different data protocol")
    for dataset, key in (("uci", "uci_online_retail"), ("m5", "m5")):
        dates = config["datasets"][dataset]
        source = protocol["datasets"][key]
        fields = ("train_end", "validation_start", "validation_end", "test_start",
                  "train_weeks", "validation_weeks", "test_weeks")
        if (any(dates[field] != source[field] for field in fields)
                or dates["test_keys_sha256"] != source["primary_test_key_sha256"]):
            raise ValueError(f"{dataset}: serving split differs from the frozen V2 split")
    if config["m5_fit"] != FIT_CONFIG:
        raise ValueError("Serving fit settings differ from the frozen estimator")
    uci_skus = sorted(load_scalers(root, "uci", contract))
    if len(uci_skus) != contract["phase1"]["uci"]["eligible_skus"]:
        raise ValueError("UCI vocabulary coverage differs from the prepared dataset")
    return root, config, contract, uci_skus


def _finalize(directory, config, contract, uci_skus, models, references, details):
    expected = set(config["pricing"]["reference_skus"])
    if references.keys() != expected or not references.keys() <= models.keys():
        raise ValueError("Validation price coverage differs from the frozen cohort")
    for sku, model in models.items():
        model["fixed_reference_price"] = references.get(sku)
    if len(models) != contract["phase1"]["m5"]["test_keys"]["series"]:
        raise ValueError("M5 model roster differs from the full weekly cohort")
    provenance = {"phase1_sha256": contract["phase1_sha256"], "protocol_sha256": contract["protocol_sha256"],
                  "fit_scope": "train_only", "references_selection_scope": "validation_only", **details}
    result = write_bundle(directory, {"uci_skus": uci_skus, "m5": models}, config, provenance)
    print(json.dumps({"model_dir": str(Path(directory).resolve()), "version": result["version"],
                      "coverage": result["coverage"]}, indent=2), flush=True)
    return result


def import_verified(root, directory, step3, followup):
    _, config, contract, uci_skus = _inputs(root)
    source = read_json(step3 / "run.json")
    fit = source["identity"]
    if (source["status"] != "complete" or source["selected"] != ARIMAX
            or fit["phase1_sha256"] != contract["phase1_sha256"] or fit["lock_sha256"] != IMPORT_LOCK_SHA256
            or any(fit[key] != value for key, value in FIT_IDENTITY.items())
            or source["test_keys_sha256"] != config["datasets"]["m5"]["test_keys_sha256"]
            or source["models_by_store"].keys() != contract["m5_stores"].keys()):
        raise ValueError("The saved Step 3 result does not match the deployment configuration")

    models = {}
    for store, meta in source["models_by_store"].items():
        path = verify_file(step3 / meta["models"], meta["models_sha256"])
        store_models = {}
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["model"] not in MODEL_KEYS:
                    continue
                sku, key = row["sku"], MODEL_KEYS[row["model"]]
                series = store_models.setdefault(sku, {})
                if not sku.endswith(f"@{store}") or key in series:
                    raise ValueError(f"Invalid or repeated saved parameter row for {sku}")
                series[key] = json.loads(row["parameters"]) if row["status"] == "ok" else None
        if (len(store_models) != contract["m5_stores"][store]["eligible_series"]
                or any(row.keys() != {"ar1", "arimax"} for row in store_models.values())):
            raise ValueError(f"{store}: saved model roster is incomplete")
        models.update(store_models)

    followup_run = read_json(followup / "run.json")
    selection = read_json(followup / "selection.json")
    evidence = followup_run["identity"]
    step3_hash = file_sha256(step3 / "run.json")
    if (followup_run["status"] != "complete" or evidence["step3_sha256"] != step3_hash
            or evidence["phase1_sha256"] != contract["phase1_sha256"]
            or evidence["fixed_price_sha256"] != selection["fixed_sha256"]):
        raise ValueError("Validation prices do not match the audited Step 3 and follow-up runs")
    path = verify_file(followup / "validation_fixed_prices.csv", selection["fixed_sha256"])
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    references = {row["sku"]: float(row["price"]) for row in rows}
    if len(rows) != len(references):
        raise ValueError("Duplicate SKU in validation reference prices")
    return _finalize(directory, config, contract, uci_skus, models, references,
                     {"mode": "import_verified", "step3_sha256": step3_hash,
                      "fixed_reference_sha256": selection["fixed_sha256"], "statsmodels": fit["statsmodels"],
                      "numpy": fit["numpy"]})


def _fit_one(task):
    sku, qty, price = task
    return sku, {key: fit_series(qty, price, variant)[0] for variant, key in MODEL_KEYS.items()}


def _train_tasks(weekly, train_weeks):
    train = weekly.loc[weekly["split"].eq("train"), ["sku", "observed_sales", "qty", "price"]].copy()
    train["qty"] = train["qty"].where(_observed(train))
    for sku, group in train.groupby("sku", sort=False, observed=True):
        if len(group) != train_weeks:
            raise ValueError(f"{sku}: incomplete train calendar")
        yield str(sku), group["qty"].to_numpy(dtype=float), group["price"].to_numpy(dtype=float)


def _reference_prices(weekly, config):
    pricing = config["pricing"]
    train = weekly.loc[weekly["split"].eq("train") & _observed(weekly)]
    band = pricing["gaussian_seasonal_bandwidth_weeks"]
    simulators = (KernelDemandSimulator(band).fit(train),
                  LagAdjustedKernelSimulator(pricing["sensitivity_weight"], band, pricing["ratio_limit"]).fit(train))
    last = pd.Timestamp(config["datasets"]["m5"]["validation_end"])
    dates = pd.date_range(end=last, periods=pricing["reference_validation_weeks"], freq="7D")
    prices = {}
    for sku, panel in weekly.groupby("sku", sort=True):
        panel = panel.set_index("week")
        episode = panel.loc[dates]
        if not episode["split"].eq("validation").all() or not _observed(episode).all():
            raise ValueError(f"{sku}: invalid reference-price validation calendar")
        anchor = panel.loc[dates[0] - pd.Timedelta(weeks=1)]
        if not anchor.observed_sales or anchor.price <= 0:
            raise ValueError(f"{sku}: invalid historical anchor for the price reference")
        surface = simulators[0].surfaces[sku]
        grid = np.linspace(surface.price_min, surface.price_max, pricing["grid_points"])
        scores = []
        for price in grid.tolist():
            total = 0.0
            for simulator in simulators:
                prior_qty, prior_price = float(anchor.qty), float(anchor.price)
                for row in episode.itertuples():
                    context = {"event_days": row.event_days, "snap_days": row.snap_days,
                               "qty_lag_1": prior_qty, "price_lag_1": prior_price}
                    prior_qty = simulator.predict_mean(sku, price, row.Index, context)
                    total += price * prior_qty
                    prior_price = price
            scores.append(total / len(simulators))
        prices[sku] = float(grid[int(np.argmax(scores))])
    return prices


def train_fixed(root, directory, cache, workers):
    root, config, contract, uci_skus = _inputs(root)
    models, reference_parts = {}, []
    n_weeks = config["datasets"]["m5"]["train_weeks"]
    reference_skus = set(config["pricing"]["reference_skus"])
    with ProcessPoolExecutor(max_workers=workers, initializer=_limit_threads) as pool:
        for store, weekly, _ in iter_weekly(root, "m5", contract, cache_root=cache):
            print(f"[final_system] {store}: fitting AR(1) and ARIMAX on {n_weeks} train weeks", flush=True)
            reference_parts.append(weekly.loc[weekly["sku"].isin(reference_skus)].copy())
            fitted = dict(pool.map(_fit_one, _train_tasks(weekly, n_weeks), chunksize=24))
            if len(fitted) != contract["m5_stores"][store]["eligible_series"] or fitted.keys() & models.keys():
                raise ValueError(f"{store}: fitted roster differs from the verified weekly panel")
            models.update(fitted)
            print(f"[final_system] {store}: done ({len(fitted):,} series)", flush=True)
    print("[final_system] Recreating the locked validation price references", flush=True)
    references = _reference_prices(pd.concat(reference_parts, ignore_index=True), config)
    return _finalize(directory, config, contract, uci_skus, models, references,
                     {"mode": "train_fixed", "statsmodels": statsmodels.__version__, "numpy": np.__version__})


def main():
    from .source_training import SOURCES, ENV_FILE, train_and_evaluate

    def load_source_env():
        if not ENV_FILE.is_file():
            return
        for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

    load_source_env()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    fit = sub.add_parser("train", help="Fit fixed configurations from the prepared weekly data")
    saved = sub.add_parser("import-verified", help="Publish the locally available audited model parameters")
    source_fit = sub.add_parser(
        "source-train", help="Prepare, chronologically test, and save a separate UCI or Lazada model run"
    )
    fit.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    fit.add_argument("--cache-dir", type=Path, default=RUNTIME / "cache")
    fit.add_argument("--workers", type=int, default=6)
    saved.add_argument("--prepared-root", type=Path, default=PHASE1_ROOT)
    saved.add_argument("--step3-root", type=Path, default=STEP3)
    saved.add_argument("--followup-root", type=Path, default=FOLLOWUP)
    source_fit.add_argument("--source", choices=SOURCES,
                            default=os.getenv("FINAL_SYSTEM_DATA_SOURCE", "uci").strip().lower())
    source_fit.add_argument("--output-root", type=Path, default=RUNTIME / "source_runs")
    for command in (fit, saved):
        command.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    if args.action == "train":
        if not 1 <= args.workers <= 61:
            parser.error("--workers must be between 1 and 61")
        train_fixed(args.prepared_root, args.model_dir, args.cache_dir, args.workers)
    elif args.action == "source-train":
        if args.source not in SOURCES:
            parser.error(f"FINAL_SYSTEM_DATA_SOURCE must be one of {SOURCES}")
        train_and_evaluate(args.source, args.output_root)
    else:
        import_verified(args.prepared_root, args.model_dir, args.step3_root, args.followup_root)


if __name__ == "__main__":
    main()
