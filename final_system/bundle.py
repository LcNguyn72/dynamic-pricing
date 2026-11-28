"""Publish and load immutable model data, with an atomic manifest pointer."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from tempfile import NamedTemporaryFile


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
RUNTIME = PACKAGE_ROOT / "runtime"
DEFAULT_MODEL_DIR = RUNTIME / "models"
DEFAULT_PREPARED_ROOT = RUNTIME / "prepared"
DEFAULT_RAW_ROOT = RUNTIME / "raw"
PROTOCOL_PATH = PACKAGE_ROOT / "configs" / "protocol_v2.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelBundle:
    manifest: dict
    data: dict


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_file(path, expected):
    if file_sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return path


def atomic_write(path, data):
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as tmp:
        staging = Path(tmp.name)
        tmp.write(data)
    try:
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def _validate(data):
    uci = data["uci_skus"]
    series = data["m5"]
    if not uci or len(uci) != len(set(uci)) or not series:
        raise ValueError("Invalid model bundle coverage")
    coverage = {"uci_skus": len(uci), "m5_series": len(series), "m5_ar1": 0, "m5_arimax": 0,
                "m5_validation_reference": 0}
    for sku, models in series.items():
        for key, count in (("ar1", 3), ("arimax", 4)):
            model = models[key]
            if model is None:
                continue
            values = [model["qty_mean"], model["qty_sd"], *model["params"]]
            if len(model["params"]) != count or model["qty_sd"] <= 0 or not all(map(math.isfinite, values)):
                raise ValueError(f"Invalid quantity parameters for {sku}")
            coverage[f"m5_{key}"] += 1
        model = models["arimax"]
        if model is not None:
            lo, hi, mean, sd = (model[key] for key in ("price_min", "price_max", "price_mean", "price_sd"))
            if not (all(map(math.isfinite, (lo, hi, mean, sd))) and 0 < lo < hi and sd > 0):
                raise ValueError(f"Invalid price parameters for {sku}")
        reference = models["fixed_reference_price"]
        if reference is not None:
            if model is None or not math.isfinite(reference) or not lo - 1e-6 <= reference <= hi + 1e-6:
                raise ValueError(f"Invalid validation reference for {sku}")
            coverage["m5_validation_reference"] += 1
    return coverage


def _model_version(config, digest):
    points = config["pricing"]["grid_points"]
    if type(points) is not int or points < 2:
        raise ValueError("Price grid must contain at least two points")
    version_basis = json.dumps({"models": digest, "config": config}, sort_keys=True, allow_nan=False).encode()
    return f"{config['id']}-{hashlib.sha256(version_basis).hexdigest()[:12]}"


def write_bundle(directory, data, config, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    coverage = _validate(data)
    packed = gzip.compress(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
                           mtime=0)
    digest = hashlib.sha256(packed).hexdigest()
    filename = f"models-{digest}.json.gz"
    version = _model_version(config, digest)
    manifest = {"schema_version": SCHEMA_VERSION, "version": version, "file": filename,
                "file_sha256": digest, "coverage": coverage, "config": config, "provenance": provenance}
    atomic_write(directory / filename, packed)
    atomic_write(directory / "manifest.json",
                 (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
    return manifest


def load_bundle(directory=DEFAULT_MODEL_DIR):
    directory = Path(directory)
    try:
        manifest = read_json(directory / "manifest.json")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"No model bundle in {directory}. Run final_system.training first.") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported model bundle schema")
    name, digest = manifest.get("file", ""), manifest.get("file_sha256", "")
    if not re.fullmatch(r"[a-f0-9]{64}", digest) or name != f"models-{digest}.json.gz":
        raise ValueError("Invalid model bundle filename")
    if manifest["version"] != _model_version(manifest["config"], digest):
        raise ValueError("Model version differs from the manifest configuration")
    payload = (directory / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("SHA256 mismatch in model bundle")
    data = json.loads(gzip.decompress(payload))
    if _validate(data) != manifest["coverage"]:
        raise ValueError("Model bundle coverage differs from manifest")
    return ModelBundle(manifest, data)
