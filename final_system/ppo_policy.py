"""Load a frozen M5 PPO actor without importing the training runtime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path

from .bundle import RUNTIME, ModelBundle, read_json


DEFAULT_PPO_PATH = RUNTIME / "ppo" / "m5-policy.json"
SCHEMA_VERSION = 1
HIDDEN_WIDTH = 32
STATE_FEATURE_COUNT = 9


class PpoScopeError(LookupError):
    """The policy was not trained for this item and store."""


class PpoArtifactError(ValueError):
    """The configured policy cannot be used with the current demand bundle."""


def price_from_action(state, action: float) -> float:
    """Map the continuous actor action into this series' training price range."""
    low, high = state["price_min"], state["price_max"]
    if not math.isfinite(action):
        raise ValueError("PPO action must be finite")
    if action <= -1.0:
        return low
    if action >= 1.0:
        return high
    return low + (action + 1.0) * (high - low) / 2.0


def policy_observation(series_ids, series_id, previous_qty, previous_price, current_price, week: date, state):
    """Use only the preceding observation, requested price context and model state."""
    intercept, beta, phi, _ = state["params"]
    low, high = state["price_min"], state["price_max"]
    quantity_z = (previous_qty - state["qty_mean"]) / state["qty_sd"]
    lag_price_z = (previous_price - state["price_mean"]) / state["price_sd"]
    without_candidate_z = intercept + phi * (quantity_z - beta * lag_price_z)
    phase = math.tau * (week.isocalendar().week - 1) / 52.0
    features = (
        quantity_z,
        _relative_price(previous_price, low, high),
        _relative_price(current_price, low, high),
        without_candidate_z + beta * (low - state["price_mean"]) / state["price_sd"],
        without_candidate_z + beta * (high - state["price_mean"]) / state["price_sd"],
        beta,
        math.log(high / low),
        math.sin(phase),
        math.cos(phase),
    )
    if not all(map(math.isfinite, features)):
        raise ValueError("PPO observation must be finite")
    scaled = tuple(max(-8.0, min(8.0, value)) for value in features)
    return scaled + tuple(float(key == series_id) for key in series_ids)


def _relative_price(price, low, high):
    # Saturate before multiplying so an extreme finite context price cannot overflow.
    scaled = (price - low) / (high - low)
    return 2 * max(-3.5, min(4.5, scaled)) - 1


def policy_version(payload):
    packed = json.dumps({key: value for key, value in payload.items() if key != "version"},
                        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return f"m5-ppo-v1-{hashlib.sha256(packed).hexdigest()[:12]}"


@dataclass(frozen=True)
class DenseLayer:
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]

    def forward(self, values):
        return tuple(bias + sum(weight * value for weight, value in zip(row, values, strict=True))
                     for row, bias in zip(self.weights, self.bias, strict=True))


@dataclass(frozen=True)
class PpoPolicy:
    version: str
    series_ids: tuple[str, ...]
    layers: tuple[DenseLayer, ...]

    @property
    def scope(self):
        return [{"sku": sku, "store_id": store} for sku, store in (key.rsplit("@", 1) for key in self.series_ids)]

    def predict_action(self, observation):
        for layer in self.layers[:-1]:
            observation = tuple(map(math.tanh, layer.forward(observation)))
        return self.layers[-1].forward(observation)[0]

    def recommend(self, *, sku, store_id, previous_qty, previous_price, current_price, week, state) -> float:
        item, separator, qualified_store = sku.partition("@")
        key = f"{item}@{store_id}"
        if key not in self.series_ids or (separator and store_id != qualified_store):
            raise PpoScopeError(f"PPO policy does not cover {sku} at {store_id}")
        observation = policy_observation(self.series_ids, key, previous_qty, previous_price, current_price, week, state)
        return price_from_action(state, self.predict_action(observation))


def _load_layer(raw, input_count, output_count):
    weights = tuple(tuple(map(float, row)) for row in raw["weights"])
    bias = tuple(map(float, raw["bias"]))
    if (len(weights) != output_count or len(bias) != output_count or not all(map(math.isfinite, bias))
            or any(len(row) != input_count or not all(map(math.isfinite, row)) for row in weights)):
        raise ValueError("Invalid PPO actor layer")
    return DenseLayer(weights, bias)


def load_ppo_policy(bundle: ModelBundle, path=None) -> PpoPolicy | None:
    supplied_path = path or os.getenv("FINAL_SYSTEM_PPO_PATH")
    policy_path = Path(supplied_path or DEFAULT_PPO_PATH)
    try:
        payload = read_json(policy_path)
    except FileNotFoundError as exc:
        if supplied_path is None:
            return None
        raise PpoArtifactError(f"PPO policy was not found: {policy_path}") from exc
    except (OSError, ValueError) as exc:
        raise PpoArtifactError(f"Cannot read PPO policy: {policy_path}") from exc

    try:
        if payload["schema_version"] != SCHEMA_VERSION or payload["version"] != policy_version(payload):
            raise ValueError("Policy schema or content version is invalid")
        if payload["demand_bundle_sha256"] != bundle.manifest["file_sha256"]:
            raise ValueError("Policy was trained with a different ARIMAX model bundle")
        series = bundle.data["m5"]
        series_ids = tuple(payload["series_ids"])
        if (not series_ids or tuple(sorted(set(series_ids))) != series_ids
                or any(key not in series or series[key]["arimax"] is None for key in series_ids)):
            raise ValueError("Invalid policy item and store coverage")
        widths = (STATE_FEATURE_COUNT + len(series_ids), HIDDEN_WIDTH, HIDDEN_WIDTH, 1)
        layers = tuple(_load_layer(layer, inputs, outputs) for layer, inputs, outputs in
                       zip(payload["layers"], widths[:-1], widths[1:], strict=True))
        return PpoPolicy(payload["version"], series_ids, layers)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        raise PpoArtifactError(f"Invalid PPO policy at {policy_path}: {exc}") from exc
