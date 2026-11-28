"""Train-only, observational demand surface for controlled pricing experiments.

The paper evaluates pricing policies inside a non-parametric simulator. This
module follows that *experimental structure*: it estimates a smooth conditional
mean from observed SKU-week price/demand pairs, then exposes identical price
bounds and demand responses to every policy. It is not a causal demand model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProductSurface:
    log_prices: np.ndarray
    quantities: np.ndarray
    week_numbers: np.ndarray
    event_days: np.ndarray
    snap_days: np.ndarray
    price_min: float
    price_max: float
    bandwidth_log_price: float
    residual_log_sd: float
    price_levels: int


def _silverman(values: np.ndarray) -> float:
    """Silverman's robust rule, with a floor for discrete retail prices."""
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 2:
        return float(np.log(1.05))
    sd = float(np.std(values, ddof=1))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    robust = float((q75 - q25) / 1.34)
    scale = min(sd, robust) if robust > 0 else sd
    return float(max(0.9 * scale * n ** (-0.2), np.log(1.05)))


def _week_number(week: Any) -> int:
    if isinstance(week, (int, np.integer)):
        return int(week)
    return int(pd.Timestamp(week).isocalendar().week)


class KernelDemandSimulator:
    """Gaussian kernel regression over the *training* price-demand support.

    The local-constant estimate is weighted by log price and cyclic week of
    year. There is no promotion feature in Online Retail II; M5 can supply one
    in a later dataset-specific adapter. The simulator must be validated at
    held-out observed prices before its policy rankings are interpreted.
    """

    def __init__(self, seasonal_bandwidth_weeks: float = 13.0):
        if seasonal_bandwidth_weeks <= 0:
            raise ValueError("seasonal_bandwidth_weeks must be positive")
        self.seasonal_bandwidth_weeks = float(seasonal_bandwidth_weeks)
        self.surfaces: dict[str, ProductSurface] = {}

    def fit(self, weekly: pd.DataFrame) -> "KernelDemandSimulator":
        required = {"sku", "week", "qty", "price"}
        missing = required - set(weekly.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        rows = weekly.loc[weekly["split"].eq("train")].copy() if "split" in weekly else weekly.copy()
        if "observed_sales" in rows:
            rows = rows.loc[rows["observed_sales"].astype(bool)]
        rows = rows.loc[(rows["price"] > 0) & (rows["qty"] >= 0)].copy()
        if rows.empty:
            raise ValueError("No observed training price-demand pairs")
        rows["sku"] = rows["sku"].astype(str)
        rows["week"] = pd.to_datetime(rows["week"])
        surfaces: dict[str, ProductSurface] = {}
        for sku, group in rows.groupby("sku", sort=True):
            if len(group) < 12 or group["price"].nunique() < 2:
                continue
            prices = group["price"].to_numpy(dtype=float)
            log_prices = np.log(prices)
            quantities = group["qty"].to_numpy(dtype=float)
            week_numbers = group["week"].dt.isocalendar().week.to_numpy(dtype=int)
            event_days = (pd.to_numeric(group["event_days"], errors="coerce").fillna(0).to_numpy(dtype=float)
                          if "event_days" in group else np.zeros(len(group), dtype=float))
            snap_days = (pd.to_numeric(group["snap_days"], errors="coerce").fillna(0).to_numpy(dtype=float)
                         if "snap_days" in group else np.zeros(len(group), dtype=float))
            bandwidth = _silverman(log_prices)
            base = ProductSurface(
                log_prices=log_prices,
                quantities=quantities,
                week_numbers=week_numbers,
                event_days=event_days,
                snap_days=snap_days,
                price_min=float(prices.min()),
                price_max=float(prices.max()),
                bandwidth_log_price=bandwidth,
                residual_log_sd=0.0,
                price_levels=int(np.unique(prices).size),
            )
            # Leave each training target out of its local fit when estimating
            # noise. In-sample residuals would suppress simulator variance.
            fitted = np.array([
                self._predict(base, p, w, {"event_days": event, "snap_days": snap}, exclude_index=i)
                for i, (p, w, event, snap) in enumerate(zip(prices, week_numbers, event_days, snap_days))
            ])
            residual = np.log1p(quantities) - np.log1p(fitted)
            sd = float(np.clip(np.std(residual, ddof=1), 0.05, 1.25))
            surfaces[sku] = ProductSurface(
                log_prices=log_prices,
                quantities=quantities,
                week_numbers=week_numbers,
                event_days=event_days,
                snap_days=snap_days,
                price_min=base.price_min,
                price_max=base.price_max,
                bandwidth_log_price=bandwidth,
                residual_log_sd=sd,
                price_levels=base.price_levels,
            )
        if not surfaces:
            raise ValueError("No SKU has at least 12 training weeks and two observed price levels")
        self.surfaces = surfaces
        return self

    def _surface(self, sku: str) -> ProductSurface:
        try:
            return self.surfaces[str(sku)]
        except KeyError as exc:
            raise KeyError(f"No fitted demand surface for SKU {sku}") from exc

    def _predict(self, surface: ProductSurface, price: float, week: Any,
                 context: Mapping[str, Any] | None = None,
                 exclude_index: int | None = None) -> float:
        if not np.isfinite(price) or price <= 0:
            raise ValueError("price must be positive and finite")
        lp = np.log(float(price))
        price_distance = (surface.log_prices - lp) / surface.bandwidth_log_price
        week_no = _week_number(week)
        raw_week_distance = np.abs(surface.week_numbers - week_no)
        week_distance = np.minimum(raw_week_distance, 52 - raw_week_distance)
        seasonal = np.exp(-0.5 * (week_distance / self.seasonal_bandwidth_weeks) ** 2)
        weights = np.exp(-0.5 * price_distance**2) * (0.15 + 0.85 * seasonal)
        if context is not None:
            event = float(context.get("event_days", 0) or 0)
            snap = float(context.get("snap_days", 0) or 0)
            # M5 calendar covariates are known before a price decision. The
            # price-drop proxy is deliberately excluded: it changes with the
            # candidate action and is not an independent promotion label.
            weights *= 0.2 + 0.8 * np.exp(-0.5 * ((surface.event_days - event) / 2.0) ** 2)
            weights *= 0.2 + 0.8 * np.exp(-0.5 * ((surface.snap_days - snap) / 2.0) ** 2)
        if exclude_index is not None:
            weights[exclude_index] = 0.0
        total = float(weights.sum())
        if total < 1e-10:
            nearest = np.argsort(np.abs(price_distance))[:3]
            if exclude_index is not None:
                nearest = nearest[nearest != exclude_index]
                if nearest.size == 0:
                    nearest = np.array([int(np.argmin(np.where(
                        np.arange(len(price_distance)) == exclude_index,
                        np.inf, np.abs(price_distance)
                    )))])
            return float(np.mean(surface.quantities[nearest]))
        return float(np.dot(weights, surface.quantities) / total)

    def predict_mean(self, sku: str, price: float, week: Any,
                     context: Mapping[str, Any] | None = None) -> float:
        """Expected simulated weekly units at a candidate price."""
        return max(0.0, self._predict(self._surface(sku), price, week, context))

    def sample(self, sku: str, price: float, week: Any, rng: np.random.Generator,
               context: Mapping[str, Any] | None = None) -> float:
        """Mean-preserving log-normal observation noise for reproducible runs."""
        surface = self._surface(sku)
        mean = self.predict_mean(sku, price, week, context)
        sd = surface.residual_log_sd
        return float(mean * np.exp(sd * rng.standard_normal() - 0.5 * sd * sd))

    def price_bounds(self, sku: str, current_price: float, max_change: float = 0.10) -> tuple[float, float]:
        """Keep every policy within training support and a common move limit."""
        if not 0 <= max_change < 1:
            raise ValueError("max_change must be in [0, 1)")
        surface = self._surface(sku)
        reference = float(np.clip(current_price, surface.price_min, surface.price_max))
        lo = max(surface.price_min, reference * (1 - max_change))
        hi = min(surface.price_max, reference * (1 + max_change))
        return float(lo), float(hi)

    def oracle_revenue(self, sku: str, week: Any, current_price: float | None = None, max_change: float = 0.10,
                       grid_size: int = 100, context: Mapping[str, Any] | None = None) -> float:
        """Common 100-price hindsight oracle over the observed train support.

        Passing ``current_price`` deliberately narrows the oracle to a local
        move range; policy comparisons use the default full support so every
        method receives the same oracle for a SKU-week.
        """
        if current_price is None:
            surface = self._surface(sku)
            lo, hi = surface.price_min, surface.price_max
        else:
            lo, hi = self.price_bounds(sku, current_price, max_change)
        prices = np.linspace(lo, hi, grid_size)
        return float(max(p * self.predict_mean(sku, float(p), week, context) for p in prices))

    def diagnostics(self) -> pd.DataFrame:
        rows = []
        for sku, surface in sorted(self.surfaces.items()):
            prices = np.exp(surface.log_prices)
            rows.append({
                "sku": sku,
                "n_train_weeks": len(prices),
                "n_price_levels": surface.price_levels,
                "price_min": surface.price_min,
                "price_max": surface.price_max,
                "price_cv": float(np.std(prices, ddof=1) / np.mean(prices)),
                "bandwidth_log_price": surface.bandwidth_log_price,
                "residual_log_sd": surface.residual_log_sd,
            })
        return pd.DataFrame(rows)


class LagAdjustedKernelSimulator(KernelDemandSimulator):
    """Validation-tuned extension that anchors the kernel level to last demand.

    The Gaussian price response is retained as a ratio relative to the last
    price, while ``recent_weight`` controls how much the last observed/simulated
    demand determines the level. This remains an observational simulator, not
    an estimate of a causal price effect.
    """

    def __init__(self, recent_weight: float, seasonal_bandwidth_weeks: float = 13.0,
                 ratio_limit: float = 2.0):
        super().__init__(seasonal_bandwidth_weeks)
        if not 0 <= recent_weight <= 1:
            raise ValueError("recent_weight must be in [0, 1]")
        if ratio_limit <= 1:
            raise ValueError("ratio_limit must exceed 1")
        self.recent_weight = float(recent_weight)
        self.ratio_limit = float(ratio_limit)

    def predict_mean(self, sku: str, price: float, week: Any,
                     context: Mapping[str, Any] | None = None) -> float:
        base = super().predict_mean(sku, price, week, context)
        if self.recent_weight == 0 or context is None:
            return base
        try:
            last_qty = float(context["qty_lag_1"])
            last_price = float(context["price_lag_1"])
        except (KeyError, TypeError, ValueError):
            return base
        if not np.isfinite(last_qty) or last_qty < 0 or not np.isfinite(last_price) or last_price <= 0:
            return base
        reference = super().predict_mean(sku, last_price, week, context)
        if reference <= 1e-8:
            return base
        ratio = float(np.clip(base / reference, 1 / self.ratio_limit, self.ratio_limit))
        anchored = last_qty * ratio
        return max(0.0, (1 - self.recent_weight) * base + self.recent_weight * anchored)
