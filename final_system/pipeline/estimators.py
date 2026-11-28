"""Fixed M5 AR(1) and ARIMAX(1,0,0) estimators."""

from __future__ import annotations

import warnings

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from threadpoolctl import threadpool_limits


BASE = "ar1_no_price"
ARIMAX = "arimax_price"
MIN_PAIRS = 12
MAXITER = 50


def _design(price, state, variant):
    if variant == BASE:
        return None
    standard_price = (np.where(np.isfinite(price) & (price > 0), price, state["price_mean"])
                      - state["price_mean"]) / state["price_sd"]
    return standard_price[:, None]


def _model(qty, price, state, variant):
    y = (qty - state["qty_mean"]) / state["qty_sd"]
    return SARIMAX(y, exog=_design(price, state, variant), order=(1, 0, 0), trend="c",
                   enforce_stationarity=True)


def fit_series(qty: np.ndarray, price: np.ndarray, variant: str):
    observed = np.isfinite(qty)
    adjacent = observed[:-1] & observed[1:]
    if adjacent.sum() < MIN_PAIRS:
        return None, "sparse_train"
    state = {"qty_mean": float(qty[observed].mean()), "qty_sd": float(qty[observed].std())}
    if state["qty_sd"] <= 1e-8:
        return None, "constant_demand"
    if variant != BASE:
        observed_price = price[observed]
        if not np.isfinite(observed_price).all() or np.any(observed_price <= 0):
            return None, "invalid_price"
        paired = np.r_[adjacent, False] | np.r_[False, adjacent]
        lo, hi = float(observed_price.min()), float(observed_price.max())
        if np.ptp(price[paired]) <= 1e-8:
            return None, "constant_price"
        state.update(price_min=lo, price_max=hi, price_mean=float(observed_price.mean()),
                     price_sd=float(observed_price.std()))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _model(qty, price, state, variant).fit(disp=False, maxiter=MAXITER)
    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
        return None, "fit_failed"
    if not result.mle_retvals.get("converged", False) or not np.isfinite(result.params).all():
        return None, "not_converged"
    state["params"] = result.params.tolist()
    if variant != BASE:
        coefficient = state["params"][result.param_names.index("x1")]
        state["price_slope"] = coefficient * state["qty_sd"] / state["price_sd"]
    return state, "ok"


def _limit_threads():
    threadpool_limits(limits=1)
