"""Regularize the shared ARIMAX price coefficient with recent observed outcomes."""

import math


MAX_FEEDBACK = 32
MIN_FEEDBACK = 4
RIDGE = 32.0


def update_arimax(state, feedback):
    low, high = state["price_min"], state["price_max"]
    eligible = sorted((row for row in feedback
                       if low <= row.price <= high and low <= row.previous_price <= high),
                      key=lambda row: row.week)[-MAX_FEEDBACK:]
    if len(eligible) < MIN_FEEDBACK:
        return state, 0
    prices = [row.price for row in eligible]
    if max(prices) - min(prices) <= 1e-6:
        return state, 0

    intercept, train_beta, phi, variance = state["params"]
    qty_mean, qty_sd = state["qty_mean"], state["qty_sd"]
    price_mean, price_sd = state["price_mean"], state["price_sd"]
    numerator, denominator = RIDGE * train_beta, RIDGE
    for row in eligible:
        # Isolate beta in the ARIMAX one-step equation.
        x = (row.price - price_mean) / price_sd - phi * (row.previous_price - price_mean) / price_sd
        y = (row.qty - qty_mean) / qty_sd - intercept - phi * (row.previous_qty - qty_mean) / qty_sd
        numerator += x * y
        denominator += x * x
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return state, 0
    beta = numerator / denominator
    return {**state, "params": [intercept, beta, phi, variance]}, len(eligible)
