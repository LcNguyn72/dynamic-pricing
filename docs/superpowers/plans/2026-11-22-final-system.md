# Final System Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. Steps use checkbox syntax for tracking.

**Goal:** Deliver the model service as a runnable `final_system/` package.

**Architecture:** Training reuses the V2 data and statistical estimators. Serving loads one verified bundle and exposes pure forecasts and price recommendations through FastAPI.

**Tech stack:** Python, FastAPI, NumPy, Pandas and statsmodels.

**Spec:** The approved package layout in the conversation and the V2 design under `docs/dynamic-pricing-final-design.md`.

## Global constraints

- Keep the existing backend, frontend and database files in their current state during this package handoff.
- Use UCI Online Retail 352 and M5 evaluation on the verified weekly splits.
- Train fixed configurations. Use the 60 previously selected SKU references from the validation policy.
- Run inference with history supplied explicitly by the application backend.

## Review focus

- Check the historical week boundary and reject duplicate or future observations.
- Distinguish a fitted M5 AR(1) or ARIMAX model from the latest-observation fallback.
- Verify bounds, reference availability, revenue arithmetic and deterministic tie-breaking.
- Load one verified model version per process and identify it in every response.
- Check the real model API over loopback using UCI and M5 historical cases.

## Implementation

- [x] Add failing tests for forecast, price-grid and bundle behavior.
- [x] Add fixed configuration, model bundle loading, validation and pricing logic.
- [x] Add data preparation, verified artifact import and fixed-statistics training commands.
- [x] Serve forecasts and recommendations through FastAPI. Exercise the published API with the full model bundle.
- [x] Document setup, model choices, API examples and coverage in the package README.

The regenerated weekly inputs match the previously audited hashes. Refitting both M5 estimators and recreating the validation references produced the same model payload as importing the audited parameters. The API checklist passed against that payload over loopback.
