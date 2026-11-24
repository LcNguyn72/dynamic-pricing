# M5 Feature Ablation Implementation Plan

**Goal:** Measure the contribution of sales history, calendar context and price to one-week M5 forecasts.

**Architecture:** Read the verified V2 panels one store at a time. Fit the same XGBoost configuration on cumulative feature groups. Align every forecast with the saved Step 3 AR(1) and ARIMAX baselines and verify the frozen test keys. Keep outputs in `experiments/artifacts/`.

**Tech stack:** Python, NumPy, Pandas and XGBoost.

**Spec:** The agreed sequence in the conversation starts with recent sales history, then calendar and SNAP, then price. Select on validation and report RMSE, MAE and pooled R² for the unchanged test keys.

## Review focus

- Form sales and price lags solely from prior weeks and preserve missing observations.
- Train all four variants on the same observed train targets. Price at the target week is an explicitly known-price scenario.
- Use paired validation and test labels and report aggregate and per-store scores, alongside the deployed-model baselines.
- Keep the new trained models in experiments and leave the model API, frontend and database outside this change.

## Implementation

- [x] Verify causal features, per-SKU boundaries and metric pooling with small tests.
- [x] Run one store to validate the path, then run all ten stores and check the frozen test hash.
- [x] Compare the validation-selected price-aware model in the ARIMAX grid on fixed historical contexts.
- [x] Diagnose UCI baseline errors, evaluate rolling forecasts on the locked keys and compare the article's metric definitions.
- [x] Summarize results and commands in the experiment documentation.
