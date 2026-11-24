# Forecast Reliability Experiments Implementation Plan

**Goal:** Tune UCI XGBoost against the live four-week mean and test whether shared M5 ARIMAX demand can cover additional SKU–store series.

**Architecture:** Train pooled UCI XGBoost on causal sales history, then compare the same hyperparameter search with calendar features. In M5, retain each series' local quantity dynamics and try a shrunk price coefficient from the same item in other stores within their common training price support. Report quality on established V2 test keys and coverage of additional price scenarios separately.

**Tech stack:** Python, Pandas, NumPy, XGBoost and the verified V2 artifacts.

**Spec:** Compare on validation before reporting frozen test outcomes. Keep experimental results separate from the live model. Explain the paired grid and PPO benchmarks.

## Review focus

- Keep target-week sales out of the feature matrix. Use known calendar dates and actual history before the target.
- Lock the UCI tuning choices on validation, including the scale applied to the XGBoost correction.
- Use train-only M5 coefficients and price support across stores. Keep forecast availability separate from authorization to recommend a new price.
- Check transferred effects on identical historical rows with a real price change within supported bounds.

## Implementation

- [x] Verify causal UCI features and same-item M5 transfer with focused tests.
- [x] Run UCI hyperparameter search and evaluate the winners on the frozen test set.
- [x] Run the M5 transfer check and compare coverage and errors.
- [x] Document both runs and explain the prior grid versus PPO feedback evidence.
