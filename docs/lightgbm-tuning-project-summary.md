# Per-Quantile LightGBM Tuning for Credit-Spread Forecasting — Project Summary

A short overview of the LightGBM hyperparameter-optimization work in the
BAA10Y credit-spread forecasting use case: what problem it solves, what was
built, and what it achieved. Links at the end point to the detailed design
docs.

## The problem

The BAA10Y use case forecasts the change in the US corporate credit spread
(Moody's Baa yield minus the 10-year Treasury yield) **1, 5 and 21 business
days ahead**, as full probability distributions scored by **CRPS** (lower is
better). LightGBM was already a strong model, but in quantile mode it trains
**13 independent boosters** — one per quantile from 2.5% to 97.5% — and all of
them shared one default configuration. A tail quantile and the median are
solving different problems, and the univariate and covariate variants see very
different feature counts, so one shared config is a compromise for all of
them.

## What was built

- **Per-quantile hyperparameters in the predictor.** `DartsLightGBMPredictor`
  gained an opt-in `per_quantile_kwargs` seam, so each of the 13 boosters can
  receive its own config. Existing call sites are unchanged.
- **A compact search space.** Tuning 7 parameters × 13 quantiles means 91
  numbers. Instead, each parameter is a line in "distance from the median"
  (`base + slope × tail_distance`), giving a symmetric **14-dimensional**
  search that Optuna can cover in tens of trials.
- **Leakage-safe validation.** Tuning validates only on origins strictly before
  the evaluation cutoff, and the check raises an error if it is violated. For
  the COVID-crash scenario, validation covers the **2008–09 financial crisis**,
  so the chosen hyperparameters are tested on a stress period they never
  saw.
- **Production-style tuning runs.** The Optuna study is saved in SQLite so a
  run can resume after interruption, trials run in parallel, and each
  scenario uses its own study file so that one scenario's tuned config can
  never leak into another.
- **Faster evaluation.** Walk-forward backtest origins now run on parallel
  threads (opt-in `n_jobs`). This exposed a real thread-safety race inside
  Darts' model construction, which was diagnosed and fixed with a small
  construction lock.
- **Tests and docs.** 24 focused unit tests cover the tuning module, and
  design guides explain each decision and the alternatives that were rejected.

## Results — COVID-crash stress test (Feb–Apr 2020, 64 daily origins)

Hyperparameters were tuned on the 5-day task and applied to all horizons. The
numbers are mean CRPS in basis points from the final run, which used the
"swap" covariate panel and 72 Optuna trials.

| Horizon | LightGBM: untuned → tuned | LightGBM + covariates: untuned → tuned |
|---|---|---|
| 5-day | 18.12 → 17.40 (**−4.0%**) | 19.12 → 18.97 (−0.8%) |
| 21-day | 84.66 → 72.07 (**−14.9%**) | 77.20 → 70.84 (**−8.2%**) |
| 1-day | 4.17 → 4.18 (+0.1%) | 3.93 → 4.12 (+4.8%) |

- Tuned LightGBM ranked **first of 10 methods** at both the 5-day and 21-day
  horizons. The other methods include naive, ETS, Kalman, AutoARIMA, linear
  regression with and without covariates, and untuned LightGBM.
- Against a naive "no change" forecast, tuned LightGBM's CRPS is about **26%
  lower at 5 days and 45% lower at 21 days**.
- The 1-day horizon did not benefit. The search optimized only for the 5-day
  task, so a config tuned for one horizon does not automatically carry over
  to another.
- **Speed:** parallel origins made the LightGBM evaluation backtest roughly
  **2–2.5× faster**. The like-for-like untuned LightGBM run went from 24.7 to
  11.9 minutes, and all four LightGBM variants together went from 6.6 to 2.5
  hours. With a full 10-method comparison now finishing in hours rather than
  overnight, it is practical to compare covariate panels and to run larger
  searches.

## Key lessons

- **Validation-window design mattered as much as the search algorithm.**
  Tuning on a calm period would have chosen configs that had never seen
  stress.
- **Saved tuning studies need scenario isolation.** A saved study keyed only
  by task ID would silently reuse another scenario's future-informed config.
- **Parallelism needs verification as well as speed.** It surfaced a latent
  race condition in a third-party library.

## Next steps

- Tune each horizon (1, 5 and 21 days) separately.
- Add a post-hoc diagnostic agent that reads finished trial histories and
  explains which quantiles and origins scored worst.

## Further reading

- [LightGBM per-quantile tuning guide](lightgbm-quantile-tuning-guide.md) — mechanics, search space, save/resume modes
- [Scenario: stress_2020](lgbm-tuning-scenario-stress-2020.md) and [Scenario: backtest_2025](lgbm-tuning-scenario-backtest-2025.md) — validation windows and the leakage risk between them
- [Backtest parallelism](backtest-parallelism.md) — design, rejected alternatives, and the Darts race-condition fix
- [Covariate panel: union vs. swap](covariate-panel-union-vs-swap.md)
- Code: [`lgbm_quantile_tuning.py`](../aieng-forecasting/aieng/forecasting/methods/numerical/lgbm_quantile_tuning.py), [`darts_regression.py`](../aieng-forecasting/aieng/forecasting/methods/numerical/darts_regression.py), notebook [`01_BAA10Y_multivariate_backtest.ipynb`](../implementations/BAA10Y_forecasting/01_BAA10Y_multivariate_backtest.ipynb)
