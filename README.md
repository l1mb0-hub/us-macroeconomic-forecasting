# US Macroeconomic Forecasting

## Research question

Can a regularized linear model predict US quarterly GDP growth better than simple historical benchmarks when evaluation is strictly chronological?

This study forecasts the next quarter's annualized log growth in real GDP using lagged GDP, consumption, investment, inflation, unemployment and Treasury bill rates. The exercise uses 203 actual US quarterly macroeconomic observations from 1959Q1 to 2009Q3. It is not a trading strategy.

## Data

`data/us_macro_1959_2009.csv` is a copy of the `statsmodels.datasets.macrodata` dataset, exported from statsmodels 0.14.6. The dataset is in the public domain. The underlying sources are FRED and the U.S. Bureau of Labor Statistics. See [statsmodels dataset description](https://www.statsmodels.org/stable/datasets/generated/macrodata.html) and [FRED](https://fred.stlouisfed.org/).

The 2009 cut-off is a limitation of the public bundled data, not a choice made after observing test results. These are historical revised figures rather than data vintages available to forecasters at the time; consequently, this is **not** a genuine real-time forecasting experiment. Do not interpret the estimates as live investment signals.

## Forecast protocol

At decision quarter `t`, the model predicts the GDP growth rate for `t+1`. Predictor values are lagged to `t-1`. The training set omits the most recent observation so it never needs a GDP-growth label for quarter `t` at decision time. This conservative buffer reduces calendar leakage but does not solve the revised-vintage limitation.

The first evaluation period, 1985Q1–1994Q3, selects Ridge regularization from a predeclared grid using expanding-window RMSE. The validation period ends at the 1994Q3 forecast origin, whose target is 1994Q4. This leaves time for that target to become available under the assumed publication delay before the first test forecast. The test forecast origins span 1995Q1–2009Q2 and use the selected hyperparameter throughout. The pipeline refits the scaler and model on the expanding historical sample; no fit operation uses evaluation data. Baselines are the prior historical mean and lagged growth persistence.

## Results

The model run writes holdout RMSE, MAE and bias for each model, plus an approximate Newey–West comparison of Ridge and historical-mean squared errors.

The corrected validation period selected Ridge alpha 1000. The test covers 58 target quarters.

| Model | RMSE (annualized percentage points) | MAE |
|---|---:|---:|
| Ridge | 2.751 | 1.909 |
| Historical mean | 2.772 | 1.901 |
| Persistence | 2.883 | 2.192 |

Ridge had slightly lower RMSE and slightly higher MAE than the historical mean. The approximate HAC p-value was 0.527, so the backtest provides little evidence of an improvement in mean squared error.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m macro_forecasting.main
```

The command writes `results.json`, a full out-of-sample forecast table and `forecasts.png`. No internet connection or API key is needed.
