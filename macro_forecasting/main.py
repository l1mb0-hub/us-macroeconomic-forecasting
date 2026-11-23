"""US quarterly GDP forecasting with expanding-window evaluation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ['gdp_growth_l1', 'gdp_growth_l2', 'cons_growth_l1', 'inv_growth_l1',
            'inflation_l1', 'unemployment_l1', 'unemployment_change_l1', 'tbill_l1',
            'real_rate_l1']


def load_data(path):
    raw = pd.read_csv(path)
    required = {'year', 'quarter', 'realgdp', 'realcons', 'realinv', 'infl', 'unemp',
                'tbilrate', 'realint'}
    if not required.issubset(raw.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(raw.columns))}')
    raw = raw.sort_values(['year', 'quarter']).reset_index(drop=True)
    periods = pd.PeriodIndex.from_fields(year=raw.year.astype(int),
                                        quarter=raw.quarter.astype(int), freq='Q')
    if not periods.is_unique or not np.all(np.diff(periods.asi8) == 1):
        raise ValueError('Quarter index is not contiguous')
    if (raw[['realgdp', 'realcons', 'realinv']] <= 0).any().any():
        raise ValueError('Economic aggregates must be positive')
    df = pd.DataFrame(index=periods)
    df['gdp_growth'] = 400 * np.log(raw.realgdp.to_numpy() / raw.realgdp.shift(1).to_numpy())
    df['cons_growth'] = 400 * np.log(raw.realcons.to_numpy() / raw.realcons.shift(1).to_numpy())
    df['inv_growth'] = 400 * np.log(raw.realinv.to_numpy() / raw.realinv.shift(1).to_numpy())
    df['inflation'] = raw.infl.to_numpy()
    df['unemployment'] = raw.unemp.to_numpy()
    df['unemployment_change'] = raw.unemp.diff().to_numpy()
    df['tbill'] = raw.tbilrate.to_numpy()
    df['real_rate'] = raw.realint.to_numpy()
    feature_source = {'gdp_growth_l1': ('gdp_growth', 1), 'gdp_growth_l2': ('gdp_growth', 2),
                      'cons_growth_l1': ('cons_growth', 1), 'inv_growth_l1': ('inv_growth', 1),
                      'inflation_l1': ('inflation', 1), 'unemployment_l1': ('unemployment', 1),
                      'unemployment_change_l1': ('unemployment_change', 1),
                      'tbill_l1': ('tbill', 1), 'real_rate_l1': ('real_rate', 1)}
    design = pd.DataFrame(index=df.index)
    for name, (col, lag) in feature_source.items():
        design[name] = df[col].shift(lag)
    design['target_growth_next_q'] = df.gdp_growth.shift(-1)
    design['historical_mean_benchmark'] = df.gdp_growth.shift(1)
    return design.dropna().copy()


def forecast_one(train, current, alpha):
    x_train = train[FEATURES].to_numpy()
    y_train = train.target_growth_next_q.to_numpy()
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x_train, y_train)
    estimate = float(model.predict(current[FEATURES].to_numpy())[0])
    mean = float(np.mean(y_train))
    persistence = float(current.gdp_growth_l1.iloc[0])
    return estimate, mean, persistence


def expanding_backtest(data, start, end, alpha):
    rows = []
    for position in range(len(data)):
        quarter = data.index[position]
        if quarter < pd.Period(start, freq='Q') or quarter > pd.Period(end, freq='Q'):
            continue
        train = data.iloc[:max(0, position - 1)]
        if len(train) < 40:
            continue
        prediction, mean, persistence = forecast_one(train, data.iloc[[position]], alpha)
        rows.append({'decision_quarter': str(quarter), 'target_quarter': str(quarter + 1),
                     'actual': float(data.iloc[position].target_growth_next_q),
                     'ridge': prediction, 'mean': mean, 'persistence': persistence,
                     'train_n': len(train)})
    return pd.DataFrame(rows)


def metrics(frame, name):
    if frame.empty:
        raise ValueError('No observations available for evaluation')
    errors = frame[name] - frame.actual
    return {'n': int(len(frame)), 'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'mae': float(np.mean(np.abs(errors))),
            'bias': float(np.mean(errors))}


def newey_west_tstat(diffs, lags=4):
    x = np.asarray(diffs, dtype=float)
    n = len(x)
    residual = x - x.mean()
    long_variance = float(np.dot(residual, residual) / n)
    for lag in range(1, min(lags + 1, n)):
        cov = float(np.dot(residual[lag:], residual[:-lag]) / n)
        long_variance += 2 * (1 - lag / (lags + 1)) * cov
    if long_variance <= 0:
        return None
    return float(x.mean() / np.sqrt(long_variance / n))


def choose_alpha(design):
    alternatives = [0.1, 1.0, 10.0, 100.0, 1000.0]
    scores = {}
    for alpha in alternatives:
        validation = expanding_backtest(design, '1985Q1', '1994Q3', alpha)
        scores[str(alpha)] = metrics(validation, 'ridge')['rmse']
    winner = min(alternatives, key=lambda k: (scores[str(k)], k))
    return winner, scores


def run(data_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    design = load_data(data_path)
    alpha, selection = choose_alpha(design)
    test = expanding_backtest(design, '1995Q1', '2009Q2', alpha)
    if len(test) < 40:
        raise ValueError('Test sample unexpectedly small')
    scores = {name: metrics(test, name) for name in ['ridge', 'mean', 'persistence']}
    diff = (test.ridge - test.actual).pow(2) - (test['mean'] - test.actual).pow(2)
    tstat = newey_west_tstat(diff, lags=4)
    stress = test[test.target_quarter.str[:4].astype(int) >= 2007]
    summary = {'dataset': 'statsmodels macrodata (FRED and BLS)',
               'sample': {'first': str(design.index.min()), 'last': str(design.index.max()),
                          'validation': '1985Q1-1994Q3', 'test': '1995Q1-2009Q2'},
               'forecast_horizon': 'next-quarter GDP annualized log growth',
               'availability': 'predictors lagged one quarter; training labels have at least one-quarter publication buffer; revised-vintage caveat',
               'selected_alpha': alpha, 'validation_rmse_by_alpha': selection,
               'test_metrics': scores,
               'ridge_vs_mean_squared_loss_hac_tstat': tstat,
               'ridge_vs_mean_squared_loss_hac_p_approx': float(2 * norm.sf(abs(tstat))) if tstat is not None else None,
               'test_2007_onward': {k: metrics(stress, k) for k in scores},
               'no_live_trading_claim': True}
    test.to_csv(output / 'out_of_sample_forecasts.csv', index=False)
    (output / 'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(test.target_quarter, test.actual, label='Observed', linewidth=1.5)
    ax.plot(test.target_quarter, test.ridge, label='Ridge forecast', linewidth=1.2)
    ax.plot(test.target_quarter, test['mean'], label='Historical mean', alpha=.8)
    ax.axhline(0, color='grey', linewidth=.6)
    ax.set(xlabel='Target quarter', ylabel='Annualized GDP growth (%)',
           title='US GDP growth: genuinely held-out expanding-window forecasts')
    ticks = np.arange(0, len(test), 8)
    ax.set_xticks(ticks, labels=test.target_quarter.iloc[ticks], rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'forecasts.png', dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='macro_forecasting/data/us_macro_1959_2009.csv')
    parser.add_argument('--output', default='macro_forecasting/outputs')
    args = parser.parse_args()
    run(args.data, args.output)
