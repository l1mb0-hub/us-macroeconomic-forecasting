from pathlib import Path

import numpy as np
import pandas as pd

from macro_forecasting.main import FEATURES, expanding_backtest, load_data, metrics, newey_west_tstat


DATA = Path(__file__).parent / 'data' / 'us_macro_1959_2009.csv'


def test_real_data_source_and_time_index():
    frame = load_data(DATA)
    assert len(frame) > 180
    assert frame.index.is_monotonic_increasing
    assert frame.index.is_unique
    assert frame.index.min() == pd.Period('1959Q4')
    assert frame.index.max() == pd.Period('2009Q2')
    assert np.isfinite(frame[FEATURES].to_numpy()).all()


def test_first_forecast_is_invariant_to_future_changes():
    original = load_data(DATA)
    early = expanding_backtest(original, '1995Q1', '1995Q2', 1000)
    changed = original.copy()
    changed.loc[changed.index >= pd.Period('1996Q1'), 'target_growth_next_q'] = 1000
    changed.loc[changed.index >= pd.Period('1996Q1'), FEATURES] = -1000
    later = expanding_backtest(changed, '1995Q1', '1995Q2', 1000)
    pd.testing.assert_frame_equal(early, later)


def test_training_buffer_excludes_current_growth_label():
    sample = load_data(DATA)
    out = expanding_backtest(sample, '1995Q1', '1995Q1', 1000)
    i = sample.index.get_loc(pd.Period('1995Q1'))
    assert int(out.train_n.iloc[0]) == i - 1


def test_metrics_and_hac():
    data = pd.DataFrame({'actual': [1., 2., 3.], 'ridge': [1., 1., 1.]})
    assert np.isclose(metrics(data, 'ridge')['rmse'], np.sqrt(5 / 3))
    assert newey_west_tstat([0.0, 0.0, 0.0]) is None
