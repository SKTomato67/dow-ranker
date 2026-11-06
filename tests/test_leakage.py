"""2. A prediction for week t uses only what is known at the close of week t; the target never enters."""
import numpy as np
import pandas as pd
import pytest

from dow_ranker import FEATS, backtest, predict_week
from dow_ranker.data import TARGET
from dow_ranker.predict import FUTURE_COLS

WEEK = "2011-05-13"


def preds(df, model, week=WEEK):
    return predict_week(df, week=week, model=model).table.set_index("stock")["pred_rel_return"].sort_index()


def test_no_future_column_is_a_feature(model):
    assert not set(FUTURE_COLS) & set(FEATS)
    assert model.features == FEATS


def test_future_columns_are_ignored(uci, model):
    rng = np.random.default_rng(0)
    scrambled = uci.copy()
    for c in FUTURE_COLS:
        scrambled[c] = rng.normal(size=len(uci)) * 100
    pd.testing.assert_series_equal(preds(uci, model), preds(scrambled, model))
    assert any("describe the future" in w for w in predict_week(uci, week=WEEK, model=model).notes)


def test_later_weeks_do_not_matter(uci, model):
    later = uci["date"] > WEEK
    changed = uci.copy()
    changed.loc[later, ["open", "high", "low", "close"]] *= 3
    changed.loc[later, "volume"] *= 10
    base = preds(uci, model)
    pd.testing.assert_series_equal(base, preds(changed, model))
    pd.testing.assert_series_equal(base, preds(uci[~later], model))


def test_only_weeks_t_and_t_minus_1_are_used(uci, model):
    old = uci["date"] < pd.Timestamp(WEEK) - pd.Timedelta(days=7)
    changed = uci.copy()
    changed.loc[old, "volume"] *= 50
    pd.testing.assert_series_equal(preds(uci, model), preds(changed, model))


def test_backtest_predictions_equal_live_predictions(uci, model):
    bt = backtest(uci, model=model)
    live = preds(uci, model)
    np.testing.assert_allclose(bt[bt["date"] == WEEK].set_index("stock")["pred"].sort_index(), live, atol=1e-12)


def test_backtest_realized_target_is_next_week(uci):
    bt = backtest(uci).set_index(["stock", "date"])
    row = uci.set_index(["stock", "date"]).loc[("AA", pd.Timestamp(WEEK))]
    week_mean = uci.loc[uci["date"] == WEEK, TARGET].mean()
    assert bt.loc[("AA", pd.Timestamp(WEEK)), "Y_rel"] == pytest.approx(row[TARGET] - week_mean)
