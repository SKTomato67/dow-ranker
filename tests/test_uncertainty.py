"""5. Intervals are honest: on the held-out Q2 weeks they cover about their nominal share.

The intervals come from the training folds only (split conformal), so Q2 is a genuine check.
Rows of the same week are correlated (one market shock moves all 30 stocks), so 390 rows are worth
far fewer independent cases; the tolerance below allows for that.
"""
import pytest
from conftest import HELD_OUT

from dow_ranker import backtest, backtest_summary

TOLERANCE = 0.05


@pytest.mark.parametrize("level", [0.5, 0.8, 0.9])
def test_held_out_coverage_is_close_to_nominal(uci, model, level):
    bt = backtest(uci, model=model, level=level, start=HELD_OUT)
    assert len(bt) == 13 * 30
    assert bt["covered"].mean() == pytest.approx(level, abs=TOLERANCE)


def test_interval_widths_increase_with_level(model):
    w = [model.halfwidth[k] for k in ("0.5", "0.8", "0.9")]
    assert w == sorted(w) and w[0] > 0


def test_model_is_not_better_than_baseline_claims(uci, model):
    """The report says the signal is ~zero: held-out MAE stays within 2 % of predicting 0."""
    s = backtest_summary(backtest(uci, model=model, start=HELD_OUT)).loc["all weeks"]
    assert abs(s["mae"] / s["mae_baseline_zero"] - 1) < 0.02
