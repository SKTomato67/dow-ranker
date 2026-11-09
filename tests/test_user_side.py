"""7. A user who cannot retrain still gets a usable answer: a verdict on this week's output and two fixes.

Retraining stays with the maintainer (test_monitoring.py). The user can (a) recalibrate the interval widths on
their own recent outcomes - same model, same predictions, new widths - and (b) leave out stocks whose
predictions are extrapolations. The verdict tells them which applies.
"""
import math

import numpy as np
import pandas as pd
import pytest
from conftest import HELD_OUT

from dow_ranker import InputError, backtest, monitor, predict_week, recalibrate, user_verdict
from dow_ranker.data import TARGET
from dow_ranker.monitor import CAUTION, UNRELIABLE, USABLE, extrapolated
from dow_ranker.synthetic import make_panel

LAST = "2011-06-24"


def noisy(uci, factor=4):
    """Q2 outcomes `factor` times as volatile as the intervals were built for."""
    d = uci.copy()
    d.loc[d["date"] >= pd.Timestamp(HELD_OUT), TARGET] *= factor
    return d


def test_recalibrated_widths_are_the_conformal_quantile_of_recent_errors(uci, model):
    r = recalibrate(uci, model=model, as_of=LAST)
    # Expected value from the backtest: errors of the last 13 Q2 weeks whose outcome is known before LAST
    bt = backtest(uci, model=model, start=HELD_OUT)
    bt = bt[bt["date"] < pd.Timestamp(LAST)]
    err = np.sort(np.abs(bt["Y_rel"] - bt["pred"]).to_numpy())
    assert r.info["recalibrated"]["n_residuals"] == len(err) == 12 * 30
    for level in (0.5, 0.8, 0.9):
        assert r.halfwidth[f"{level:g}"] == pytest.approx(err[math.ceil((len(err) + 1) * level) - 1])


def test_recalibration_changes_only_the_widths(uci, model):
    r = recalibrate(uci, model=model, as_of=LAST)
    a = predict_week(uci, week=LAST, model=model).table
    b = predict_week(uci, week=LAST, model=r).table
    pd.testing.assert_series_equal(a["pred_rel_return"], b["pred_rel_return"])
    assert r.coef == model.coef and r.train_range == model.train_range
    assert r.halfwidth != model.halfwidth


def test_recalibration_uses_only_weeks_before_as_of(uci, model):
    as_of = pd.Timestamp("2011-06-03")
    base = recalibrate(uci, model=model, as_of=as_of).halfwidth
    changed = uci.copy()
    changed.loc[changed["date"] >= as_of, TARGET] = 50.0
    changed.loc[changed["date"] > as_of, ["open", "close"]] *= 2
    assert recalibrate(changed, model=model, as_of=as_of).halfwidth == base


@pytest.mark.parametrize("as_of", ["2011-05-20", "2011-04-15"])
def test_recalibration_needs_eight_weeks_of_outcomes(uci, model, as_of):
    with pytest.raises(InputError, match="at least 8 weeks"):
        recalibrate(uci, model=model, as_of=as_of)                      # 7 and 2 weeks known


def test_recalibration_needs_realized_returns(model):
    with pytest.raises(InputError, match="at least 8 weeks"):
        recalibrate(make_panel(n_weeks=12), model=model)


def test_recalibration_repairs_coverage_on_later_weeks(uci, model):
    """The real check: fitted on weeks before 2011-06-03, judged on the weeks after it."""
    data, as_of = noisy(uci), "2011-06-03"
    r = recalibrate(data, model=model, as_of=as_of)
    shipped = backtest(data, model=model, start=as_of)["covered"].mean()
    recal = backtest(data, model=r, start=as_of)["covered"].mean()
    assert shipped < 0.5 and recal == pytest.approx(0.8, abs=0.12)


def test_verdict_on_held_out_q2_is_usable(uci, model):
    v = user_verdict(monitor(uci, model=model, as_of=LAST), can_recalibrate=True)
    assert v.level == USABLE and v.actions == []


def test_verdict_when_intervals_fail_offers_the_fixes(uci, model):
    m = monitor(noisy(uci), model=model, as_of=LAST)
    can = user_verdict(m, can_recalibrate=True)
    cannot = user_verdict(m, can_recalibrate=False)
    assert can.level == cannot.level == UNRELIABLE
    assert "Recalibrate" in " ".join(can.actions) and "90 %" in " ".join(cannot.actions)
    fixed = user_verdict(m, recalibrated=True, can_recalibrate=True)
    assert fixed.level == USABLE and "recalibrated" in fixed.headline


def test_verdict_on_drift(model):
    p = make_panel(n_weeks=4)
    last = p["date"] == p["date"].max()
    some = last & p["stock"].isin(p["stock"].unique()[:3])
    most = last & p["stock"].isin(p["stock"].unique()[:20])
    for rows, level in [(some, CAUTION), (most, UNRELIABLE)]:
        q = p.copy()
        q.loc[rows, "volume"] *= 25
        v = user_verdict(monitor(q, model=model))
        assert v.level == level and "training range" in " ".join(v.actions)


def test_old_model_is_caution_not_unreliable(model):
    """Age alone is the maintainer's problem: the user is told to be careful, not that the output is broken."""
    v = user_verdict(monitor(make_panel(n_weeks=4, start="2026-09-04"), model=model))
    assert v.level == CAUTION and "maintainer" in " ".join(v.actions)


def test_extrapolated_ignores_the_market_return(model):
    p = make_panel(n_weeks=4)
    last = p["date"] == p["date"].max()
    p.loc[last, ["close", "high"]] = p.loc[last, ["close", "high"]].mul(1.04, axis=0)   # whole market +4 %
    p.loc[last & (p["stock"] == "AA"), "volume"] *= 25
    t = predict_week(p, model=model).table
    assert t["out_of_range"].str.contains("mkt_ret").all()
    assert extrapolated(t).sum() == 1 and extrapolated(t)[t["stock"] == "AA"].all()
