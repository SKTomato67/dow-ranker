"""6. Monitoring and the retraining rule: the flag fires when (and only when) the documented checks fail.

Rule: retrain every quarter (91 days after the training data ends); earlier if 80 % intervals cover less than
70 % over the last 8 weeks with known outcomes, or more than half of the stocks in the latest week have a
stock-level feature outside the training range. Two quarters without retraining -> retrain now.
"""
import pandas as pd
import pytest
from conftest import HELD_OUT

from dow_ranker import backtest, monitor
from dow_ranker.cli import main
from dow_ranker.data import TARGET
from dow_ranker.monitor import DUE, NOW, OK
from dow_ranker.synthetic import make_panel

LAST = "2011-06-24"


def check(result, name):
    return next(c for c in result.checks if c.name == name)


def test_held_out_q2_needs_no_retraining(uci, model):
    r = monitor(uci, model=model, as_of=LAST)
    assert r.status == OK and not r.retrain
    # Expected value from the (separately tested) backtest: the last 8 weeks whose outcome is known by LAST
    bt = backtest(uci, model=model, start=HELD_OUT)
    weeks = sorted(bt["date"].unique())
    last8 = bt[bt["date"].isin(weeks[weeks.index(pd.Timestamp(LAST)) - 8:weeks.index(pd.Timestamp(LAST))])]
    assert check(r, "coverage").value == pytest.approx(last8["covered"].mean())


def test_uses_only_weeks_up_to_as_of(uci, model):
    as_of = pd.Timestamp("2011-05-27")
    base = monitor(uci, model=model, as_of=as_of)
    later = uci["date"] > as_of
    changed = uci.copy()
    changed.loc[later, ["open", "close", "volume"]] *= 3
    changed.loc[changed["date"] >= as_of, TARGET] = 50.0          # this week's outcome is not known yet either
    for data in (changed, uci[~later]):
        r = monitor(data, model=model, as_of=as_of)
        assert [c.value for c in r.checks] == [c.value for c in base.checks] and r.status == base.status


def test_coverage_breakdown_triggers_retrain(uci, model):
    noisy = uci.copy()
    q2 = noisy["date"] >= pd.Timestamp(HELD_OUT)
    noisy.loc[q2, TARGET] *= 4                                        # 4x the volatility the intervals were built for
    r = monitor(noisy, model=model, as_of=LAST)
    assert check(r, "coverage").triggered and r.status == NOW


def test_coverage_is_not_judged_before_a_full_window(uci, model):
    noisy = uci.copy()
    noisy.loc[noisy["date"] >= pd.Timestamp(HELD_OUT), TARGET] *= 4
    r = monitor(noisy, model=model, as_of="2011-05-13")               # only 6 weeks with known outcomes
    assert not check(r, "coverage").triggered and r.status == OK


def test_input_drift_triggers_retrain(model):
    p = make_panel(n_weeks=4)
    last = p["date"] == p["date"].max()
    big = last & p["stock"].isin(p["stock"].unique()[:20])
    p.loc[big, "volume"] *= 25                                        # 2026-size dollar volume for 20 of 30 stocks
    r = monitor(p, model=model)
    assert check(r, "input_drift").value == pytest.approx(20 / 30)
    assert r.status == NOW


def test_extreme_market_week_is_not_drift(model):
    p = make_panel(n_weeks=4)
    last = p["date"] == p["date"].max()
    p.loc[last, ["close", "high"]] = p.loc[last, ["close", "high"]].mul(1.04, axis=0)   # whole market +4 %
    r = monitor(p, model=model)
    assert r.weekly["extreme_market"].iloc[-1]
    assert not check(r, "input_drift").triggered and r.status == OK


@pytest.mark.parametrize("start, status", [("2011-04-01", OK), ("2011-06-17", DUE), ("2011-09-30", NOW)])
def test_schedule(model, start, status):
    r = monitor(make_panel(n_weeks=4, start=start), model=model)      # last week: start + 3 weeks
    assert r.status == status


def test_cli_exit_code_signals_retrain_now(uci_path, tmp_path):
    assert main(["monitor", str(uci_path)]) == 0
    p = make_panel(n_weeks=4, start="2026-09-04")
    p.to_csv(tmp_path / "recent.csv", index=False)
    assert main(["monitor", str(tmp_path / "recent.csv")]) == 1
