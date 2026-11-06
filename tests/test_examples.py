"""The two example files behave as documented: one clean, one with every kind of problem."""
import sys
from pathlib import Path

import pandas as pd
import pytest

from dow_ranker import InputError, clean, predict_week

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import make_example_data  # noqa: E402


def test_good_data_ranks_every_week_without_warnings():
    good = clean(make_example_data.good())
    for week in sorted(good["date"].unique())[1:]:
        r = predict_week(good, week=week)
        assert len(r.table) == 30 and r.warnings == [] and r.excluded == {}


def test_bad_data_reports_every_problem():
    bad = make_example_data.bad()
    r = predict_week(bad, week="2026-10-09")
    assert r.excluded == {
        "AA": "missing or unparseable value in week 2026-10-09",
        "AXP": "missing or unparseable value in week 2026-10-09",
        "BA": "non-positive price or volume in week 2026-10-09",
        "BAC": "high below low in week 2026-10-09",
        "CAT": "open/close outside high-low in week 2026-10-09",
        "CSCO": "no row for the previous week (5-9 days before 2026-10-09)",
        "CVX": "non-positive price or volume in week 2026-10-02",
    }
    text = " | ".join(r.warnings)
    for expected in ["unparseable date", "25 stocks instead of 30", "AAPL, NVDA", "after the training data ends",
                     "range (1 stocks)", "log_vol_ratio", "log_dollar_vol"]:
        assert expected in text
    flagged = r.table.set_index("stock")["out_of_range"]
    assert "range" in flagged["INTC"] and "log_vol_ratio" in flagged["DD"]


def test_bad_data_thin_week_is_refused():
    with pytest.raises(InputError, match="only 4 usable"):
        predict_week(make_example_data.bad())             # latest week: 4 stocks, a Thursday


def test_example_files_are_current():
    here = Path(make_example_data.__file__).parent
    for name, make in [("data_good.csv", make_example_data.good), ("data_bad.csv", make_example_data.bad)]:
        pd.testing.assert_frame_equal(pd.read_csv(here / name), pd.read_csv(pd.io.common.StringIO(make().to_csv(index=False))))
