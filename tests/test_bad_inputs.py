"""3. Bad inputs are refused or warned about, never silently turned into a ranking."""
import numpy as np
import pandas as pd
import pytest

from dow_ranker import InputError, predict_week
from dow_ranker.synthetic import DOW_2011, make_panel

LAST = "2011-04-22"                # last week of make_panel()


def last_week(df):
    return df["date"] == pd.Timestamp(LAST)


def test_clean_input_ranks_all_stocks_without_warnings(panel, model):
    r = predict_week(panel, model=model)
    assert r.week == pd.Timestamp(LAST)
    assert len(r.table) == 30 and list(r.table["rank"]) == list(range(1, 31))
    assert r.warnings == [] and r.excluded == {}
    assert (r.table["lower"] < r.table["pred_rel_return"]).all() and (r.table["pred_rel_return"] < r.table["upper"]).all()


@pytest.mark.parametrize("drop", ["close", "volume", "date", "stock"])
def test_missing_column_is_refused(panel, drop):
    with pytest.raises(InputError, match=drop):
        predict_week(panel.drop(columns=drop))


def test_empty_input_is_refused():
    with pytest.raises(InputError):
        predict_week(pd.DataFrame(columns=["stock", "date", "open", "high", "low", "close", "volume"]))


@pytest.mark.parametrize("column, value, reason", [
    ("close", np.nan, "missing"),
    ("close", "n/a", "unparseable"),
    ("open", -5.0, "non-positive"),
    ("volume", 0, "non-positive"),
    ("low", 1e6, "high below low"),
])
def test_bad_value_excludes_only_that_stock(panel, column, value, reason):
    df = panel.astype({column: object})
    df.loc[last_week(df) & (df["stock"] == "BA"), column] = value
    r = predict_week(df)
    assert "BA" not in set(r.table["stock"]) and len(r.table) == 29
    assert reason in r.excluded["BA"] and any("excluded 1 stock" in w for w in r.warnings)


def test_open_close_outside_high_low_is_excluded(panel):
    df = panel.copy()
    df.loc[last_week(df) & (df["stock"] == "BA"), "close"] *= 1.5
    r = predict_week(df)
    assert r.excluded["BA"].startswith("open/close outside high-low")


def test_bad_previous_week_excludes_stock(panel):
    df = panel.copy()
    df.loc[(df["date"] == "2011-04-15") & (df["stock"] == "CAT"), "volume"] = -1
    assert "CAT" in predict_week(df).excluded


def test_missing_previous_week_excludes_stock(panel):
    df = panel[~((panel["date"] == "2011-04-15") & (panel["stock"] == "CVX"))]
    r = predict_week(df)
    assert "previous week" in r.excluded["CVX"]


def test_duplicate_rows_are_refused(panel):
    with pytest.raises(InputError, match="duplicate"):
        predict_week(pd.concat([panel, panel.iloc[[0]]]))


def test_week_not_in_data_is_refused(panel):
    with pytest.raises(InputError, match="not in the data"):
        predict_week(panel, week="2011-12-30")


def test_single_week_is_refused(panel):
    with pytest.raises(InputError, match="usable"):
        predict_week(panel[last_week(panel)])


def test_too_few_stocks_is_refused(panel):
    with pytest.raises(InputError, match="at least 5"):
        predict_week(panel[panel["stock"].isin(["AA", "AXP", "BA", "BAC"])])


def test_fewer_stocks_warns(panel):
    r = predict_week(panel[panel["stock"].isin(DOW_2011[:20])])
    assert len(r.table) == 20 and any("20 stocks instead of 30" in w for w in r.warnings)


def test_unseen_ticker_warns_but_predicts(uci, model):
    df = uci.copy()
    df.loc[df["stock"] == "AA", "stock"] = "NVDA"
    r = predict_week(df, week="2011-05-13", model=model)
    assert "NVDA" in set(r.table["stock"]) and any("NVDA" in w for w in r.warnings)


def test_values_outside_training_range_are_flagged(panel):
    df = panel.copy()
    df.loc[last_week(df) & (df["stock"] == "AXP"), "volume"] *= 1000
    r = predict_week(df)
    row = r.table.set_index("stock").loc["AXP"]
    assert "log_vol_ratio" in row["out_of_range"] and "log_dollar_vol" in row["out_of_range"]
    assert any("outside the training range" in w for w in r.warnings)
    assert (r.table.set_index("stock").drop("AXP")["out_of_range"] == "").all()


def test_dates_long_after_training_warn():
    r = predict_week(make_panel(start="2026-10-16"))
    assert any("after the training data ends" in w for w in r.warnings)


def test_non_friday_week_warns(panel):
    df = panel.copy()
    df.loc[last_week(df), "date"] = pd.Timestamp("2011-04-21")      # Good Friday 2011: week ends Thursday
    r = predict_week(df)
    assert len(r.table) == 30 and any("not a Friday" in w for w in r.warnings)


def test_uci_text_format_parses_like_numbers(panel):
    text = panel.copy()
    for c in ["open", "high", "low", "close"]:
        text[c] = text[c].map(lambda v: f"${v:.10f}")
    text["date"] = text["date"].map(lambda d: f"{d.month}/{d.day}/{d.year}")
    a, b = predict_week(panel).table, predict_week(text).table
    np.testing.assert_allclose(a["pred_rel_return"], b["pred_rel_return"], atol=1e-9)


def test_unknown_interval_level_is_refused(panel):
    with pytest.raises(ValueError, match="level"):
        predict_week(panel, level=0.95)
