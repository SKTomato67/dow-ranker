"""The six Part II features: everything is known at the Friday close of week t."""
import numpy as np
import pandas as pd

from .data import TARGET

WEEK_GAP = (5, 9)                  # days between two consecutive weekly rows of a stock
FEATS = ["rel_ret", "mkt_ret", "ret_lag1", "range", "log_vol_ratio", "log_dollar_vol"]


def build_features(df, week0=None):
    """Adds FEATS to a cleaned frame (data.clean); rows whose features cannot be computed are dropped.

    Week t uses only rows of weeks t and t-1: the lag needs the same stock 5-9 days earlier (holiday
    weeks end on Thursday, e.g. Good Friday 2011-04-22),
    and mkt_ret is the equal-weighted mean return of the stocks present in week t.
    week0: optional table (index stock; columns ret_week0, prev_weeks_volume) that fills the lags of
    the first week, as in the notebook (Yahoo data for 27-31 Dec 2010).
    If the target column is present, Y (next week's return) and Y_rel (Y minus its week mean) are added.
    """
    m = df.sort_values(["stock", "date"]).reset_index(drop=True)
    m["ret"] = (m["close"] - m["open"]) / m["open"] * 100              # this week's open-to-close return
    prev = m.groupby("stock")[["date", "volume", "ret"]].shift(1)
    gap = (m["date"] - prev["date"]).dt.days
    consecutive = gap.between(*WEEK_GAP)                                 # a missing week -> no lag
    m["previous_weeks_volume"] = prev["volume"].where(consecutive)
    m["ret_lag1"] = prev["ret"].where(consecutive)
    if week0 is not None:
        first = m["date"] == m["date"].min()
        m.loc[first, "previous_weeks_volume"] = m.loc[first, "stock"].map(week0["prev_weeks_volume"])
        m.loc[first, "ret_lag1"] = m.loc[first, "stock"].map(week0["ret_week0"])
    m["range"] = (m["high"] - m["low"]) / m["open"] * 100                # weekly high-low range (volatility proxy)
    m["log_vol_ratio"] = np.log(m["volume"] / m["previous_weeks_volume"])
    m["log_dollar_vol"] = np.log(m["volume"] * m["close"])
    m["mkt_ret"] = m.groupby("date")["ret"].transform("mean")           # equal-weighted market return in week t
    m["rel_ret"] = m["ret"] - m["mkt_ret"]
    m = m.dropna(subset=FEATS).reset_index(drop=True)
    if TARGET in m:
        m["Y"] = m[TARGET]
        m["Y_rel"] = m["Y"] - m.groupby("date")["Y"].transform("mean")
    return m
