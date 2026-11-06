"""Writes two simulated input files (no real market data, safe to share):

  data_good.csv  10 clean weeks of the 30 Dow stocks of 2011, with realized next-week returns,
                 so every tab of the app works and no week raises a warning.
  data_bad.csv   recent weeks (2026) with one problem of every kind the tool checks for.

Usage: python examples/make_example_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from dow_ranker.data import TARGET
from dow_ranker.synthetic import make_panel

OUT = Path(__file__).resolve().parent


def with_target(panel):
    """Adds the realized next-week return (UCI definition: next week's close vs next week's open, in %)."""
    p = panel.sort_values(["stock", "date"]).copy()
    nxt = p.groupby("stock")[["open", "close"]].shift(-1)
    p[TARGET] = (nxt["close"] - nxt["open"]) / nxt["open"] * 100
    return p


def good():
    return with_target(make_panel(n_weeks=10, start="2011-04-01", seed=2, own_sd=1.6))   # 2011-like spread of returns


def bad():
    p = make_panel(n_stocks=32, n_weeks=6, start="2026-09-04", seed=2)       # 30 Dow 2011 tickers + S30, S31
    p["stock"] = p["stock"].replace({"S30": "NVDA", "S31": "AAPL"})             # not in the training data
    p["volume"] = p["volume"].astype(object)
    week, prev = pd.Timestamp("2026-10-09"), pd.Timestamp("2026-10-02")
    at = lambda stock, d=week: (p["stock"] == stock) & (p["date"] == d)

    p.loc[at("AA"), "close"] = np.nan                                         # missing value
    p.loc[at("AXP"), "volume"] = "n/a"                                         # unparseable value
    p.loc[at("BA"), "open"] = -5.0                                             # negative price
    p.loc[at("BAC"), "low"] = p.loc[at("BAC"), "high"] * 1.05                  # high below low
    p.loc[at("CAT"), "close"] = p.loc[at("CAT"), "high"] * 1.10                # close above the week's high
    p = p[~at("CSCO", prev)]                                                   # previous week missing
    p.loc[at("CVX", prev), "volume"] = 0                                       # previous week unusable
    p.loc[at("DD"), "volume"] = p.loc[at("DD"), "volume"] * 40                 # volume spike: out of training range
    p.loc[at("INTC"), "high"] = p.loc[at("INTC"), "open"] * 1.25               # +25 % intraweek high: range out of range
    for s in ["JPM", "MSFT", "XOM"]:                                           # 2026-size dollar volume
        p.loc[at(s), "volume"] = p.loc[at(s), "volume"] * 25

    for c in ["open", "high", "low", "close"]:                                 # UCI text format with $ signs
        p[c] = p[c].map(lambda v: f"${v:.2f}" if pd.notna(v) else "")
    p["date"] = p["date"].dt.strftime("%Y-%m-%d")
    junk = p.iloc[[0]].assign(stock="GE", date="2026-13-45")                   # unparseable date
    thin = p[p["date"] == "2026-10-09"].tail(4).assign(date="2026-10-15")      # Thursday week with only 4 stocks
    return pd.concat([p, junk, thin], ignore_index=True)


if __name__ == "__main__":
    good().to_csv(OUT / "data_good.csv", index=False)
    bad().to_csv(OUT / "data_bad.csv", index=False)
    print(f"wrote {OUT / 'data_good.csv'} and {OUT / 'data_bad.csv'}")
