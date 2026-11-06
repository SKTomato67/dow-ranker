"""A synthetic weekly panel in the input format, for examples and tests that must run without the UCI data."""
import numpy as np
import pandas as pd


DOW_2011 = ["AA", "AXP", "BA", "BAC", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "HD", "HPQ", "IBM", "INTC", "JNJ",
            "JPM", "KO", "KRFT", "MCD", "MMM", "MRK", "MSFT", "PFE", "PG", "T", "TRV", "UTX", "VZ", "WMT", "XOM"]


def make_panel(n_stocks=30, n_weeks=4, start="2011-04-01", seed=0, own_sd=2.0):
    """Random-walk weekly OHLCV rows; scales chosen to fall inside the training ranges of the shipped model."""
    rng = np.random.default_rng(seed)
    stocks = (DOW_2011 + [f"S{i:02d}" for i in range(30, n_stocks)])[:n_stocks]   # 2011 Dow tickers first
    dates = pd.date_range(start, periods=n_weeks, freq="W-FRI")
    price = rng.uniform(30, 100, n_stocks)
    rows = []
    for d in dates:
        ret = rng.normal(0.2, 0.4) + rng.normal(0, own_sd, n_stocks)      # market move + own move, in %
        open_ = price * (1 + rng.normal(0, 0.003, n_stocks))
        close = open_ * (1 + ret / 100)
        high = np.maximum(open_, close) * (1 + rng.uniform(0.005, 0.02, n_stocks))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.005, 0.02, n_stocks))
        volume = np.round(np.exp(rng.normal(np.log(3e7), 0.2, n_stocks)))
        rows.append(pd.DataFrame({"stock": stocks, "date": d, "open": open_, "high": high, "low": low,
                                  "close": close, "volume": volume}))
        price = close
    return pd.concat(rows, ignore_index=True)
