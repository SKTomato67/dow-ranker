"""Reading weekly stock data and downloading the UCI Dow Jones Index data set."""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

REQUIRED = ["stock", "date", "open", "high", "low", "close", "volume"]
PRICES = ["open", "high", "low", "close"]
TARGET = "percent_change_next_weeks_price"     # only used for training and backtests, never as an input
UCI_URL = "https://archive.ics.uci.edu/static/public/312/dow+jones+index.zip"
UCI_FILE = "dow_jones_index.data"


class InputError(ValueError):
    """The input cannot be used for a prediction (the message says why)."""


def _to_number(s):
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        s = s.astype(str).str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")       # unparseable -> NaN, reported later as a missing value


def clean(df):
    """Checks the columns and converts types. Raises InputError if a required column is missing.

    Prices may carry a $ sign (as in the UCI file); dates may be ISO (2011-01-07) or US (1/7/2011).
    Values that cannot be parsed become NaN and are reported by the checks in predict.py.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise InputError("the input has no rows")
    df = df.rename(columns=lambda c: str(c).strip().lower())
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise InputError(f"missing required column(s): {', '.join(missing)}; needed: {', '.join(REQUIRED)}")
    out = df.copy()
    out["stock"] = out["stock"].astype(str).str.strip().str.upper()
    out["date"] = pd.to_datetime(out["date"], errors="coerce", format="mixed")
    for c in PRICES + ["volume"] + ([TARGET] if TARGET in out else []):
        out[c] = _to_number(out[c])
    return out.sort_values(["stock", "date"]).reset_index(drop=True)


def read(path):
    """Reads a CSV of weekly rows (one row per stock and week) and cleans it."""
    return clean(pd.read_csv(path))


def fetch_uci(dest="data"):
    """Downloads the UCI Dow Jones Index data set (CC BY 4.0) into dest; returns the file path."""
    try:                                            # trust the OS certificate store (corporate / antivirus TLS proxies)
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(UCI_URL, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    for name in (UCI_FILE, "dow_jones_index.names"):
        (dest / name).write_bytes(z.read(name))
    return dest / UCI_FILE

