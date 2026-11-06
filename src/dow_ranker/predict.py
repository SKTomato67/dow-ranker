"""Ranking the stocks of one week, with input checks; and backtests over many weeks."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import PRICES, REQUIRED, TARGET, InputError, clean
from .features import FEATS, WEEK_GAP, build_features
from .model import RidgeModel

MIN_STOCKS = 5                     # mkt_ret and the ranking make no sense on fewer stocks
EXPECTED_STOCKS = 30
STALE_DAYS = 92                    # warn when the week is more than ~a quarter after the training data
FUTURE_COLS = [TARGET, "next_weeks_open", "next_weeks_close", "days_to_next_dividend", "percent_return_next_dividend"]


@dataclass
class Prediction:
    week: pd.Timestamp
    table: pd.DataFrame            # one row per stock, best predicted first
    warnings: list = field(default_factory=list)
    excluded: dict = field(default_factory=dict)   # stock -> reason
    notes: list = field(default_factory=list)      # information, not a problem with the data


def _as_frame(data):
    return clean(data if isinstance(data, pd.DataFrame) else pd.read_csv(data))


def row_problems(df):
    """Reason why a row cannot be used, or '' if it is fine."""
    p = pd.Series("", index=df.index)
    checks = [
        (df[REQUIRED].isna().any(axis=1), "missing or unparseable value"),
        ((df[PRICES] <= 0).any(axis=1) | (df["volume"] <= 0), "non-positive price or volume"),
        (df["high"] < df["low"], "high below low"),
        ((df[["open", "close"]].lt(df["low"] * (1 - 1e-6), axis=0)
          | df[["open", "close"]].gt(df["high"] * (1 + 1e-6), axis=0)).any(axis=1), "open/close outside high-low"),
    ]
    for bad, reason in checks:
        p[bad & (p == "")] = reason
    return p


def predict_week(data, week=None, model=None, level=0.8):
    """Predicts next week's relative return (Y_rel) for every stock in `week` and ranks them.

    data: DataFrame (or CSV path) with columns stock, date, open, high, low, close, volume and at least
    the week before `week`. week: date of the week to rank (default: the latest in data).
    Returns a Prediction; raises InputError if no sensible prediction is possible.
    """
    model = model or RidgeModel.load()
    df = _as_frame(data)
    warn, excluded, notes = [], {}, []

    ignored = [c for c in FUTURE_COLS if c in df]
    if ignored:
        notes.append(f"ignored columns that describe the future: {', '.join(ignored)}")
    df = df[REQUIRED]                                   # nothing else can reach the features

    if df["date"].isna().any():
        warn.append(f"dropped {df['date'].isna().sum()} row(s) with an unparseable date")
        df = df[df["date"].notna()]
    dup = df.duplicated(["stock", "date"], keep=False)
    if dup.any():
        raise InputError(f"duplicate stock-week rows: {sorted(df.loc[dup, 'stock'].unique())}")

    week = df["date"].max() if week is None else pd.Timestamp(week)
    if week not in set(df["date"]):
        raise InputError(f"week {week.date()} is not in the data ({df['date'].min().date()} to {df['date'].max().date()})")
    df = df[df["date"] <= week]                         # only what is known at the Friday close of `week`
    df = df[df["date"] >= week - pd.Timedelta(days=WEEK_GAP[1])]   # weeks t and t-1 are all the features use

    in_week = set(df.loc[df["date"] == week, "stock"])
    problems = row_problems(df)
    for i in problems[problems != ""].index:
        if df.at[i, "stock"] in in_week:
            excluded[df.at[i, "stock"]] = f"{problems[i]} in week {df.at[i, 'date'].date()}"
    df = df[problems == ""]

    feats = build_features(df)
    feats = feats[feats["date"] == week].set_index("stock")
    for s in sorted(in_week - set(feats.index)):
        excluded.setdefault(s, f"no row for the previous week (5-9 days before {week.date()})")
    if excluded:
        warn.append(f"excluded {len(excluded)} stock(s): " + "; ".join(f"{s}: {r}" for s, r in sorted(excluded.items())))
    if len(feats) < MIN_STOCKS:
        raise InputError(f"only {len(feats)} usable stock(s) in week {week.date()}; need at least {MIN_STOCKS}")
    if len(feats) != EXPECTED_STOCKS:
        warn.append(f"{len(feats)} stocks instead of {EXPECTED_STOCKS}: the market return is their average, "
                    "so ranks are relative to this set")
    new = sorted(set(feats.index) - set(model.info.get("train_stocks", feats.index)))
    if new:
        warn.append(f"stocks not in the training data (the model does not use tickers, but check): {', '.join(new)}")
    if week.day_of_week != 4:
        warn.append(f"{week.date()} is not a Friday (a holiday week?); the model was trained on weekly closes")
    train_end = pd.Timestamp(model.info["train_weeks"][1]) if "train_weeks" in model.info else None
    if train_end is not None and (week - train_end).days > STALE_DAYS:
        warn.append(f"week {week.date()} is {(week - train_end).days} days after the training data ends "
                    f"({train_end.date()}); consider retraining (dow-ranker train)")

    oor = model.out_of_range(feats)
    if oor.values.any():
        counts = oor.sum()
        warn.append("features outside the training range (prediction is an extrapolation): "
                    + ", ".join(f"{f} ({n} stocks)" for f, n in counts[counts > 0].items()))

    pred = model.predict(feats)
    lo, hi = model.interval(pred, level)
    table = pd.DataFrame({"stock": feats.index, "pred_rel_return": pred, "lower": lo, "upper": hi,
                          "out_of_range": [", ".join(oor.columns[r]) for r in oor.values]})
    table = table.join(feats[FEATS].reset_index(drop=True))
    table = table.sort_values(["pred_rel_return", "stock"], ascending=[False, True]).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return Prediction(week=week, table=table, warnings=warn, excluded=excluded, notes=notes)


def backtest(data, model=None, level=0.8, start=None):
    """Predictions for every week of data from `start` on, joined with the realized Y_rel where known.

    For checking a model on held-out weeks and for monitoring: coverage, MAE and Spearman IC per week.
    Rows with invalid values are dropped as in predict_week (same features, no warnings).
    """
    model = model or RidgeModel.load()
    df = _as_frame(data)
    df = df[df["date"].notna() & (row_problems(df) == "")]
    m = build_features(df)
    if start is not None:
        m = m[m["date"] >= pd.Timestamp(start)]
    if "Y_rel" not in m:
        raise InputError(f"backtest needs the realized next-week return column '{TARGET}'")
    out = m[["stock", "date", "Y_rel"]].copy()
    out["pred"] = model.predict(m)
    out["lower"], out["upper"] = model.interval(out["pred"].to_numpy(), level)
    out["covered"] = out["Y_rel"].between(out["lower"], out["upper"])
    out["out_of_range"] = model.out_of_range(m).any(axis=1).to_numpy()
    return out.dropna(subset=["Y_rel"]).reset_index(drop=True)


def backtest_summary(bt):
    """Per-week MAE, coverage and Spearman IC of a backtest, plus an 'all weeks' row."""
    from scipy import stats
    g = bt.groupby("date")
    weekly = pd.DataFrame({
        "n": g.size(),
        "mae": g.apply(lambda w: (w["Y_rel"] - w["pred"]).abs().mean()),
        "mae_baseline_zero": g.apply(lambda w: w["Y_rel"].abs().mean()),
        "coverage": g["covered"].mean(),
        "ic": g.apply(lambda w: stats.spearmanr(w["pred"], w["Y_rel"])[0]),
    })
    total = weekly.agg({"n": "sum", "ic": "mean"})
    total["mae"] = (bt["Y_rel"] - bt["pred"]).abs().mean()
    total["mae_baseline_zero"] = bt["Y_rel"].abs().mean()
    total["coverage"] = bt["covered"].mean()
    weekly.index = weekly.index.strftime("%Y-%m-%d")
    weekly.loc["all weeks"] = total[weekly.columns]
    return weekly
