"""Command line: dow-ranker fetch-data | predict | backtest | monitor | train."""
import argparse
import sys

import pandas as pd

from .data import InputError, fetch_uci, read
from .features import build_features
from .model import RidgeModel, fit
from .monitor import monitor
from .predict import backtest, backtest_summary, predict_week


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dow-ranker", description="Rank Dow stocks by predicted next-week relative return.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch-data", help="download the UCI Dow Jones Index data set")
    f.add_argument("--dest", default="data")

    p = sub.add_parser("predict", help="rank the stocks of one week")
    p.add_argument("csv")
    p.add_argument("--week", help="date of the week (default: latest in the file)")
    p.add_argument("--level", type=float, default=0.8, help="interval level: 0.5, 0.8 or 0.9")
    p.add_argument("--model", help="model JSON (default: the shipped Part II model)")
    p.add_argument("--out", help="write the table to this CSV")

    b = sub.add_parser("backtest", help="coverage, MAE and IC per week on data with realized returns")
    b.add_argument("csv")
    b.add_argument("--start", help="first week to evaluate (e.g. 2011-04-01 = after the training data)")
    b.add_argument("--level", type=float, default=0.8)
    b.add_argument("--model")

    mo = sub.add_parser("monitor", help="checks of the retraining rule (coverage, input drift, schedule) as of a week")
    mo.add_argument("csv")
    mo.add_argument("--as-of", help="week to judge from (default: latest in the file); later weeks are ignored")
    mo.add_argument("--level", type=float, default=0.8)
    mo.add_argument("--model")

    t = sub.add_parser("train", help="re-tune and refit the ridge on new data")
    t.add_argument("csv")
    t.add_argument("--since", help="first week to train on (recommended: 13 weeks before --until)")
    t.add_argument("--until", help="last week to train on (default: all)")
    t.add_argument("--week0", help="optional table filling the first week's lags (notebook: data/week0_yahoo.csv)")
    t.add_argument("--out", required=True, help="where to save the model JSON")

    a = ap.parse_args(argv)
    model = RidgeModel.load(getattr(a, "model", None)) if a.cmd in ("predict", "backtest", "monitor") else None
    pd.set_option("display.width", 160)
    try:
        if a.cmd == "fetch-data":
            print(f"saved {fetch_uci(a.dest)}")
        elif a.cmd == "predict":
            r = predict_week(a.csv, week=a.week, model=model, level=a.level)
            for n in r.notes:
                print(f"NOTE: {n}", file=sys.stderr)
            for w in r.warnings:
                print(f"WARNING: {w}", file=sys.stderr)
            cols = ["rank", "stock", "pred_rel_return", "lower", "upper", "out_of_range"]
            print(f"Week {r.week.date()}: predicted next-week return relative to the average stock (pp), "
                  f"{a.level:.0%} interval")
            print(r.table[cols].round(3).to_string(index=False))
            if a.out:
                r.table.to_csv(a.out, index=False)
        elif a.cmd == "backtest":
            s = backtest_summary(backtest(a.csv, model=model, level=a.level, start=a.start))
            print(s.round(3).to_string())
        elif a.cmd == "monitor":
            r = monitor(a.csv, model=model, level=a.level, as_of=a.as_of)
            print(f"As of week {r.as_of.date()}: {r.status.upper()}")
            for c in r.checks:
                print(f"  [{'RETRAIN' if c.triggered else 'ok'}] {c.text}")
            if r.retrain:
                print(f"  -> dow-ranker train <csv> --since {(r.as_of - pd.Timedelta(weeks=12)).date()} "
                      f"--until {r.as_of.date()} --out models/ridge_{r.as_of.date()}.json")
            return 1 if r.status == "retrain now" else 0
        elif a.cmd == "train":
            df = read(a.csv)
            if a.since:
                df = df[df["date"] >= pd.Timestamp(a.since)]
            if a.until:
                df = df[df["date"] <= pd.Timestamp(a.until)]
            week0 = pd.read_csv(a.week0, index_col="stock") if a.week0 else None
            m = fit(build_features(df, week0), info={"week0_lags": bool(a.week0)})
            m.save(a.out)
            print(f"alpha {m.alpha:.4g}, CV MAE {m.info['cv_mae']:.4f} (baseline 0: {m.info['cv_mae_baseline_zero']:.4f}), "
                  f"mean IC {m.info['cv_mean_ic']:.3f}; saved {a.out}")
    except InputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
