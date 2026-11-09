"""Monitoring and the retraining rule: is the shipped model still fit for use as of a given week?

Three checks, each with a fixed threshold:

  1. Interval coverage over the last WINDOW weeks with known outcomes falls more than COVERAGE_DROP below
     nominal (80 % -> below 70 %).                                                     -> retrain now
  2. More than OOR_SHARE of the stocks in the latest week have a stock-level feature outside the training
     range.                                                                            -> retrain now
     mkt_ret is left out of this check: it is one number shared by all stocks, so a single extreme market
     week (2011-04-29, -06-03, -06-10 in Q2) puts every stock out of range at once. Such weeks are shown
     as `extreme_market` instead.
  3. More than RETRAIN_EVERY_DAYS (one quarter) since the training data ended.          -> scheduled retrain due
     More than two quarters (the scheduled retrain was missed).                        -> retrain now

Checks 1-2 are triggers, check 3 is the regular schedule. Input drift (2) needs no outcomes, so it works on
the latest week straight away; coverage (1) needs realized next-week returns for a full WINDOW of weeks
(weekly coverage has sd ~0.12 in Q2, so an 8-week average has sd ~0.04 and 70 % is ~2.4 sd below 80 %).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import clean
from .features import build_features
from .model import RidgeModel
from .predict import row_problems

WINDOW = 8                         # weeks of outcomes for the rolling coverage (8 x 30 = 240 stock-weeks)
COVERAGE_DROP = 0.10               # Part III: retrain when 80 % intervals cover less than ~70 %
OOR_SHARE = 0.5                    # Part III: retrain when most inputs are out of range
RETRAIN_EVERY_DAYS = 91            # one quarter: the horizon the held-out test (Q1 -> Q2) supports

MARKET_FEATS = ["mkt_ret"]         # the same value for every stock of a week

OK, DUE, NOW = "ok", "scheduled retrain due", "retrain now"


@dataclass
class Check:
    name: str
    value: float                   # NaN if it cannot be computed yet
    threshold: float
    triggered: bool
    text: str
    judged: bool = True            # False: not enough data to judge yet (never triggered then)


@dataclass
class MonitorResult:
    as_of: pd.Timestamp
    train_end: pd.Timestamp
    weekly: pd.DataFrame           # one row per week after training: n, share_oor, coverage, rolling_coverage, ic, ...
    checks: list = field(default_factory=list)
    status: str = OK               # OK, DUE or NOW

    @property
    def retrain(self):
        return self.status != OK

    @property
    def next_scheduled(self):
        return self.train_end + pd.Timedelta(days=RETRAIN_EVERY_DAYS)


def _prepare(data, model, as_of, start):
    """Features of the valid rows from `start` (default: after training) up to `as_of` (default: latest week)."""
    df = clean(data if isinstance(data, pd.DataFrame) else pd.read_csv(data))
    df = df[df["date"].notna()]
    df = df[row_problems(df) == ""]
    as_of = df["date"].max() if as_of is None else pd.Timestamp(as_of)
    df = df[df["date"] <= as_of]
    train_end = pd.Timestamp(model.info["train_weeks"][1])
    start = train_end + pd.Timedelta(days=1) if start is None else pd.Timestamp(start)
    m = build_features(df)
    return m[m["date"] >= start].reset_index(drop=True), as_of, train_end


def _known(m, as_of):
    """Rows whose outcome is known at as_of: a week's outcome is known once the following week has closed."""
    known = m.dropna(subset=["Y_rel"]) if "Y_rel" in m else m.iloc[0:0]
    return known[known["date"] < as_of]


def monitor(data, model=None, level=0.8, as_of=None, start=None):
    """Monitoring table and retraining verdict using only the weeks up to `as_of` (default: latest week).

    data: cleaned frame or CSV path in the input format; the realized-return column is optional.
    start: first week to monitor (default: the day after the model's training data ends).
    """
    model = model or RidgeModel.load()
    m, as_of, train_end = _prepare(data, model, as_of, start)
    oor = model.out_of_range(m)
    m["oor"] = oor.drop(columns=MARKET_FEATS).any(axis=1).to_numpy()
    m["extreme_market"] = oor[MARKET_FEATS].any(axis=1).to_numpy()
    g = m.groupby("date")
    weekly = pd.DataFrame({"n": g.size(), "share_oor": g["oor"].mean(), "extreme_market": g["extreme_market"].any()})

    known = _known(m, as_of)
    if len(known):
        pred = model.predict(known)
        lo, hi = model.interval(pred, level)
        known = known.assign(pred=pred, covered=(known["Y_rel"] >= lo) & (known["Y_rel"] <= hi),
                             err=np.abs(known["Y_rel"] - pred), err0=known["Y_rel"].abs())
        k = known.groupby("date")
        weekly = weekly.join(pd.DataFrame({
            "coverage": k["covered"].mean(),
            "mae": k["err"].mean(),
            "mae_baseline_zero": k["err0"].mean(),
            "ic": k.apply(lambda w: w["pred"].corr(w["Y_rel"], method="spearman"), include_groups=False),
            "n_known": k.size(),
        }))
        # Rolling coverage pooled over the stock-weeks of the last WINDOW weeks with outcomes
        cov = weekly.dropna(subset=["coverage"])
        hits = k["covered"].sum().rolling(WINDOW, min_periods=1).sum()
        rows = cov["n_known"].rolling(WINDOW, min_periods=1).sum()
        weekly["rolling_hits"], weekly["rolling_rows"] = hits, rows
        weekly["rolling_coverage"] = hits / rows
        weekly["rolling_weeks"] = cov["n_known"].rolling(WINDOW, min_periods=1).count()
    weekly = weekly.reset_index(names="date")

    checks = []
    floor = level - COVERAGE_DROP
    last_cov = weekly.dropna(subset=["rolling_coverage"]).tail(1) if "rolling_coverage" in weekly else weekly.iloc[0:0]
    if len(last_cov) and last_cov["rolling_weeks"].iloc[0] >= WINDOW:
        c, nw = float(last_cov["rolling_coverage"].iloc[0]), int(last_cov["rolling_weeks"].iloc[0])
        below = last_cov["rolling_hits"].iloc[0] < floor * last_cov["rolling_rows"].iloc[0] - 1e-9   # counts, not floats
        checks.append(Check("coverage", c, floor, bool(below),
                            f"{level:.0%} intervals covered {c:.1%} of the last {nw} weeks with known outcomes "
                            f"(retrain below {floor:.0%})"))
    else:
        nw = int(last_cov["rolling_weeks"].iloc[0]) if len(last_cov) else 0
        checks.append(Check("coverage", float(last_cov["rolling_coverage"].iloc[0]) if len(last_cov) else np.nan,
                            floor, False, f"only {nw} week(s) with known outcomes; coverage is judged from "
                            f"{WINDOW} weeks on" + ("" if "Y_rel" in m else " (the data have no realized returns)"),
                            judged=False))

    latest = weekly[weekly["date"] == as_of]
    if len(latest):
        s = float(latest["share_oor"].iloc[0])
        checks.append(Check("input_drift", s, OOR_SHARE, s > OOR_SHARE,
                            f"{s:.0%} of the stocks in week {as_of.date()} have a stock-level feature outside the training range "
                            f"(retrain above {OOR_SHARE:.0%})"))
    else:
        checks.append(Check("input_drift", np.nan, OOR_SHARE, False,
                            f"no features for week {as_of.date()} (it is in the training period or lacks a previous week)",
                            judged=False))

    age = (as_of - train_end).days
    overdue = age > 2 * RETRAIN_EVERY_DAYS                  # a whole quarter past the due date
    checks.append(Check("schedule", float(age), float(RETRAIN_EVERY_DAYS), age > RETRAIN_EVERY_DAYS,
                        (f"week {as_of.date()} is inside the training period" if age <= 0 else
                         f"{age} days since the training data ended ({train_end.date()}); regular retrain every "
                         f"{RETRAIN_EVERY_DAYS} days" + (" - overdue by more than a quarter" if overdue else ""))))

    triggered = {c.name for c in checks if c.triggered}
    status = (NOW if triggered & {"coverage", "input_drift"} or overdue
              else DUE if "schedule" in triggered else OK)
    return MonitorResult(as_of=as_of, weekly=weekly, checks=checks, status=status, train_end=train_end)


# ---------------------------------------------------------------------------------------------------------------
# The user's side. Retraining is the maintainer's duty (the rule above). A user who cannot retrain gets a verdict on
# this week's output and two things they can do themselves: recalibrate the intervals on their own recent outcomes,
# and leave out the stocks whose features lie outside the training range.

RECAL_WEEKS = 13                   # recalibrate on at most the last quarter of known outcomes ...
RECAL_MIN_WEEKS = WINDOW           # ... and at least as many weeks as the coverage check needs

USABLE, CAUTION, UNRELIABLE = "usable", "use with caution", "not reliable"


def recalibrate(data, model=None, as_of=None):
    """Copy of `model` whose interval half-widths come from the user's own recent outcomes.

    Split conformal, as in training: for every interval level, the ceil((n+1) level)-th smallest absolute error of
    the model on the last RECAL_WEEKS weeks whose outcome is known at `as_of` (after the training period; at least
    RECAL_MIN_WEEKS weeks). Coefficients, predictions and training ranges stay the same; only the widths change.
    Raises InputError if there are not enough weeks with realized returns.
    """
    from .data import InputError
    from .model import conformal_halfwidth

    model = model or RidgeModel.load()
    m, as_of, _ = _prepare(data, model, as_of, None)
    known = _known(m, as_of)
    weeks = sorted(known["date"].unique())[-RECAL_WEEKS:]
    if len(weeks) < RECAL_MIN_WEEKS:
        raise InputError(f"recalibration needs realized returns for at least {RECAL_MIN_WEEKS} weeks after the "
                         f"training period and before {as_of.date()}; the data have {len(weeks)}")
    known = known[known["date"].isin(weeks)]
    abs_err = np.abs(known["Y_rel"] - model.predict(known)).to_numpy()
    info = dict(model.info, recalibrated={"weeks": [str(weeks[0].date()), str(weeks[-1].date())],
                                          "n_residuals": len(abs_err), "shipped_halfwidth": dict(model.halfwidth)})
    return RidgeModel(features=model.features, mean=model.mean, scale=model.scale, coef=model.coef,
                      intercept=model.intercept, alpha=model.alpha, train_range=model.train_range,
                      halfwidth={k: conformal_halfwidth(abs_err, float(k)) for k in model.halfwidth}, info=info)


@dataclass
class Verdict:
    level: str                     # USABLE, CAUTION or UNRELIABLE
    headline: str
    actions: list = field(default_factory=list)    # what the user can do now
    notes: list = field(default_factory=list)      # context, no action needed


def user_verdict(m, recalibrated=False, can_recalibrate=False):
    """Can the user rely on this week's output? From a MonitorResult (computed with the shipped model).

    recalibrated: the user is already using intervals recalibrated on their data.
    can_recalibrate: the data allow it (enough weeks with realized returns).
    """
    c = {ch.name: ch for ch in m.checks}
    cov, drift, sched = c["coverage"], c["input_drift"], c["schedule"]
    red, yellow, actions, notes = [], [], [], []

    if cov.triggered and not recalibrated:
        red.append(f"the {cov.threshold + COVERAGE_DROP:.0%} intervals are too narrow for your data: they covered "
                   f"{cov.value:.0%} of the outcomes of the last {WINDOW} weeks")
        actions.append("Switch on 'Recalibrate intervals on my data' in the sidebar: the widths are then taken from "
                       "the model's errors on your recent weeks." if can_recalibrate else
                       "Read the 90 % interval as roughly an 80 % one.")
    elif cov.triggered:
        notes.append(f"The shipped intervals failed on your data ({cov.value:.0%} coverage); you are using intervals "
                     "recalibrated on your recent weeks instead.")
    if drift.triggered:
        red.append(f"{drift.value:.0%} of the stocks this week have features the model never saw in training, so "
                   "their predictions are extrapolations")
        actions.append("Treat this week's ranking as unreliable. 'Only stocks inside the training range' on the "
                       "Ranking tab shows the few that are not extrapolated.")
    elif drift.judged and drift.value > 0:
        yellow.append(f"{drift.value:.0%} of the stocks this week are outside the training range")
        actions.append("Switch on 'Only stocks inside the training range' on the Ranking tab to leave them out.")
    if sched.triggered:
        yellow.append(f"the model is {int(sched.value)} days past its training data, longer than the one quarter "
                      "it was tested for")
        actions.append("Use the output with more caution. Retraining is done by the maintainer every quarter; "
                       "nothing for you to do.")

    if recalibrated and not cov.triggered:
        notes.append("Intervals recalibrated on your recent weeks: they assume next week is as volatile as those.")
    if not cov.judged:
        notes.append("Interval coverage could not be checked: that needs realized returns for "
                     f"{WINDOW} weeks after {m.train_end.date()}.")
    if len(m.weekly) and m.weekly["date"].iloc[-1] == m.as_of and m.weekly["extreme_market"].iloc[-1]:
        notes.append("This week's average market move was outside the training range (an extreme week for the whole "
                     "market); this shifts all stocks alike and does not change the ranking.")

    if red:
        level, reasons = UNRELIABLE, red
    elif yellow:
        level, reasons = CAUTION, yellow
    else:
        level, reasons = USABLE, (["the shipped intervals failed on your data, the recalibrated ones are in use"]
                                  if cov.triggered else ["no check found a problem"])
    headline = f"{level.capitalize()}: " + "; ".join(reasons) + "."
    return Verdict(level=level, headline=headline, actions=actions, notes=notes)


def extrapolated(table):
    """For a ranking table (Prediction.table): True where a stock-level feature is outside the training range.

    The market return is ignored: it is the same for every stock, so it does not single out any stock's prediction.
    """
    return table["out_of_range"].map(lambda s: any(f and f not in MARKET_FEATS for f in s.split(", "))).astype(bool)
