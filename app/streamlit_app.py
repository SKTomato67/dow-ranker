"""Dow weekly ranker: streamlit run app/streamlit_app.py"""
import io
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dow_ranker import FEATS, InputError, RidgeModel, backtest, backtest_summary, clean, predict_week
from dow_ranker.data import TARGET, fetch_uci
from dow_ranker.monitor import (CAUTION, COVERAGE_DROP, NOW, OK, OOR_SHARE, RECAL_MIN_WEEKS, RECAL_WEEKS,
                                RETRAIN_EVERY_DAYS, UNRELIABLE, WINDOW, extrapolated, monitor, recalibrate,
                                user_verdict)
from dow_ranker.predict import row_problems
from dow_ranker.synthetic import make_panel

ROOT = Path(__file__).resolve().parents[1]
UCI = ROOT / "data" / "dow_jones_index.data"
EXAMPLES = {"Example: clean data": ROOT / "examples" / "data_good.csv",
            "Example: data with errors": ROOT / "examples" / "data_bad.csv"}
BLUE, GREY, RED = "#2a78d6", "#8a8984", "#e34948"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FEATURE_NAMES = {"rel_ret": "return vs. average stock (%)", "mkt_ret": "average return of all stocks (%)",
                 "ret_lag1": "last week's return (%)", "range": "high-low range (% of open)",
                 "log_vol_ratio": "log(volume / last week's volume)", "log_dollar_vol": "log(volume x close)"}
LEVELS = {"50 %": 0.5, "80 %": 0.8, "90 %": 0.9}

st.set_page_config(page_title="Dow weekly ranker", layout="wide")
model = RidgeModel.load()


@st.cache_resource(show_spinner="Downloading the UCI Dow Jones data...")
def ensure_uci():
    """The data are not in the repository; a fresh deployment downloads them once (CC BY 4.0)."""
    if not UCI.exists():
        try:
            fetch_uci(UCI.parent)
        except OSError:
            pass                                                                # offline: the examples still work
    return UCI.exists()


ensure_uci()


def read_upload(up):
    """Reads an uploaded CSV; semicolon files (Czech/European Excel) are read with decimal commas."""
    text = up.getvalue().decode("utf-8-sig", errors="replace")
    header = text.splitlines()[0] if text.strip() else ""
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, decimal="," if sep == ";" else ".")


def show_upload_help(expanded):
    with st.expander("What to upload", expanded=expanded):
        st.markdown("""
**What happens with your file.** The model is already trained (on Q1 2011 of the UCI Dow Jones data) and does not
change. Your file is only used to *predict*: the app ranks the stocks of one week of your data. If the file also
contains the realized next-week returns, the *Track record* tab checks how good those predictions were.

**Format.** A CSV file with **one row per stock and week**. Column names are not case-sensitive; extra columns are ignored.

| Column | Required | Format | Example |
|---|---|---|---|
| `stock` | yes | ticker or any name | `MSFT` |
| `date` | yes | last trading day of the week: `2011-05-13` or `5/13/2011` | `2011-05-13` |
| `open` | yes | price at the start of the week (a `$` sign is allowed) | `25.80` |
| `high` | yes | highest price of the week | `25.96` |
| `low` | yes | lowest price of the week | `24.95` |
| `close` | yes | price at the end of the week | `25.03` |
| `volume` | yes | number of shares traded in the week | `382293426` |
| `percent_change_next_weeks_price` | no | realized return of the *next* week in %, only for the Track record | `-1.88` |

**Rules.**
- At least **two consecutive weeks** for each stock (5 to 9 days apart): the model needs last week's return and volume.
- At least **5 stocks** per week, ideally about 30; the ranking is relative to the stocks in the file.
- **Weekly** data only. Daily or monthly rows are refused, because the weeks are not 5-9 days apart.
- Separator: comma with decimal point, or semicolon with decimal comma (Excel in Czech settings).
- Prices in the same currency for all stocks. Values the model never saw in training are allowed but flagged.

**Example** (real rows of the UCI data; the table above shows MSFT in week 2011-05-13):
```
stock,date,open,high,low,close,volume
AA,2011-05-06,17.27,17.96,16.83,17.15,225053559
AA,2011-05-13,17.16,17.62,16.75,17.10,111630753
AXP,2011-05-06,49.37,50.47,49.08,50.20,43997724
AXP,2011-05-13,50.06,50.46,49.22,49.49,35427164
```
Weekly prices are available e.g. from Yahoo Finance (*Historical data*, frequency *Weekly*).
""")
        good = EXAMPLES["Example: clean data"]
        if good.exists():
            st.download_button("Download an example file (10 clean weeks, 30 stocks)", good.read_bytes(),
                               "dow_ranker_example.csv", "text/csv")


def show_verdict(v, compact=False):
    """The user's answer to 'can I use this week's output?' with what they can do about it."""
    box = {UNRELIABLE: st.error, CAUTION: st.warning}.get(v.level, st.success)
    body = f"**{v.headline}**"
    if v.actions:
        body += "\n\n**What you can do:**\n" + "\n".join(f"- {a}" for a in v.actions)
    box(body)
    if v.notes and not compact:
        for n in v.notes:
            st.caption(n)


def show_ranking(r, level_label, verdict, recalibrated):
    if verdict is not None:
        show_verdict(verdict, compact=True)
    for n in r.notes:
        st.info(n)
    if r.warnings:
        with st.expander(f"Data warnings ({len(r.warnings)})"):
            for w in r.warnings:
                st.warning(w)
    t = r.table
    flagged = extrapolated(t)
    if flagged.any():
        if st.toggle(f"Only stocks inside the training range (hides {int(flagged.sum())} extrapolated)", key="only_in_range"):
            t = t[~flagged]
            st.caption("Ranks are kept from the full week, so gaps show where hidden stocks were.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Stocks ranked", len(t))
    c2.metric("Spread of predictions (top - bottom)", f"{t['pred_rel_return'].iloc[0] - t['pred_rel_return'].iloc[-1]:.2f} pp")
    c3.metric(f"Width of each {level_label} interval", f"{t['upper'].iloc[0] - t['lower'].iloc[0]:.2f} pp")

    base = alt.Chart(t).encode(y=alt.Y("stock:N", sort=alt.SortField("rank"), title=None))
    tooltip = ["rank", "stock", alt.Tooltip("pred_rel_return", format=".3f", title="prediction (pp)"),
               alt.Tooltip("lower", format=".2f"), alt.Tooltip("upper", format=".2f"), "out_of_range"]
    chart = (
        alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=GREY, strokeDash=[4, 4]).encode(x="x:Q")
        + base.mark_rule(color=BLUE, strokeWidth=2, opacity=0.45).encode(
            x=alt.X("lower:Q", title=f"next-week return vs. average stock (pp), {level_label} interval"), x2="upper:Q",
            tooltip=tooltip)
        + base.mark_point(color=BLUE, filled=True, size=70).encode(x="pred_rel_return:Q", tooltip=tooltip)
    ).properties(height=max(200, 18 * len(t)))
    st.altair_chart(chart, width="stretch", theme="streamlit")
    st.caption(f"About {1 - LEVELS[level_label]:.0%} of outcomes fall outside a {level_label} interval by design, so a "
               "single stock outside its interval is normal. The Monitoring tab checks whether it happens too often."
               + (" Intervals: recalibrated on your recent weeks." if recalibrated else ""))
    st.dataframe(t.round(4), hide_index=True, width="stretch")
    st.download_button("Download table (CSV)", t.to_csv(index=False), f"ranking_{r.week.date()}.csv", "text/csv")


def show_data(df, r, week):
    problems = row_problems(df[df["date"].notna()])
    ok = df[df["date"].notna()][problems == ""]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Stocks", df["stock"].nunique())
    c3.metric("Weeks", df["date"].nunique())
    c4.metric("Rows with problems", int((problems != "").sum() + df["date"].isna().sum()))
    bad = df[df["date"].notna()][problems != ""].assign(problem=problems[problems != ""])
    if len(bad) or df["date"].isna().any():
        with st.expander("Rows with problems (left out of everything below)", expanded=True):
            show = pd.concat([bad, df[df["date"].isna()].assign(problem="unparseable date")])
            st.dataframe(show[["stock", "date", "open", "high", "low", "close", "volume", "problem"]],
                         hide_index=True, width="stretch")

    # Price paths of chosen stocks: close as an index, 100 = the stock's first week
    st.subheader("Price paths")
    stocks = sorted(ok["stock"].unique())
    default = (list(r.table["stock"].head(2)) + list(r.table["stock"].tail(2))) if r is not None else stocks[:4]
    chosen = st.multiselect("Stocks (default: the top 2 and bottom 2 of the ranking)", stocks, default=default,
                            max_selections=len(SERIES))
    if chosen:
        p = ok[ok["stock"].isin(chosen)].sort_values("date").copy()
        p["index"] = p["close"] / p.groupby("stock")["close"].transform("first") * 100
        color = alt.Color("stock:N", scale=alt.Scale(domain=chosen, range=SERIES[:len(chosen)]), legend=alt.Legend(orient="top"))
        tip = ["stock", alt.Tooltip("date:T", format="%Y-%m-%d"), alt.Tooltip("close:Q", format=".2f"),
               alt.Tooltip("index:Q", format=".1f")]
        lines = alt.Chart(p).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40, filled=True)).encode(
            x=alt.X("date:T", title=None), y=alt.Y("index:Q", title="close, first week = 100", scale=alt.Scale(zero=False)),
            color=color, tooltip=tip)
        rule = alt.Chart(pd.DataFrame({"d": [pd.Timestamp(week)]})).mark_rule(color=GREY, strokeDash=[4, 4]).encode(x="d:T")
        layers = lines + rule
        if len(chosen) <= 4:                                                    # few series: label line ends directly
            last = p.groupby("stock").tail(1)
            layers += alt.Chart(last).mark_text(align="left", dx=8, fontSize=12).encode(
                x="date:T", y="index:Q", text="stock:N", color=alt.value(GREY))
        st.altair_chart(layers.properties(height=320), width="stretch", theme="streamlit")
        st.caption("Dashed line: the week being ranked.")

    # The market each week: equal-weighted average return of the stocks in the file
    st.subheader("Market return per week")
    mk = (ok.assign(ret=(ok["close"] - ok["open"]) / ok["open"] * 100)
          .groupby("date").agg(mkt_ret=("ret", "mean"), stocks=("stock", "size")).reset_index())
    bars = alt.Chart(mk).mark_bar(color=BLUE, cornerRadiusEnd=4, size=16).encode(
        x=alt.X("date:T", title=None), y=alt.Y("mkt_ret:Q", title="average weekly return (%)"),
        tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), alt.Tooltip("mkt_ret:Q", format=".2f", title="avg return %"), "stocks:Q"])
    st.altair_chart(bars.properties(height=240), width="stretch", theme="streamlit")

    # This week's inputs against what the model saw in training
    st.subheader(f"Model inputs in week {pd.Timestamp(week).date()} vs. the training range")
    st.caption("Each dot is a stock. The grey band is the range seen in training (Q1 2011); "
               "red dots are outside it, so their prediction is an extrapolation.")
    if r is None:
        st.info("No prediction for this week, so there are no model inputs to show.")
        return
    cols = st.columns(3)
    for i, feat in enumerate(FEATS):
        lo, hi = model.train_range[feat]
        d = r.table[["stock", feat]].rename(columns={feat: "value"})
        d["outside"] = ~d["value"].between(lo, hi)
        band = alt.Chart(pd.DataFrame({"lo": [lo], "hi": [hi]})).mark_rect(color=GREY, opacity=0.18).encode(x="lo:Q", x2="hi:Q")
        dots = alt.Chart(d).mark_circle(size=70, opacity=0.85).encode(
            x=alt.X("value:Q", title=FEATURE_NAMES[feat], scale=alt.Scale(zero=False)),
            y=alt.Y("jitter:Q", axis=None, scale=alt.Scale(domain=[-1, 1])),
            color=alt.condition("datum.outside", alt.value(RED), alt.value(BLUE)),
            tooltip=["stock", alt.Tooltip("value:Q", format=".3f"), "outside"]
        ).transform_calculate(jitter="sqrt(-2*log(random()))*cos(2*PI*random()) * 0.3")
        labels = alt.Chart(d[d["outside"]]).mark_text(dy=-12, fontSize=11, color=GREY).encode(x="value:Q", text="stock:N")
        cols[i % 3].altair_chart((band + dots + labels).properties(height=150, title=feat),
                                 width="stretch", theme="streamlit")


def show_monitoring(m, level, level_label, verdict):
    as_of = m.as_of.date()
    st.markdown(f"Can you use the output of week **{as_of}**? Judged from that week and earlier ones only "
                "(pick another week in the sidebar to see what the monitor said then).")
    show_verdict(verdict)

    w = m.weekly
    floor = level - COVERAGE_DROP
    c1, c2, c3 = st.columns(3)
    cov_check = next(c for c in m.checks if c.name == "coverage")
    c1.metric(f"{level_label} coverage, last {WINDOW} weeks", "–" if pd.isna(cov_check.value) else f"{cov_check.value:.1%}",
              f"retrain below {floor:.0%}", delta_color="off")
    drift = next(c for c in m.checks if c.name == "input_drift")
    c2.metric("Stocks out of training range (this week)", "–" if pd.isna(drift.value) else f"{drift.value:.0%}",
              f"retrain above {OOR_SHARE:.0%}", delta_color="off")
    c3.metric("Days since training data", f"{(m.as_of - m.train_end).days}",
              f"next retrain {m.next_scheduled.date()}", delta_color="off")
    if w.empty:
        st.info("No weeks after the training period up to this week, so there is nothing to monitor yet.")
        return

    a, b = st.columns(2)
    if "rolling_coverage" in w and w["rolling_coverage"].notna().any():
        cw = w.dropna(subset=["rolling_coverage"]).assign(judged=lambda d: d["rolling_weeks"] >= WINDOW)
        ref = pd.DataFrame({"y": [level, floor], "what": ["nominal", "retrain below"]})
        chart = (alt.Chart(cw).mark_line(color=BLUE, strokeWidth=2).encode(
                     x=alt.X("date:T", title=None), y=alt.Y("rolling_coverage:Q", title=f"coverage, last {WINDOW} weeks",
                                                          scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")))
                 + alt.Chart(cw).mark_point(filled=True, size=60).encode(
                     x="date:T", y="rolling_coverage:Q",
                     color=alt.condition("datum.judged", alt.value(BLUE), alt.value(GREY)),
                     tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), alt.Tooltip("rolling_coverage:Q", format=".1%"),
                              alt.Tooltip("rolling_weeks:Q", title="weeks in window"),
                              alt.Tooltip("coverage:Q", format=".0%", title="this week alone")])
                 + alt.Chart(ref).mark_rule(strokeDash=[4, 4]).encode(
                     y="y:Q", color=alt.Color("what:N", scale=alt.Scale(domain=["nominal", "retrain below"], range=[GREY, RED]),
                                              legend=alt.Legend(orient="top", title=None)))
                 ).properties(title="Rolling interval coverage", height=260)
        a.altair_chart(chart, width="stretch", theme="streamlit")
        a.caption(f"Shipped (2011) intervals. Grey points: fewer than {WINDOW} weeks in the window, not judged "
                  "(one week's coverage swings by about ±12 pp).")
    else:
        a.info(f"No realized returns (`{TARGET}`) up to this week, so coverage cannot be checked; "
               "input drift and the schedule still are.")
    drift_chart = (alt.Chart(w).mark_bar(cornerRadiusEnd=4, size=14).encode(
                       x=alt.X("date:T", title=None),
                       y=alt.Y("share_oor:Q", title="share of stocks", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                       color=alt.condition("datum.extreme_market", alt.value("#eda100"), alt.value(BLUE)),
                       tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), alt.Tooltip("share_oor:Q", format=".0%"),
                                alt.Tooltip("extreme_market:N", title="market return out of range")])
                   + alt.Chart(pd.DataFrame({"y": [OOR_SHARE]})).mark_rule(color=RED, strokeDash=[4, 4]).encode(y="y:Q")
                   ).properties(title="Stocks with a feature outside the training range", height=260)
    b.altair_chart(drift_chart, width="stretch", theme="streamlit")
    b.caption("Stock-level features only. Amber bars: the market return itself was outside the training range that week "
              "(one number for all stocks, so it is shown, not counted). Red line: retrain above it.")

    with st.expander("For the maintainer: retraining rule", expanded=False):
        if m.status == OK:
            st.success(f"No retraining needed. Next scheduled retrain: {m.next_scheduled.date()}.")
        elif m.status == NOW:
            st.error("Retrain now: " + "; ".join(c.text for c in m.checks if c.triggered) + ".")
        else:
            st.warning(f"Scheduled retrain due (every {RETRAIN_EVERY_DAYS} days; the triggers are fine).")
        for c in m.checks:
            st.markdown(f"{'🔴' if c.triggered else '🟢'} {c.text}")
        since = (m.as_of - pd.Timedelta(weeks=12)).date()
        st.markdown(f"""
Retraining is the maintainer's job, not the user's. Train on the **last 13 weeks** (one quarter, the size of the original training set), with realized returns:
```
dow-ranker train my_weeks.csv --since {since} --until {as_of} --out models/ridge_{as_of}.json
dow-ranker predict my_weeks.csv --model models/ridge_{as_of}.json
```
`train` re-runs the same walk-forward tuning of alpha and recomputes the training ranges and interval widths. Keep the
old JSON. With a signal this weak, retraining mainly recalibrates the intervals and ranges; expect alpha to stay large.
""")


read_error = None
with st.sidebar:
    st.header("Input")
    sources = ((["UCI Dow Jones 2011"] if UCI.exists() else []) + [k for k, v in EXAMPLES.items() if v.exists()]
               + ["Upload CSV", "Synthetic example"])
    source = st.radio("Data", sources)
    if source in EXAMPLES:
        raw = pd.read_csv(EXAMPLES[source])
    elif source == "Upload CSV":
        up = st.file_uploader("CSV with weekly rows (format: see the main page)", type="csv")
        try:
            raw = read_upload(up) if up else None
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, IndexError) as e:
            raw, read_error = None, f"Could not read the file as CSV: {e}"
    elif source == "Synthetic example":
        raw = make_panel(seed=st.number_input("Seed", 0, 999, 0))
    else:
        raw = pd.read_csv(UCI)
    if not UCI.exists():
        st.caption("Run `dow-ranker fetch-data` to add the UCI data set.")
    level_label = st.radio("Interval", list(LEVELS), index=1, horizontal=True)
    level = LEVELS[level_label]

st.title("Dow weekly ranker")
st.caption("Predicted return of each stock next week **relative to the average stock** (percentage points), "
           "from the ridge model of Part II. The signal is weak: read the intervals before the ranks.")

if source == "Upload CSV":
    show_upload_help(expanded=raw is None)
if raw is None:
    if read_error:
        st.error(read_error)
    else:
        st.info("Upload a CSV in the sidebar; the format is described above.")
    st.stop()
try:
    df = clean(raw)
except InputError as e:
    st.error(str(e))
    st.stop()

weeks = sorted(df["date"].dropna().unique(), reverse=True)
with st.sidebar:
    week = st.selectbox("Week to rank", weeks[:-1] or weeks, format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))
    try:
        recal, recal_error = recalibrate(df, model=model, as_of=week), None
    except InputError as e:
        recal, recal_error = None, str(e)
    use_recal = st.toggle("Recalibrate intervals on my data", value=False, disabled=recal is None,
                          help=f"Takes the interval widths from the model's errors on your last {RECAL_MIN_WEEKS}-"
                               f"{RECAL_WEEKS} weeks with realized returns instead of from 2011. The predictions do not "
                               "change. " + (recal_error or ""))
    if use_recal and recal is not None:
        old, new = model.halfwidth[f"{level:g}"], recal.halfwidth[f"{level:g}"]
        st.caption(f"{level_label} interval: ±{old:.2f} pp (2011) -> ±{new:.2f} pp (your weeks "
                   f"{recal.info['recalibrated']['weeks'][0]} to {recal.info['recalibrated']['weeks'][1]}).")
active = recal if use_recal and recal is not None else model

try:
    mon = monitor(df, model=model, level=level, as_of=week)           # shipped intervals, as the maintainer sees them
    verdict = user_verdict(mon, recalibrated=active is not model, can_recalibrate=recal is not None)
except InputError:
    mon = verdict = None
try:
    r, refused = predict_week(df, week=week, model=active, level=level), None
except InputError as e:
    r, refused = None, str(e)

tab_rank, tab_data, tab_track, tab_mon, tab_model = st.tabs(["Ranking", "Data", "Track record", "Monitoring",
                                                              "About the model"])

with tab_rank:
    if refused:
        st.error(f"No prediction for this week: {refused}")
    else:
        show_ranking(r, level_label, verdict, recalibrated=active is not model)

with tab_data:
    show_data(df, r, week)

with tab_track:
    if TARGET not in df:
        st.info(f"The input has no realized returns (`{TARGET}`), so there is nothing to check yet.")
    else:
        start = pd.Timestamp(model.info["train_weeks"][1]) + pd.Timedelta(days=1)
        bt = backtest(df, model=active, level=level, start=start)
        if bt.empty:
            st.info("No weeks after the training period with a realized next-week return.")
        else:
            s = backtest_summary(bt)
            tot, weekly = s.loc["all weeks"], s.drop("all weeks").reset_index(names="week")
            st.markdown(f"Weeks after the training data ({weekly['week'].iloc[0]} to {weekly['week'].iloc[-1]}), "
                        "never used for fitting or tuning.")
            if active is not model:
                st.warning("Recalibrated intervals: their widths come from some of these weeks, so the coverage on "
                           "those weeks is close to nominal by construction. Only later weeks are a real check.")
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{level_label} interval coverage", f"{tot['coverage']:.1%}", f"{tot['coverage'] - level:+.1%} vs nominal",
                      delta_color="off")
            c2.metric("MAE model / predict 0", f"{tot['mae']:.3f} / {tot['mae_baseline_zero']:.3f}")
            c3.metric("Mean weekly Spearman IC", f"{tot['ic']:.3f}")
            cov = (alt.Chart(weekly).mark_line(color=BLUE, strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True))
                   .encode(x=alt.X("week:T", title=None), y=alt.Y("coverage:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                           tooltip=["week:T", alt.Tooltip("coverage:Q", format=".0%"), "n:Q"])
                   + alt.Chart(pd.DataFrame({"y": [level]})).mark_rule(color=GREY, strokeDash=[4, 4]).encode(y="y:Q")
                   ).properties(title=f"Weekly coverage (dashed: nominal {level_label})", height=260)
            ic = (alt.Chart(weekly).mark_bar(color=BLUE, cornerRadiusEnd=4, size=14)
                  .encode(x=alt.X("week:T", title=None), y=alt.Y("ic:Q", title="Spearman IC"),
                          tooltip=["week:T", alt.Tooltip("ic:Q", format=".3f")])
                  ).properties(title="Weekly rank correlation of prediction and outcome", height=260)
            a, b = st.columns(2)
            a.altair_chart(cov, width="stretch", theme="streamlit")
            b.altair_chart(ic, width="stretch", theme="streamlit")
            st.dataframe(s.round(3), width="stretch")

with tab_mon:
    if mon is None:
        st.info("Nothing to monitor for this week.")
    else:
        show_monitoring(mon, level, level_label, verdict)

with tab_model:
    i = model.info
    recal_text = "" if active is model else (
        f"**Now recalibrated on your data** ({active.info['recalibrated']['n_residuals']} errors, weeks "
        f"{active.info['recalibrated']['weeks'][0]} to {active.info['recalibrated']['weeks'][1]}): "
        + ", ".join(f"{float(k):.0%}: ±{v:.2f} pp" for k, v in active.halfwidth.items()) + ".")
    st.markdown(f"""
**Model.** Standardized features -> ridge regression (alpha = {model.alpha:.0f}, tuned by walk-forward CV on MAE),
trained on {i['n_train_rows']} stock-weeks, {i['train_weeks'][0]} to {i['train_weeks'][1]} (UCI Dow Jones Index, Q1 2011).
CV MAE {i['cv_mae']:.4f} vs {i['cv_mae_baseline_zero']:.4f} for predicting 0; mean weekly IC {i['cv_mean_ic']:.3f}.

**Intervals.** Split-conformal: the {i['n_oof_residuals']} out-of-fold absolute residuals give a half-width of
{', '.join(f"{float(k):.0%}: ±{v:.2f} pp" for k, v in model.halfwidth.items())}. Same width for every stock.
{recal_text}

**Inputs.** Only weeks t and t-1 of each stock; columns about the future are ignored. The model does not use tickers,
so any set of stocks works; the "market" is the average of the stocks you give it.

**Use it for:** a weak, honest signal and a check of it. **Not for:** trading decisions; the held-out error equals that of predicting 0.
""")
    st.dataframe(pd.DataFrame({"coefficient (pp per 1 sd)": model.coef, "training min": [model.train_range[f][0] for f in model.features],
                               "training max": [model.train_range[f][1] for f in model.features]}, index=model.features).round(4))
    st.caption(f"Trained with {', '.join(f'{k} {v}' for k, v in i['versions'].items())}.")
