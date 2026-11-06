"""Dow weekly ranker: streamlit run app/streamlit_app.py"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from dow_ranker import InputError, RidgeModel, backtest, backtest_summary, clean, predict_week
from dow_ranker.data import TARGET
from dow_ranker.synthetic import make_panel

UCI = Path(__file__).resolve().parents[1] / "data" / "dow_jones_index.data"
BLUE, GREY = "#2a78d6", "#8a8984"
LEVELS = {"50 %": 0.5, "80 %": 0.8, "90 %": 0.9}

st.set_page_config(page_title="Dow weekly ranker", layout="wide")
model = RidgeModel.load()

with st.sidebar:
    st.header("Input")
    sources = (["UCI Dow Jones 2011"] if UCI.exists() else []) + ["Upload CSV", "Synthetic example"]
    source = st.radio("Data", sources)
    if source == "Upload CSV":
        up = st.file_uploader("Weekly rows: stock, date, open, high, low, close, volume", type="csv")
        raw = pd.read_csv(up) if up else None
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

if raw is None:
    st.info("Upload a CSV with at least two consecutive weeks per stock.")
    st.stop()
try:
    df = clean(raw)
except InputError as e:
    st.error(str(e))
    st.stop()

weeks = sorted(df["date"].dropna().unique(), reverse=True)
with st.sidebar:
    week = st.selectbox("Week to rank", weeks[:-1] or weeks, format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))

tab_rank, tab_track, tab_model = st.tabs(["Ranking", "Track record", "About the model"])

with tab_rank:
    try:
        r = predict_week(df, week=week, model=model, level=level)
    except InputError as e:
        st.error(f"No prediction: {e}")
        st.stop()
    for w in r.warnings:
        st.warning(w)
    t = r.table
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
    st.dataframe(t.round(4), hide_index=True, width="stretch")
    st.download_button("Download table (CSV)", t.to_csv(index=False), f"ranking_{r.week.date()}.csv", "text/csv")

with tab_track:
    if TARGET not in df:
        st.info(f"The input has no realized returns (`{TARGET}`), so there is nothing to check yet.")
    else:
        start = pd.Timestamp(model.info["train_weeks"][1]) + pd.Timedelta(days=1)
        bt = backtest(df, model=model, level=level, start=start)
        if bt.empty:
            st.info("No weeks after the training period with a realized next-week return.")
        else:
            s = backtest_summary(bt)
            tot, weekly = s.loc["all weeks"], s.drop("all weeks").reset_index(names="week")
            st.markdown(f"Weeks after the training data ({weekly['week'].iloc[0]} to {weekly['week'].iloc[-1]}), "
                        "never used for fitting or tuning.")
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

with tab_model:
    i = model.info
    st.markdown(f"""
**Model.** Standardized features -> ridge regression (alpha = {model.alpha:.0f}, tuned by walk-forward CV on MAE),
trained on {i['n_train_rows']} stock-weeks, {i['train_weeks'][0]} to {i['train_weeks'][1]} (UCI Dow Jones Index, Q1 2011).
CV MAE {i['cv_mae']:.4f} vs {i['cv_mae_baseline_zero']:.4f} for predicting 0; mean weekly IC {i['cv_mean_ic']:.3f}.

**Intervals.** Split-conformal: the {i['n_oof_residuals']} out-of-fold absolute residuals give a half-width of
{', '.join(f"{float(k):.0%}: ±{v:.2f} pp" for k, v in model.halfwidth.items())}. Same width for every stock.

**Inputs.** Only weeks t and t-1 of each stock; columns about the future are ignored. The model does not use tickers,
so any set of stocks works; the "market" is the average of the stocks you give it.

**Use it for:** a weak, honest signal and a check of it. **Not for:** trading decisions; the held-out error equals that of predicting 0.
""")
    st.dataframe(pd.DataFrame({"coefficient (pp per 1 sd)": model.coef, "training min": [model.train_range[f][0] for f in model.features],
                               "training max": [model.train_range[f][1] for f in model.features]}, index=model.features).round(4))
    st.caption(f"Trained with {', '.join(f'{k} {v}' for k, v in i['versions'].items())}.")
