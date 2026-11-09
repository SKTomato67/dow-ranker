# dow-ranker

Ranks the stocks of one week by their **predicted return next week relative to the average stock**, with
an honest prediction interval and checks on the input. It packages the ridge model from my 4EK481/4EK666
semestral project (Dow Jones Index, UCI) so it can be run again on new weeks, by someone else.

**For whom:** a student or analyst who wants a weekly relative-return signal *and* an honest measure of how
weak it is. The project found almost no predictable signal: on the held-out weeks the model's error equals
that of predicting 0. The tool shows this openly: the intervals (±2.2 pp at 80 %) are about 20 times wider
than the spread of the predictions. It is not a trading tool.

**Try it online:** [https://dow-ranker-ditvrzdqyxmtn5rphkj4dx.streamlit.app/](https://dow-ranker-ditvrzdqyxmtn5rphkj4dx.streamlit.app/) (Streamlit Community Cloud; no installation needed).

## Install

Python 3.10 or newer.

```
pip install "dow-ranker[app] @ git+https://github.com/SKTomato67/dow-ranker@v0.1.0"
```

or from a clone: `pip install -e ".[app,dev]"`. Then download the data (UCI, CC BY 4.0):

```
dow-ranker fetch-data            # -> data/dow_jones_index.data
```

## Use

**Command line**

```
dow-ranker predict data/dow_jones_index.data --week 2011-05-13          # rank one week (default: latest)
dow-ranker predict my_weeks.csv --level 0.9 --out ranking.csv
dow-ranker backtest data/dow_jones_index.data --start 2011-04-01        # coverage, MAE, IC per week
dow-ranker monitor new_data.csv                                         # retraining rule; exit code 1 = retrain now
dow-ranker train new_data.csv --since 2026-07-17 --until 2026-10-09 --out models/ridge_2026Q3.json
```

**Python**

```python
import dow_ranker as dr

r = dr.predict_week("data/dow_jones_index.data", week="2011-05-13", level=0.8)
r.table        # rank, stock, pred_rel_return, lower, upper, out_of_range, the 6 features
r.warnings     # anything the user should know (excluded stocks, extrapolation, stale model, ...)
```

**App**

Online at [https://dow-ranker-ditvrzdqyxmtn5rphkj4dx.streamlit.app/](https://dow-ranker-ditvrzdqyxmtn5rphkj4dx.streamlit.app/), or locally:

```
streamlit run app/streamlit_app.py
```

Pick the data (UCI, upload a CSV, or a synthetic example), the week and the interval level. Tabs: the ranking
with intervals, the data, the track record on weeks after the training period, monitoring (judged as of the selected week) and a description of the model.

**If the app says the output is not reliable.** Every tab starts with a verdict on the selected week (usable / use
with caution / not reliable) and what you can do about it; you never need to retrain:

- *Intervals too narrow for your data* (coverage check failed): switch on **Recalibrate intervals on my data** in
  the sidebar. The interval widths are then taken from the model's errors on your last 8-13 weeks with realized
  returns (split conformal, as in training); the predictions do not change. Without enough weeks, read the 90 %
  interval as roughly an 80 % one.
- *Stocks outside the training range*: switch on **Only stocks inside the training range** on the Ranking tab.
- *Model older than a quarter*: use the output with more caution; retraining is the maintainer's job (below).

The input is one row per stock and week with `stock, date, open, high, low, close, volume`, and must contain the
week before the one you rank (see [data/README.md](data/README.md)). `examples/quickstart.ipynb` walks through it.

## What the model is

Six features known at the Friday close of week *t*, from weeks *t* and *t−1* only: this week's return relative to
the average stock, the average (market) return, last week's return, the high-low range, log volume ratio to last
week, and log dollar volume. They are standardized, then fed to a ridge regression (alpha = 1421, tuned by
walk-forward CV on MAE) trained on Q1 2011 (360 stock-weeks). The model does not use tickers, so changes in which
stocks are in the Dow do not break it: the "market" is the average of whatever stocks you give it.

Intervals are split conformal: the 80 % interval is ± the 80 % quantile of the 210 out-of-fold absolute
residuals from the training folds. Every stock gets the same width.

## How it is tested

`pytest` (73 tests). GitHub Actions runs them on Linux and Windows, Python 3.10 and 3.13, after every push.

| Requirement | Test | Result |
|---|---|---|
| Same answers as the Part II model | `scripts/notebook_reference.py` runs the notebook's own cells and saves its predictions; the package must match them for all 750 stock-weeks, and retraining must reproduce the shipped model | max difference 3·10⁻⁷ pp (the notebook used UCI's return rounded to 5 decimals); same alpha and coefficients |
| Uses only what is known | scrambling the future columns, changing or deleting all later weeks, or changing weeks before *t−1* leaves the prediction unchanged | unchanged |
| Bad inputs | missing column, empty file, duplicates, week not in the data, < 5 stocks → refused; missing/unparseable/negative values, high < low, no previous week → that stock excluded with a reason; unknown tickers, values outside the training range, non-Friday weeks, data long after training → warning | all as specified |
| Reproducible | same input → same output; row order irrelevant; model stored as JSON (no pickle) with library versions; fresh install in a clean venv | identical |
| Retraining rule | `monitor` must flag inflated outcome volatility, 2026-size dollar volume and an overdue schedule; must not flag held-out Q2, a single extreme market week, or coverage before 8 weeks are known; must ignore weeks after the as-of week | as specified; Q2 never flags |
| User without retraining | verdict per week (usable / caution / not reliable) with the fixes; recalibrated widths equal the conformal quantile of the user's recent errors, use only earlier weeks, leave predictions unchanged, need 8 weeks of outcomes; the extrapolation filter ignores the market return | on 4x-volatile Q2 outcomes: shipped intervals cover < 50 % of later weeks, recalibrated ones about 80 % |
| Honest uncertainty | coverage on the 13 held-out Q2 weeks (390 rows, never used in training or tuning) must be within 5 pp of nominal | 50 %: 46.4 %, 80 %: 76.4 %, 90 %: 86.2 % |

The held-out MAE is 1.484 against 1.481 for predicting 0, and the mean weekly Spearman IC is 0.04. A test
checks that the tool never looks better than this.

## Monitoring and retraining

This section is for the maintainer; users of the app never retrain (see *If the app says the output is not
reliable* above). Check every week after the Friday close (`dow-ranker monitor`, or the Monitoring tab). **Retrain once a quarter**
(91 days after the training data ends), **and earlier** if either trigger fires:

| Check | Retrain when | Why this threshold |
|---|---|---|
| Interval coverage, last 8 weeks with known outcomes | below nominal − 10 pp (80 % → 70 %) | one week's coverage swings by ±12 pp (sd in Q2), so an 8-week average has sd ≈ 4 pp; 70 % is ≈ 2.4 sd low. Judged only once 8 weeks are known |
| Input drift: stocks in the latest week with a stock-level feature outside the training range | more than 50 % | the model would be extrapolating for most stocks. The market return is excluded: it is one number for all stocks, so one extreme market week would flag everything |
| Schedule | 91 days → retrain due; 182 days → retrain now | Q1 → Q2 is the only horizon the held-out test supports |

On the held-out Q2 the rule never fires; the first full window (week of 20 May 2011) sits exactly at 70.0 %
(168 of 240), the weakest point of the equal-width intervals.

To retrain: `dow-ranker train` on the **last 13 weeks** (`--since`, `--until`; at least 7 weeks) re-runs the same
tuning and saves a new JSON with its own training ranges and interval widths, which you pass to `--model`. Keep the
old JSON. With a signal this weak, retraining mainly recalibrates the intervals and ranges.

## Layout

```
src/dow_ranker/   data.py (reading, UCI download) features.py model.py (fit, JSON) predict.py (checks, ranking, backtest)
                  monitor.py (retraining rule)
                  cli.py synthetic.py models/ridge_v1.json
app/              streamlit_app.py
tests/            one file per requirement; reference/ = notebook predictions
scripts/          notebook_reference.py (re-creates tests/reference from the course notebook)
```

Data: Brown, M. S., Pelosi, M. & Dirska, H. (2013). Dow Jones Index [Dataset]. UCI Machine Learning Repository,
https://doi.org/10.24432/C5788V. Licence of the code: MIT.
