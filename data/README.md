# Data

Nothing in this folder is committed. Download the data with

```
dow-ranker fetch-data
```

| File | Source | Licence | Needed for |
|---|---|---|---|
| `dow_jones_index.data`, `dow_jones_index.names` | [UCI Machine Learning Repository: Dow Jones Index](https://archive.ics.uci.edu/dataset/312/dow+jones+index) (Brown, Pelosi & Dirska, 2013), 750 rows = 30 stocks x 25 weeks, Jan-Jun 2011 | CC BY 4.0 | the app's default data, backtests, most tests |
| `week0_yahoo.csv` (optional) | Week 27-31 Dec 2010 from Yahoo Finance, built by the course notebook (section 11) | Yahoo terms, not redistributed | only `test_refit_reproduces_shipped_model`, which retrains the exact Part II model; skipped without it |

## Your own data

One row per stock and week, with at least the week before the week you want to rank:

| column | meaning |
|---|---|
| `stock` | ticker (any, the model does not use it) |
| `date` | last trading day of the week, `2011-05-13` or `5/13/2011` |
| `open`, `high`, `low`, `close` | weekly prices; a `$` sign is allowed |
| `volume` | shares traded in the week |
| `percent_change_next_weeks_price` | optional: realized next-week return in %, only for `dow-ranker backtest` |
