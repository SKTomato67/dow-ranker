"""1. The package gives the same answers as the Part II model in the notebook."""
import json

import numpy as np
import pandas as pd
import pytest
from conftest import REF, TRAIN_END

from dow_ranker import FEATS, build_features, fit, predict_week

TOL = 1e-6                         # pp; the notebook used UCI's percent_change_price, rounded to 5 decimals


def test_shipped_coefficients_match_notebook(model):
    ref = json.loads((REF / "notebook_model.json").read_text())
    assert model.features == FEATS
    assert model.alpha == pytest.approx(ref["alpha"], rel=1e-12)
    np.testing.assert_allclose(model.coef, [ref["coef"][f] for f in FEATS], atol=1e-7)
    assert model.info["cv_mae"] == pytest.approx(ref["cv_mae"], abs=1e-6)


def test_predictions_match_notebook(uci, week0, model):
    ref = pd.read_csv(REF / "notebook_predictions.csv", parse_dates=["date"])
    f = build_features(uci, week0)             # without week 0 the first week has no lags and is skipped
    got = ref.merge(f.assign(pred=model.predict(f))[["stock", "date", "pred"]], on=["stock", "date"])
    assert len(got) == len(ref) - (0 if week0 is not None else 30)
    np.testing.assert_allclose(got["pred"], got["pred_notebook"], atol=TOL)


@pytest.mark.parametrize("week", ["2011-04-08", "2011-05-20", "2011-06-24"])
def test_predict_week_matches_notebook(uci, model, week):
    ref = pd.read_csv(REF / "notebook_predictions.csv", parse_dates=["date"])
    ref = ref[ref["date"] == week].set_index("stock")["pred_notebook"]
    table = predict_week(uci, week=week, model=model).table.set_index("stock")
    np.testing.assert_allclose(table.loc[ref.index, "pred_rel_return"], ref, atol=TOL)
    assert list(table.index) == list(ref.sort_values(ascending=False).index)   # same ranking


def test_refit_reproduces_shipped_model(uci, week0, model):
    if week0 is None:
        pytest.skip("data/week0_yahoo.csv (Yahoo data, not redistributed) is needed to refit the exact Part II model")
    m = fit(build_features(uci[uci["date"] <= TRAIN_END], week0))
    assert m.alpha == model.alpha
    np.testing.assert_allclose(m.coef, model.coef, atol=1e-10)
    np.testing.assert_allclose(list(m.halfwidth.values()), list(model.halfwidth.values()), atol=1e-10)
