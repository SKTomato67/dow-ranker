"""The Part II ridge model: fitting (same procedure as notebook section 13.2) and a portable JSON form."""
import json
import math
import platform
from dataclasses import asdict, dataclass, field
from importlib import resources

import numpy as np
import pandas as pd

from .features import FEATS

ALPHAS = np.logspace(-2, 6, 60)    # notebook grid
MIN_TRAIN_WEEKS = 5                # first walk-forward fold trains on 5 weeks
LEVELS = (0.5, 0.8, 0.9)
DEFAULT_MODEL = "ridge_v1.json"


@dataclass
class RidgeModel:
    """Standardize -> ridge on Y_rel, stored as plain numbers so it predicts without scikit-learn."""
    features: list
    mean: list
    scale: list
    coef: list
    intercept: float
    alpha: float
    train_range: dict               # feature -> [min, max] in the training data
    halfwidth: dict                 # interval level (str) -> conformal half-width in pp
    info: dict = field(default_factory=dict)

    def predict(self, X):
        Z = (np.asarray(X[self.features], dtype=float) - np.asarray(self.mean)) / np.asarray(self.scale)
        return Z @ np.asarray(self.coef) + self.intercept

    def interval(self, pred, level=0.8):
        key = f"{level:g}"
        if key not in self.halfwidth:
            raise ValueError(f"interval level {level} not available; choose one of {sorted(self.halfwidth)}")
        h = self.halfwidth[key]
        return pred - h, pred + h

    def out_of_range(self, X):
        """Boolean frame: True where a feature is outside the range seen in training."""
        return pd.DataFrame({f: ~X[f].between(*self.train_range[f]) for f in self.features}, index=X.index)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path=None):
        """Loads a saved model; without a path, the model shipped with the package."""
        if path is None:
            text = resources.files("dow_ranker.models").joinpath(DEFAULT_MODEL).read_text(encoding="utf-8")
        else:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        return cls(**json.loads(text))


def walk_forward_folds(dates):
    """Expanding-window folds over weeks: train on all weeks before k, validate on week k."""
    weeks = np.sort(dates.unique())
    return [(np.where(dates.isin(weeks[:k]))[0], np.where(dates == weeks[k])[0])
            for k in range(MIN_TRAIN_WEEKS, len(weeks))]


def conformal_halfwidth(abs_resid, level):
    """Split-conformal quantile: the ceil((n+1) level)-th smallest absolute residual."""
    n = len(abs_resid)
    return float(np.sort(abs_resid)[min(n, math.ceil((n + 1) * level)) - 1])


def fit(train, alphas=ALPHAS, levels=LEVELS, info=None):
    """Tunes alpha by walk-forward CV (MAE) and refits on all of train (a build_features frame with Y_rel).

    Interval half-widths come from the out-of-fold residuals of the tuned alpha (split conformal).
    """
    import sklearn
    from scipy import stats
    from sklearn.base import clone
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train = train.dropna(subset=["Y_rel"]).sort_values(["date", "stock"]).reset_index(drop=True)
    X, y = train[FEATS], train["Y_rel"]
    folds = walk_forward_folds(train["date"])
    if len(folds) < 2:
        raise ValueError(f"need at least {MIN_TRAIN_WEEKS + 2} weeks to tune alpha, got {train['date'].nunique()}")
    cv = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": alphas},
                      cv=folds, scoring="neg_mean_absolute_error").fit(X, y)
    best = cv.best_estimator_
    oof, ic = np.full(len(y), np.nan), []
    for t, v in folds:
        oof[v] = clone(best).fit(X.iloc[t], y.iloc[t]).predict(X.iloc[v])
        ic.append(stats.spearmanr(oof[v], y.iloc[v])[0])
    abs_resid = np.abs(y - oof)[~np.isnan(oof)].to_numpy()
    scaler, ridge = best[0], best[-1]
    info = dict(info or {})
    info.update({
        "target": "Y_rel = next week's return minus the mean of the stocks in that week (pp)",
        "train_weeks": [str(train["date"].min().date()), str(train["date"].max().date())],
        "n_train_rows": len(train), "train_stocks": sorted(train["stock"].unique()),
        "cv_mae": float(-cv.best_score_), "cv_mae_baseline_zero": float(np.mean([np.abs(y.iloc[v]).mean() for _, v in folds])),
        "cv_mean_ic": float(np.mean(ic)), "n_oof_residuals": len(abs_resid),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__},
    })
    return RidgeModel(
        features=list(FEATS), mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), coef=ridge.coef_.tolist(),
        intercept=float(ridge.intercept_), alpha=float(cv.best_params_["ridge__alpha"]),
        train_range={f: [float(X[f].min()), float(X[f].max())] for f in FEATS},
        halfwidth={f"{lv:g}": conformal_halfwidth(abs_resid, lv) for lv in levels}, info=info)
