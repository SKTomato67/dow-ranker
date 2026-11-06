"""Export the Part II ridge predictions straight from the course notebook.

Replays only the notebook cells that the final ridge depends on (data loading, week 0, features,
training set, section 13.2) in one namespace, then writes the notebook's own predictions for every
stock-week. tests/test_reference.py checks the package against this file.

Usage: python scripts/notebook_reference.py <folder with Koudelka_semestral_ntbk.ipynb and data/>
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

CELLS = [2, 4, 31, 32, 54]          # imports, load + clean, week 0, FEATS/build_features/train, CV folds
RIDGE_CELL = 59                     # 13.2 Ridge on Y_rel (needs `alphas` from cell 56, which also runs the slow RF search)

project = Path(sys.argv[1]).resolve()
out = Path(__file__).resolve().parents[1] / "tests" / "reference"
nb = json.loads((project / "Koudelka_semestral_ntbk.ipynb").read_text(encoding="utf-8"))
src = lambda i: "".join(nb["cells"][i]["source"])
assert "13.2 Ridge on Y_rel" in src(RIDGE_CELL - 1), "notebook cells moved: update CELLS / RIDGE_CELL"

os.chdir(project)
ns = {}
for i in CELLS:
    exec(src(i), ns)
exec("alphas = np.logspace(-2, 6, 60)", ns)     # the line from cell 56
exec(src(RIDGE_CELL), ns)

m, ridge = ns["m"], ns["ridge_best"]
ref = m[["stock", "date", "quarter"]].copy()
ref["pred_notebook"] = ridge.predict(m[ns["FEATS"]])
ref.to_csv(out / "notebook_predictions.csv", index=False)
coef = {"alpha": float(ns["best_alpha"]), "intercept": float(ridge[-1].intercept_),
        "coef": dict(zip(ns["FEATS"], map(float, ridge[-1].coef_))),
        "cv_mae": float(-ns["ridge_cv"].best_score_)}
(out / "notebook_model.json").write_text(json.dumps(coef, indent=2))
print(f"wrote {len(ref)} predictions and the coefficients to {out}")
