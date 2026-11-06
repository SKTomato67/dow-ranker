"""4. Same input, same output; the model file is portable and records how it was made."""
import pandas as pd

from dow_ranker import RidgeModel, predict_week
from dow_ranker.cli import main


def test_same_input_same_output(panel):
    pd.testing.assert_frame_equal(predict_week(panel).table, predict_week(panel).table)


def test_row_order_does_not_matter(panel):
    shuffled = panel.sample(frac=1, random_state=1)
    pd.testing.assert_frame_equal(predict_week(panel).table, predict_week(shuffled).table)


def test_model_json_roundtrip(tmp_path, panel, model):
    model.save(tmp_path / "m.json")
    again = RidgeModel.load(tmp_path / "m.json")
    assert again == model
    pd.testing.assert_frame_equal(predict_week(panel, model=model).table, predict_week(panel, model=again).table)


def test_model_records_versions_and_training_window(model):
    assert set(model.info["versions"]) >= {"python", "numpy", "pandas", "scikit-learn"}
    assert model.info["train_weeks"] == ["2011-01-07", "2011-03-25"]
    assert model.info["n_train_rows"] == 360


def test_cli_predict_and_errors(tmp_path, panel, capsys):
    panel.to_csv(tmp_path / "week.csv", index=False)
    assert main(["predict", str(tmp_path / "week.csv"), "--out", str(tmp_path / "out.csv")]) == 0
    assert len(pd.read_csv(tmp_path / "out.csv")) == 30
    panel.drop(columns="volume").to_csv(tmp_path / "bad.csv", index=False)
    assert main(["predict", str(tmp_path / "bad.csv")]) == 2
    assert "volume" in capsys.readouterr().err
