from pathlib import Path

import pandas as pd
import pytest

from dow_ranker import RidgeModel, fetch_uci, read
from dow_ranker.synthetic import make_panel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REF = Path(__file__).resolve().parent / "reference"
TRAIN_END = "2011-03-25"           # last week of Q1 = training data of the shipped model
HELD_OUT = "2011-04-01"            # Q2: never used for training or tuning


@pytest.fixture(scope="session")
def uci_path():
    path = DATA / "dow_jones_index.data"
    if not path.exists():
        try:
            fetch_uci(DATA)
        except OSError as e:
            pytest.skip(f"UCI data not available and download failed: {e}")
    return path


@pytest.fixture(scope="session")
def uci(uci_path):
    return read(uci_path)


@pytest.fixture(scope="session")
def week0():
    path = DATA / "week0_yahoo.csv"
    return pd.read_csv(path, index_col="stock") if path.exists() else None


@pytest.fixture(scope="session")
def model():
    return RidgeModel.load()


@pytest.fixture
def panel():
    return make_panel()
