"""Rank Dow Jones stocks by predicted next-week return relative to the average stock (Part II ridge model)."""
from .data import InputError, clean, fetch_uci, read
from .features import FEATS, build_features
from .model import RidgeModel, fit
from .monitor import MonitorResult, monitor
from .predict import Prediction, backtest, backtest_summary, predict_week

__version__ = "0.1.0"
__all__ = ["InputError", "clean", "fetch_uci", "read", "FEATS", "build_features", "RidgeModel", "fit",
           "Prediction", "backtest", "backtest_summary", "predict_week", "MonitorResult", "monitor"]
