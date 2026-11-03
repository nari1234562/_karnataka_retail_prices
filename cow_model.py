"""
cow_model.py - separate COW forecasting method ("Cow Ensemble v2") for Karnataka milk price forecasting
========================================================================================================
Equal-weight average of the predicted CHANGE (target = Rate(t+h) - Rate(t), one model per horizon h)
of THREE learners:
  1. Cow HGB weighted    HistGradientBoosting on the LEAN input list, rows weighted by recency (half-life 12 months)
  2. Cow HGB             the same learner, all rows weighted equally
  3. Cow Ridge momentum  Ridge on 14 momentum / mean-reversion / season / flag inputs only

Cow Ensemble v2 = rate(t) + mean of the three predicted changes.
"""
from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

METHOD = "Cow Ensemble v2"                                    # the cow headline / default method
MEMBERS = ["Cow HGB weighted", "Cow HGB", "Cow Ridge momentum"]
COW_COLUMNS = [METHOD] + MEMBERS                              # columns added to the step-2 detail table for COW only
MIN_TRAIN_ROWS = 25
HALF_LIFE_MONTHS = 12.0                                       # recency weight: a row 12 months older counts half

LEAN: List[str] = [
    "diff_1", "diff_3", "diff_6", "diff_mean6", "diff_std6", "accel", "dev_ma3", "dev_ma6", "dev_ma12", "yoy", "range12_pos",
    "rel_mean", "rel_ch3", "state_ch3", "tgt_month_sin", "tgt_month_cos", "tgt_summer", "smp_ch3", "smp_ch6", "ghee_ch3",
    "feed_ch3", "maize_ch3", "soy_meal_ch3", "cpi_ch3", "fat_t", "snf_t", "temp_max_t", "rain_3m"
]
MOMENTUM: List[str] = [
    "diff_1", "diff_3", "diff_6", "diff_mean6", "accel", "state_ch3", "dev_ma3", "dev_ma6", "dev_ma12", "smp_ch3",
    "tgt_month_sin", "tgt_month_cos"
]


def _hgb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=42)


def _ridge():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=30.0))


def _recency_weights(target_month: pd.Series, t: pd.Timestamp) -> np.ndarray:
    age_months = (t - pd.to_datetime(target_month)).dt.days.to_numpy() / 30.4
    return np.exp(-np.log(2.0) * age_months / HALF_LIFE_MONTHS)


def cow_columns(table: pd.DataFrame, t, districts: List[str], H: int, scenarios: Dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Forecasts of the cow model standing at origin ``t``. One row per (district, horizon); columns = COW_COLUMNS
    (+ '<column> | <scenario>' for every weather scenario). ``table`` = step-1 training table of COW. Only rows whose
    target month is <= t are used for fitting."""
    t = pd.Timestamp(t)
    scenarios = scenarios or {}
    parts = []
    for h in range(1, H + 1):
        th = table[table["horizon"] == h]
        train = th[(th["target_month"] <= t) & th["target"].notna()]
        test = th[(th["origin"] == t) & th["district"].isin(districts)]
        if test.empty or len(train) < MIN_TRAIN_ROWS:
            continue
        assert train["target_month"].max() <= t, "leakage in cow_model"
        w = _recency_weights(train["target_month"], t)
        y = train["target"]
        m_w = _hgb().fit(train[LEAN], y, sample_weight=w)
        m_h = _hgb().fit(train[LEAN], y)
        m_r = _ridge().fit(train[MOMENTUM], y)

        def predict(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
            c = {MEMBERS[0]: m_w.predict(frame[LEAN]), MEMBERS[1]: m_h.predict(frame[LEAN]), MEMBERS[2]: m_r.predict(frame[MOMENTUM])}
            c[METHOD] = np.mean([c[k] for k in MEMBERS], axis=0)
            return c

        res = pd.DataFrame({"district": test["district"].to_numpy(), "horizon": h})
        base = test["rate_t"].to_numpy()
        for k, c in predict(test).items():
            res[k] = base + c
        for nm, tb in scenarios.items():                      # same fitted models, other forward-weather inputs of forecast rows
            ts = tb[(tb["horizon"] == h) & (tb["origin"] == t)].set_index("district").reindex(test["district"])
            for k, c in predict(ts).items():
                res[f"{k} | {nm}"] = base + c
        parts.append(res)
    if not parts:
        return pd.DataFrame(columns=["district", "horizon"] + COW_COLUMNS)
    return pd.concat(parts, ignore_index=True)
