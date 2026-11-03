"""
step2_backtest.py - FILE 2 of 4: Walk-Forward Backtesting & Multi-Holdout Report
(Adapted for Karnataka State Milk Price Forecasting)
================================================================================
Models:
  - COW     : "Cow Ensemble v2" (HistGradientBoosting weighted, HistGradientBoosting, Ridge momentum via cow_model.py)
  - BUFFALO : Direct / RecursiveX Ridge, XGBoost, and Blend
Evaluations:
  - TEST A  : 2025 1-month ahead walk-forward backtest (origins 2025-01 to 2025-12)
  - TEST B  : March origins (2026 vs 2025) checking CBE (N+1 <= 2.0) and Q1 (<= 5.0) targets
  - PART D  : Multi-Holdout Report across 6 batches spaced 2 months apart (MultiHoldout_Report.xlsx)
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import step1_data_features as s1
import cow_model

try:
    from xgboost import XGBRegressor
    XGB_IS_REAL = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    XGB_IS_REAL = False

warnings.filterwarnings("ignore")
log = logging.getLogger("milk")

# ------------------------------------------------------------------ settings
TEST_A_FIRST_ORIGIN, TEST_A_LAST_ORIGIN = "2025-01-01", "2025-12-01"
TEST_B_ORIGIN, TEST_B_PRIOR_ORIGIN = "2026-03-01", "2025-03-01"
COMPARE_FIRST_ORIGIN = "2025-01-01"
CBE_TARGET, Q1_TARGET, CLIENT_CBE_MAE = 2.0, 5.0, 2.37
MIN_TRAIN_ROWS = 25
N_BOOT, BOOT_SEED = 2000, 0

# Multi-holdout configuration
HOLDOUT_N, HOLDOUT_STEP, HOLDOUT_LAST = 6, 2, None
HEADLINE_METHOD = "RecursiveX Blend"
HEADLINE_BY_TYPE = {"COW": cow_model.METHOD}
HOLDOUT_H = 6
HOLDOUT_QUARTERS = {"Q1": (1, 2, 3), "Q2": (4, 5, 6)}
HOLDOUT_QUARTER_TARGET = {"Q1": 5.0, "Q2": 8.0}

RECX_LAG = 12
FAMILIES = ["Direct", "RecursiveX"]
LEARNERS = ["Ridge", "XGB", "Blend"]
BASELINES = ["Naive", "Seasonal-trend naive", "Drift (6m avg)"]
COMPARATORS = ["Rolling 3M avg", "Rolling 6M avg", "ARIMA(1,1,0)"]
LEARNED = [f"{f} {m}" for f in FAMILIES for m in LEARNERS]
ALL_METHODS = BASELINES + LEARNED


# ------------------------------------------------------------- equations & importance helpers for Step 3
def ridge_equation(ridge, feats) -> Tuple[float, pd.Series, float]:
    sc, rg = ridge.steps[1][1], ridge.steps[2][1]
    coef = rg.coef_ / sc.scale_
    b0 = rg.intercept_ - np.sum(rg.coef_ * sc.mean_ / sc.scale_)
    return float(b0), pd.Series(coef, index=feats), float(rg.alpha_)


def equation_frame(mt: str, label: str, ridge, feats, horizon) -> pd.DataFrame:
    b0, coef, alpha = ridge_equation(ridge, feats)
    return pd.DataFrame({
        "Milk Type": mt.title(), "Model": label, "Horizon": horizon,
        "Term": ["Intercept"] + list(feats), "Coefficient": [b0] + coef.tolist(),
        "Median used if missing": [np.nan] + ridge.steps[0][1].statistics_.tolist(), "Ridge alpha": alpha
    })


_SHAP_WARNED = False


def shap_importance(xgb, X: pd.DataFrame) -> pd.Series:
    global _SHAP_WARNED
    try:
        import shap
        return pd.Series(np.abs(shap.TreeExplainer(xgb).shap_values(X)).mean(axis=0), index=X.columns)
    except Exception as exc:
        if not _SHAP_WARNED:
            log.warning("SHAP unavailable (%s); using model feature importance instead", exc)
            _SHAP_WARNED = True
        return pd.Series(xgb.feature_importances_, index=X.columns)


def ridge_std_coef(ridge, feats) -> pd.Series:
    return pd.Series(ridge.steps[2][1].coef_, index=feats)


def pct(s: pd.Series) -> pd.Series:
    t = s.abs().sum()
    return s.abs() / t * 100 if t > 0 else s * 0.0


# -------------------------------------------------------------------- models
def make_ridge():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         RidgeCV(alphas=np.logspace(-1, 3, 13)))


if XGB_IS_REAL:
    def make_xgb():
        return XGBRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, min_child_weight=3,
                            subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=5.0,
                            random_state=42, n_jobs=1)
else:
    class _StandIn(GradientBoostingRegressor):
        def _fill(self, X):
            X = np.array(X, dtype=float)
            i = np.where(np.isnan(X))
            X[i] = np.take(self._med, i[1])
            return X

        def fit(self, X, y):
            self._med = np.nan_to_num(np.nanmedian(np.asarray(X, dtype=float), axis=0))
            return super().fit(self._fill(X), y)

        def predict(self, X):
            return super().predict(self._fill(X))

    def make_xgb():
        return _StandIn(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, max_features=0.7,
                        min_samples_leaf=3, random_state=42)


def fit_pair(train: pd.DataFrame, feats: List[str]):
    X, y = train[feats], train["target"]
    return make_ridge().fit(X, y), make_xgb().fit(X, y)


def baseline_changes(rows: pd.DataFrame) -> Dict[str, np.ndarray]:
    h = rows["horizon"].to_numpy()
    return {
        "Naive": np.zeros(len(rows)),
        "Seasonal-trend naive": rows["ly_change"].fillna(0.0).to_numpy(),
        "Drift (6m avg)": rows["diff_mean6"].fillna(0.0).to_numpy() * h,
    }


def price_feature_frame(R: pd.DataFrame, target_month: pd.Timestamp) -> pd.DataFrame:
    F = pd.DataFrame({k: W.iloc[-1] for k, W in s1.price_block(R).items()})
    F["ly_change"] = R.iloc[-12] - R.iloc[-13]
    for k, v in s1.calendar_features(pd.DatetimeIndex([target_month])).items():
        F[k] = v[0]
    return F


def prepare_recx(panel: pd.DataFrame, market: pd.DataFrame, table: pd.DataFrame, cfg: s1.Config, spec: Dict) -> Dict:
    of = s1.origin_features(panel, market)
    wx = of[spec["observed"]]
    t1 = table[table["horizon"] == 1].copy()
    idx = pd.MultiIndex.from_arrays([t1[s1.DISTRICT], t1["origin"]])
    if spec["observed"]:
        lagged = wx.groupby(level=s1.DISTRICT).shift(RECX_LAG - 1).reindex(idx)
        for c in lagged.columns:
            t1[c] = lagged[c].to_numpy()
    m0 = s1.projected_market_features(market, 0, cfg.proj_window).reindex(t1["origin"])
    for c in spec["market"]:
        t1[c] = m0[c].to_numpy()
    return {"train": t1, "wx": wx, "market": market, "cfg": cfg, "spec": spec, "feats": spec["all"], "table": table}


def paths_from_fit(panel: pd.DataFrame, ridge, xgb, t: pd.Timestamp, H: int, feats: List[str], exog) -> Dict[str, Dict[int, pd.Series]]:
    base = panel["rate"].unstack(s1.DISTRICT).sort_index().loc[:t]
    out = {}
    for kind in LEARNERS:
        R, paths = base.copy(), {}
        for k in range(1, H + 1):
            d = R.index[-1]
            tgt = d + pd.DateOffset(months=1)
            F = pd.concat([price_feature_frame(R, tgt), exog(k).reindex(R.columns)], axis=1)
            live = (R.loc[d].notna() & F["diff_1"].notna()).to_numpy()
            pr, px = ridge.predict(F[feats]), xgb.predict(F[feats])
            chg = {"Ridge": pr, "XGB": px, "Blend": (pr + px) / 2}[kind]
            new = pd.Series(np.nan, index=R.columns)
            new[live] = R.loc[d].to_numpy()[live] + chg[live]
            R.loc[tgt] = new
            paths[k] = new
        out[kind] = paths
    return out


def make_exog(recx: Dict, t: pd.Timestamp):
    wx, market, cfg, spec = recx["wx"], recx["market"], recx["cfg"], recx["spec"]
    cache = {}
    fed = spec["table_fed"]
    tbl = recx["table"]
    at_t = tbl[tbl["origin"] == t].set_index([s1.DISTRICT, "horizon"])[fed] if fed else None

    def exog(k: int) -> pd.DataFrame:
        s_lag = t + pd.DateOffset(months=k - RECX_LAG)
        w = wx.xs(s_lag, level=s1.DATE) if s_lag in wx.index.levels[1] else wx.xs(wx.index.levels[1].max(), level=s1.DATE)
        if k - 1 not in cache:
            cache[k - 1] = s1.projected_market_features(market, k - 1, cfg.proj_window).loc[t]
        w = w.assign(**{c: cache[k - 1][c] for c in spec["market"]})
        if fed and at_t is not None:
            blk = at_t.xs(k, level="horizon").copy()
            flags = [c for c in fed if c in s1.FLAG_FEATURES]
            if flags and k > 1:
                prev = at_t.xs(k - 1, level="horizon")
                for c in flags:
                    blk[c] = (k * blk[c] - (k - 1) * prev[c].reindex(blk.index)).clip(0.0, 1.0).round(6)
            w = pd.concat([w, blk], axis=1)
        return w
    return exog


def arima110(y: np.ndarray, H: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 12:
        return np.full(H, np.nan)
    d = np.diff(y)
    x, z = d[:-1], d[1:]
    xm, zm, var = x.mean(), z.mean(), ((x - x.mean()) ** 2).sum()
    phi = float(np.clip(((x - xm) * (z - zm)).sum() / var, -0.95, 0.95)) if var > 0 else 0.0
    c = float(zm - phi * xm)
    pred_d = np.zeros(H)
    prev = d[-1]
    for i in range(H):
        prev = c + phi * prev
        pred_d[i] = prev
    return y[-1] + np.cumsum(pred_d)


def comparator_forecasts(panel: pd.DataFrame, t: pd.Timestamp, H: int, districts: List[str]) -> pd.DataFrame:
    R = panel["rate"].unstack(s1.DISTRICT).sort_index().loc[:t]
    rows = []
    for d in districts:
        y = R[d].dropna().to_numpy()
        if len(y) < 3:
            continue
        last = y[-1]
        r3 = float(np.mean(y[-3:]))
        r6 = float(np.mean(y[-6:])) if len(y) >= 6 else r3
        ar = arima110(y, H)
        for h in range(1, H + 1):
            rows.append({
                s1.DISTRICT: d, "horizon": h,
                "Rolling 3M avg": r3, "Rolling 6M avg": r6,
                "ARIMA(1,1,0)": ar[h - 1] if not np.isnan(ar[h - 1]) else last
            })
    return pd.DataFrame(rows).set_index([s1.DISTRICT, "horizon"]) if rows else pd.DataFrame(columns=[s1.DISTRICT, "horizon"] + COMPARATORS).set_index([s1.DISTRICT, "horizon"])


def forecast_origin(panel: pd.DataFrame, table: pd.DataFrame, t: pd.Timestamp, H: int, cfg: s1.Config,
                    recx: Dict, districts: List[str] | None = None, keep_fits: bool = False,
                    scenarios: Dict[str, pd.DataFrame] | None = None, cow_extra: bool = False) -> Tuple[pd.DataFrame, Dict]:
    scenarios = scenarios or {}
    if districts is None:
        districts = s1.forecastable_districts(panel, t, cfg.forecast_min_months)
    comps = comparator_forecasts(panel, t, H, districts)
    feats = recx["feats"]
    parts, fits = [], {}

    # RecursiveX fitting
    t1 = table[table["horizon"] == 1]
    trx = t1[(t1["target_month"] <= t) & t1["target"].notna()]
    s1.assert_no_leakage(trx, t)
    recx_fit = fit_pair(trx, feats)
    r1, x1 = recx_fit
    recx_paths = paths_from_fit(panel, r1, x1, t, H, feats, make_exog(recx, t))
    scn_paths = {nm: paths_from_fit(panel, r1, x1, t, H, feats, make_exog({**recx, "table": tb}, t)) for nm, tb in scenarios.items()}
    if keep_fits:
        fits[("RecursiveX", 1)] = (*recx_fit, trx)

    for h in range(1, H + 1):
        th = table[table["horizon"] == h]
        train = th[(th["target_month"] <= t) & th["target"].notna()]
        s1.assert_no_leakage(train, t)
        test = th[(th["origin"] == t) & th[s1.DISTRICT].isin(districts)]
        if test.empty or len(train) < MIN_TRAIN_ROWS:
            continue
        res = test[[s1.DISTRICT, "origin", "horizon", "target_month", "rate_t", "rate_actual"]].copy()
        res["n_train"] = len(train)
        for m, c in baseline_changes(test).items():
            res[m] = test["rate_t"].to_numpy() + c
        cdf = comps.reindex(pd.MultiIndex.from_arrays([test[s1.DISTRICT], test["horizon"]]))
        for m in COMPARATORS:
            res[m] = cdf[m].to_numpy()
        r, x = fit_pair(train, feats)
        pr, px = r.predict(test[feats]), x.predict(test[feats])
        for name, c in (("Ridge", pr), ("XGB", px), ("Blend", (pr + px) / 2)):
            res[f"Direct {name}"] = test["rate_t"].to_numpy() + c
        if keep_fits:
            fits[("Direct", h)] = (r, x, train)
        for kind in LEARNERS:
            res[f"RecursiveX {kind}"] = recx_paths[kind][h].reindex(test[s1.DISTRICT]).to_numpy()
        for nm, tb in scenarios.items():
            ts = tb[(tb["horizon"] == h) & (tb["origin"] == t)].set_index(s1.DISTRICT).reindex(test[s1.DISTRICT])
            pr_s, px_s = r.predict(ts[feats]), x.predict(ts[feats])
            for kind, c in (("Ridge", pr_s), ("XGB", px_s), ("Blend", (pr_s + px_s) / 2)):
                res[f"Direct {kind} | {nm}"] = test["rate_t"].to_numpy() + c
                res[f"RecursiveX {kind} | {nm}"] = scn_paths[nm][kind][h].reindex(test[s1.DISTRICT]).to_numpy()
        vol = panel["ts_kg"].reindex(pd.MultiIndex.from_arrays([test[s1.DISTRICT], test["target_month"]]))
        res["weight_ts_kg"] = vol.to_numpy()
        parts.append(res)
    out = pd.concat(parts, ignore_index=True)
    if cow_extra:
        cw = cow_model.cow_columns(table, t, districts, H, scenarios)
        out = out.merge(cw, on=[s1.DISTRICT, "horizon"], how="left")
    return out, fits


def fiscal_quarter(month: pd.Timestamp) -> str:
    fy = month.year if month.month >= 4 else month.year - 1
    return f"FY{fy % 100}-{(fy + 1) % 100} Q{((month.month - 4) % 12) // 3 + 1}"


def run_backtest(mt: str, panel: pd.DataFrame, table: pd.DataFrame, cfg: s1.Config, recx: Dict, H: int = 6) -> pd.DataFrame:
    last_actual = panel["rate"].dropna().index.get_level_values(s1.DATE).max()
    origins = pd.date_range(COMPARE_FIRST_ORIGIN, last_actual - pd.DateOffset(months=1), freq="MS")
    rows = []
    for t in origins:
        d, _ = forecast_origin(panel, table, t, H, cfg, recx, cow_extra=(mt == "COW"))
        d["milk_type"] = mt
        rows.append(d)
        log.info("  %s origin %s done", mt, f"{t:%b %Y}")
    d = pd.concat(rows, ignore_index=True)
    d = d[d["rate_actual"].notna()].copy()
    d["fiscal_quarter"] = d["target_month"].map(fiscal_quarter)
    return d


# ------------------------------------------------------------------ metrics
def _dir_acc(g: pd.DataFrame, m: str) -> float:
    if m == "Naive":
        return np.nan
    return float((np.sign(g[m] - g["rate_t"]) == np.sign(g["rate_actual"] - g["rate_t"])).mean() * 100)


def volume_weighted_mae(g: pd.DataFrame, method: str) -> float:
    w = g["weight_ts_kg"].fillna(0.0)
    mask = g["rate_actual"].notna() & g[method].notna() & (w > 0)
    if not mask.any():
        return np.nan
    err = (g.loc[mask, method] - g.loc[mask, "rate_actual"]).abs()
    return float((err * w.loc[mask]).sum() / w.loc[mask].sum())


def metric_row(g: pd.DataFrame, m: str) -> Dict[str, float]:
    e = g[m] - g["rate_actual"]
    mae = e.abs().mean()
    vmae = volume_weighted_mae(g, m) if "weight_ts_kg" in g.columns else np.nan
    naive_mae = (g["Naive"] - g["rate_actual"]).abs().mean()
    chg_f, chg_a = g[m] - g["rate_t"], g["rate_actual"] - g["rate_t"]
    vol = chg_f.std() / chg_a.std() if len(g) > 2 and chg_a.std() > 0 else np.nan
    dacc = _dir_acc(g, m)
    skill = 1 - mae / naive_mae if naive_mae > 0 else np.nan
    trust = np.nan
    if m != "Naive" and not np.isnan(skill):
        trust = 100 * (0.4 * min(1.0, CBE_TARGET / mae) + 0.3 * float(np.clip(skill, 0, 1))
                       + 0.3 * float(np.clip((dacc / 100 - 0.5) / 0.5, 0, 1)))
    return {
        "n": len(g), "MAE": mae, "VW-MAE": vmae, "RMSE": float(np.sqrt((e ** 2).mean())),
        "MAPE %": (e.abs() / g["rate_actual"]).mean() * 100, "Bias (fcst-actual)": e.mean(),
        "Directional acc %": dacc, "Volatility score": vol, "Skill vs naive": skill, "Trust score": trust
    }


def metrics_table(d: pd.DataFrame, by: List[str], methods: List[str] | None = None) -> pd.DataFrame:
    methods = methods or [m for m in ALL_METHODS if m in d.columns]
    rec = []
    for keys, g in d.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for m in methods:
            rec.append({**dict(zip(by, keys)), "method": m, **metric_row(g, m)})
    return pd.DataFrame(rec)


def aggregate_mae(d: pd.DataFrame, by: List[str], methods: List[str] | None = None) -> pd.DataFrame:
    methods = methods or [m for m in ALL_METHODS if m in d.columns]
    rec = []
    for keys, g in d.groupby(by):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for m in methods:
            w = g["weight_ts_kg"].fillna(0)
            gg = g.assign(_w=w, _a=g["rate_actual"] * w, _f=g[m] * w)
            s = gg.groupby(["origin", "horizon"])[["_w", "_a", "_f"]].sum()
            s = s[s["_w"] > 0]
            err = (s["_f"] / s["_w"] - s["_a"] / s["_w"]).abs()
            rec.append({**dict(zip(by, keys)), "method": m, "n_months": len(err), "Aggregate MAE": err.mean(),
                        "Volume-weighted MAE": err.mean()})
    return pd.DataFrame(rec)


def pivot(t: pd.DataFrame, index: List[str], value: str = "MAE") -> pd.DataFrame:
    cols = [m for m in ALL_METHODS if m in set(t["method"])]
    return t.pivot_table(index=index, columns="method", values=value)[cols].reset_index()


def bootstrap_gain(d: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    rng = np.random.default_rng(BOOT_SEED)
    rec = []
    for mt, g in d.groupby("milk_type"):
        origins = np.array(sorted(g["origin"].unique()))
        idx = {o: g["origin"] == o for o in origins}
        en = {o: (g.loc[idx[o], "Naive"] - g.loc[idx[o], "rate_actual"]).abs().sum() for o in origins}
        cnt = {o: idx[o].sum() for o in origins}
        for m in methods:
            if m == "Naive":
                continue
            em = {o: (g.loc[idx[o], m] - g.loc[idx[o], "rate_actual"]).abs().sum() for o in origins}
            gains = []
            for _ in range(N_BOOT):
                pick = rng.choice(origins, size=len(origins), replace=True)
                n = sum(cnt[o] for o in pick)
                gains.append((sum(en[o] for o in pick) - sum(em[o] for o in pick)) / n)
            tot = sum(cnt.values())
            rec.append({
                "milk_type": mt, "method": m, "gain vs naive (Rs/kg TS)": (sum(en.values()) - sum(em.values())) / tot,
                "95% low": np.percentile(gains, 2.5), "95% high": np.percentile(gains, 97.5),
                "clear gain?": "YES" if np.percentile(gains, 2.5) > 0 else "not clear"
            })
    return pd.DataFrame(rec)


# ------------------------------------------------------------------- Test A & B
def test_a(detail: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    a = detail[(detail["horizon"] == 1) & (detail["origin"] >= TEST_A_FIRST_ORIGIN) & (detail["origin"] <= TEST_A_LAST_ORIGIN)].copy()
    a["month"] = a["target_month"].dt.strftime("%Y-%m")
    met = metrics_table(a, ["milk_type"])
    agg = aggregate_mae(a, ["milk_type"])
    by_q = metrics_table(a, ["milk_type", "fiscal_quarter"])
    best = {mt: g[g["method"].isin(LEARNED)].sort_values("MAE").iloc[0]["method"] for mt, g in met.groupby("milk_type")}
    ctl = []
    for mt, g in by_q.groupby("milk_type"):
        for m in dict.fromkeys([best[mt], "Naive"]):
            for _, r in g[g["method"] == m].iterrows():
                ctl.append({"milk_type": mt, "method": m, "fiscal_quarter": r["fiscal_quarter"], "n": r["n"], "MAE": r["MAE"],
                            "target": CBE_TARGET, "status": "IN CONTROL" if r["MAE"] <= CBE_TARGET else "OVER TARGET"})
    return {
        "A_Metrics_2025": met,
        "A_Aggregate_Level": agg,
        "A_By_District": pivot(metrics_table(a, ["milk_type", s1.DISTRICT]), ["milk_type", s1.DISTRICT]),
        "A_By_Month": pivot(metrics_table(a, ["milk_type", "month"]), ["milk_type", "month"]),
        "A_By_Quarter": pivot(by_q, ["milk_type", "fiscal_quarter"]),
        "A_Control_Check": pd.DataFrame(ctl),
        "A_Bootstrap": bootstrap_gain(a, LEARNED),
        "A_Forecast_vs_Actual": a.drop(columns=["month"]).assign(origin=a["origin"].dt.date, target_month=a["target_month"].dt.date),
    }


def test_b(detail: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    rec, rows = [], []
    for label, origin in (("2026: train to Mar-26, forecast Apr-Jun 2026", TEST_B_ORIGIN),
                          ("2025: train to Mar-25, forecast Apr-Jun 2025", TEST_B_PRIOR_ORIGIN)):
        b = detail[(detail["origin"] == origin) & (detail["horizon"] <= 3)].copy()
        b["window"] = label
        rows.append(b)
        for mt, g in b.groupby("milk_type"):
            for m in ALL_METHODS + (cow_model.COW_COLUMNS if mt == "COW" else []):
                if m not in g.columns:
                    continue
                h1 = g[g["horizon"] == 1]
                q1 = g.groupby(s1.DISTRICT).agg(act=("rate_actual", "mean"), fcst=(m, "mean"))
                e_q1 = (q1["fcst"] - q1["act"]).abs().mean()
                rec.append({
                    "window": label, "origin": origin, "milk_type": mt, "method": m,
                    "h1_month": f"{pd.Timestamp(origin) + pd.DateOffset(months=1):%b-%y}",
                    "h1_MAE": (h1[m] - h1["rate_actual"]).abs().mean(),
                    "h1_target": CBE_TARGET, "h1_check": "MEETS" if (h1[m] - h1["rate_actual"]).abs().mean() <= CBE_TARGET else "OVER",
                    "Q1_months": f"{pd.Timestamp(origin) + pd.DateOffset(months=1):%b}-{pd.Timestamp(origin) + pd.DateOffset(months=3):%b %y}",
                    "Q1_MAE": e_q1, "Q1_target": Q1_TARGET, "Q1_check": "MEETS" if e_q1 <= Q1_TARGET else "OVER",
                })
    res = pd.DataFrame(rec)
    fva = pd.concat(rows, ignore_index=True)
    fva = fva.assign(origin=fva["origin"].dt.date, target_month=fva["target_month"].dt.date)
    return {"B_Summary": res, "B_Forecast_vs_Actual": fva}


# ------------------------------------------------------------------- PART D: Multi-Holdout
HOLDOUT_METHODS = ["Naive", "Rolling 3M avg", "Rolling 6M avg", "ARIMA(1,1,0)", "Seasonal-trend naive", "Drift (6m avg)"] + LEARNED + cow_model.COW_COLUMNS
MODELS_HO = [f"N+{h}" for h in range(1, HOLDOUT_H + 1)] + list(HOLDOUT_QUARTERS)


def holdout_origins(last_actual: pd.Timestamp, n: int = HOLDOUT_N, step: int = HOLDOUT_STEP, last: str | None = None) -> List[pd.Timestamp]:
    latest = last_actual - pd.DateOffset(months=HOLDOUT_H)
    end = pd.Timestamp(last) if last else latest
    if end > latest:
        raise ValueError(f"holdout origin {end:%b %Y} has fewer than {HOLDOUT_H} actual months after it")
    origins = [end - pd.DateOffset(months=step * i) for i in range(n - 1, -1, -1)]
    return origins


def _stats(g: pd.DataFrame, m: str) -> Dict[str, float]:
    g = g[g[m].notna() & g["rate_actual"].notna()]
    if g.empty:
        return {"n": 0, "MAE": np.nan, "VW-MAE": np.nan, "Aggregate MAE": np.nan, "MAPE %": np.nan, "Bias (fcst-actual)": np.nan}
    e = g[m] - g["rate_actual"]
    w = g["weight_ts_kg"].fillna(0.0)
    vw = float((e.abs() * w).sum() / w.sum()) if w.sum() > 0 else np.nan
    s = g.assign(_w=w, _a=g["rate_actual"] * w, _f=g[m] * w).groupby("origin")[["_w", "_a", "_f"]].sum()
    s = s[s["_w"] > 0]
    agg = float((s["_f"] / s["_w"] - s["_a"] / s["_w"]).abs().mean()) if len(s) else np.nan
    return {"n": len(g), "MAE": float(e.abs().mean()), "VW-MAE": vw, "Aggregate MAE": agg,
            "MAPE %": float((e.abs() / g["rate_actual"]).mean() * 100), "Bias (fcst-actual)": float(e.mean())}


def _quarter_frame(d: pd.DataFrame, hs: Tuple[int, ...], methods: List[str]) -> pd.DataFrame:
    g = d[d["horizon"].isin(hs)]
    key = [s1.DISTRICT, "origin"]
    g = g[g.groupby(key)["horizon"].transform("size") == len(hs)]
    if g.empty:
        return pd.DataFrame()
    q = g.groupby(key).agg(rate_t=("rate_t", "first"), rate_actual=("rate_actual", "mean"), weight_ts_kg=("weight_ts_kg", "mean")).reset_index()
    idx = pd.MultiIndex.from_frame(q[key])
    for m in methods:
        q[m] = g.groupby(key)[m].apply(lambda s: s.mean() if s.notna().all() else np.nan).reindex(idx).to_numpy()
    return q


# ---------------------------------------------------------- method comparison
def comparison(detail: pd.DataFrame, last_actual: pd.Timestamp, cfg: s1.Config):
    d4 = detail[(detail["horizon"] <= 4) & (detail["origin"] <= last_actual - pd.DateOffset(months=4))]
    d6 = detail[detail["origin"] <= last_actual - pd.DateOffset(months=6)]
    h1_4 = metrics_table(d4, ["milk_type"])
    by_h = pivot(metrics_table(d6, ["milk_type", "horizon"]), ["milk_type", "horizon"])
    sheets = {"C_Method_h1-4": h1_4, "C_MAE_by_Horizon_h1-6": by_h}
    best = {mt: g[g["method"].isin(LEARNED)].sort_values("MAE").iloc[0]["method"] for mt, g in h1_4.groupby("milk_type")}
    best["COW"] = cow_model.METHOD
    bands: Dict = {}
    for (mt, h), g in detail.groupby(["milk_type", "horizon"]):
        methods = LEARNED + (cow_model.COW_COLUMNS if mt == "COW" else [])
        for m in methods:
            if m not in g.columns:
                continue
            e = (g["rate_actual"] - g[m]).dropna().to_numpy()
            if len(e) >= 5:
                bands.setdefault(mt, {}).setdefault(m, {})[str(int(h))] = [float(np.percentile(e, 10)), float(np.percentile(e, 90)), len(e)]
    return sheets, best, bands


def holdout_sheets(detail: pd.DataFrame, origins: List[pd.Timestamp], headline: str = HEADLINE_METHOD) -> Dict[str, pd.DataFrame]:
    origins = [pd.Timestamp(o) for o in origins]
    d = detail[detail["origin"].isin(origins) & (detail["horizon"] <= HOLDOUT_H) & detail["rate_actual"].notna()].copy()
    methods = [m for m in HOLDOUT_METHODS if m in d.columns]

    batches = pd.DataFrame([{"Batch": i + 1, "Origin": o.date(),
                             "Forecast months": f"{o + pd.DateOffset(months=1):%b-%y} .. {o + pd.DateOffset(months=HOLDOUT_H):%b-%y}",
                             "Q1": f"{o + pd.DateOffset(months=1):%b}-{o + pd.DateOffset(months=3):%b %y}",
                             "Q2": f"{o + pd.DateOffset(months=4):%b}-{o + pd.DateOffset(months=6):%b %y}"}
                            for i, o in enumerate(origins)])
    batch_no = {o: i + 1 for i, o in enumerate(origins)}
    long_rows, per_batch, fva = [], [], []

    for mt, dm in d.groupby("milk_type"):
        hl = HEADLINE_BY_TYPE.get(mt, headline)
        qf = {q: _quarter_frame(dm, hs, methods) for q, hs in HOLDOUT_QUARTERS.items()}
        for m in methods:
            for h in range(1, HOLDOUT_H + 1):
                g = dm[dm["horizon"] == h]
                st = _stats(g, m)
                tgt = CBE_TARGET if h == 1 else np.nan
                long_rows.append({"Milk Type": mt.title(), "Method": m, "Model": f"N+{h}", "Measure": f"error of month t+{h}",
                                  "Batches": g["origin"].nunique(), **st, "Target": tgt})
            for q, hs in HOLDOUT_QUARTERS.items():
                tgt = HOLDOUT_QUARTER_TARGET[q]
                if len(qf[q]):
                    st = _stats(qf[q], m)
                    long_rows.append({"Milk Type": mt.title(), "Method": m, "Model": q, "Measure": "error of quarter AVERAGE",
                                      "Batches": qf[q]["origin"].nunique(), **st, "Target": tgt})
        for m in dict.fromkeys([hl, "Naive", "Rolling 3M avg"]):
            for o in origins:
                go = dm[dm["origin"] == o]
                row = {"Milk Type": mt.title(), "Method": m, "Batch": batch_no[o], "Origin": o.date()}
                for h in range(1, HOLDOUT_H + 1):
                    row[f"N+{h} MAE"] = _stats(go[go["horizon"] == h], m)["MAE"]
                for q, hs in HOLDOUT_QUARTERS.items():
                    qo = qf[q][qf[q]["origin"] == o] if len(qf[q]) else pd.DataFrame()
                    row[f"{q} MAE (quarter avg)"] = _stats(qo, m)["MAE"] if len(qo) else np.nan
                per_batch.append(row)
        x = dm[[s1.DISTRICT, "origin", "horizon", "target_month", "rate_t", "rate_actual", hl, "Rolling 3M avg", "Naive", "weight_ts_kg"]].copy()
        x.insert(0, "Milk Type", mt.title())
        x.insert(1, "Batch", x["origin"].map(batch_no))
        x["Abs error model"] = (x[hl] - x["rate_actual"]).abs()
        x["MAPE % model"] = x["Abs error model"] / x["rate_actual"] * 100
        x["Abs error rolling 3M"] = (x["Rolling 3M avg"] - x["rate_actual"]).abs()
        x["Abs error last month repeat"] = (x["Naive"] - x["rate_actual"]).abs()
        x = x.rename(columns={hl: f"Model forecast ({hl})", "rate_actual": "Actual", "rate_t": "Last actual at origin",
                              "Rolling 3M avg": "Last-3-month average", "Naive": "Last-month repeat", "weight_ts_kg": "TS kg (weight)"})
        x["origin"], x["target_month"] = x["origin"].dt.date, x["target_month"].dt.date
        fva.append(x.sort_values(["Batch", "horizon", s1.DISTRICT]))

    long = pd.DataFrame(long_rows)
    long["Check (district MAE)"] = np.where(long["Target"].isna(), "n/a", np.where(long["MAE"] <= long["Target"], "MEETS", "OVER"))
    long["Check (weighted MAE)"] = np.where(long["Target"].isna(), "n/a", np.where(long["VW-MAE"] <= long["Target"], "MEETS", "OVER"))
    order = {m: i for i, m in enumerate(MODELS_HO)}
    morder = {m: i for i, m in enumerate(methods)}
    long["_o"], long["_m"] = long["Model"].map(order), long["Method"].map(morder)
    long = long.sort_values(["Milk Type", "_o", "Measure", "_m"]).drop(columns=["_o", "_m"]).reset_index(drop=True)

    def wide(value: str, measure: str) -> pd.DataFrame:
        sub = long[(long["Measure"].isin([f"error of month t+{h}" for h in range(1, HOLDOUT_H + 1)] + [measure]))]
        p = sub.pivot_table(index=["Milk Type", "Method"], columns="Model", values=value)
        p = p[[c for c in MODELS_HO if c in p.columns]].reset_index()
        p["_m"] = p["Method"].map(morder)
        return p.sort_values(["Milk Type", "_m"]).drop(columns="_m")

    qm = "error of quarter AVERAGE"
    comp_mae, comp_vw = wide("MAE", qm), wide("VW-MAE", qm)
    comp_agg = wide("Aggregate MAE", qm)

    chk = []
    for mt in sorted(long["Milk Type"].unique()):
        for model in MODELS_HO:
            meas = f"error of month t+{model[2:]}" if model.startswith("N+") else qm
            g = long[(long["Milk Type"] == mt) & (long["Model"] == model) & (long["Measure"] == meas)].set_index("Method")
            hl = HEADLINE_BY_TYPE.get(str(mt).upper(), headline)
            if hl not in g.index:
                continue
            h = g.loc[hl]
            pb = pd.DataFrame(per_batch)
            pb = pb[(pb["Milk Type"] == mt) & (pb["Method"] == hl)]
            col = f"{model} MAE" if model.startswith("N+") else f"{model} MAE (quarter avg)"
            k_ok = int((pb[col] <= h["Target"]).sum()) if h["Target"] == h["Target"] else np.nan
            simple = [m for m in ("Naive", "Rolling 3M avg", "Rolling 6M avg", "ARIMA(1,1,0)") if m in g.index]
            best_simple = min(simple, key=lambda m: g.loc[m, "MAE"])
            chk.append({
                "Milk Type": mt, "Model": model, "Our model": hl, "Batches": int(h["Batches"]),
                "Test points (district x batch)": int(h["n"]), "MAE": h["MAE"], "Weighted MAE": h["VW-MAE"],
                "Aggregate MAE": h["Aggregate MAE"], "MAPE %": h["MAPE %"], "Target": h["Target"],
                "Check (MAE)": "n/a" if h["Target"] != h["Target"] else "MEETS" if h["MAE"] <= h["Target"] else "OVER",
                "Check (weighted MAE)": "n/a" if h["Target"] != h["Target"] else "MEETS" if h["VW-MAE"] <= h["Target"] else "OVER",
                "Batches within target": k_ok, "Last month repeated MAE": g.loc["Naive", "MAE"],
                "Rolling 3M avg MAE": g.loc["Rolling 3M avg", "MAE"], "Best simple method": best_simple,
                "Best simple MAE": g.loc[best_simple, "MAE"], "Gain over best simple method %": (1 - h["MAE"] / g.loc[best_simple, "MAE"]) * 100
            })
    check = pd.DataFrame(chk)

    return {
        "H_Batches": batches, "H_Target_Check": check, "Comparison_MAE": comp_mae, "Comparison_Weighted_MAE": comp_vw,
        "Comparison_Aggregate_MAE": comp_agg, "H_Per_Batch": pd.DataFrame(per_batch), "H_Long": long,
        "H_Forecast_vs_Actual": pd.concat(fva, ignore_index=True)
    }


def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)


def holdout_main(detail: pd.DataFrame, last_actual: pd.Timestamp, outdir: Path) -> Dict[str, pd.DataFrame]:
    origins = holdout_origins(last_actual, HOLDOUT_N, HOLDOUT_STEP, HOLDOUT_LAST)
    sheets = holdout_sheets(detail, origins, HEADLINE_METHOD)
    write_excel(outdir / "MultiHoldout_Report.xlsx", sheets)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(f"\n=== MULTI-HOLDOUT: {len(origins)} batches, origins " + ", ".join(f"{o:%b-%y}" for o in origins) + f" | our model = {HEADLINE_METHOD} (buffalo) / {HEADLINE_BY_TYPE['COW']} (cow) ===")
    print(sheets["H_Batches"].to_string(index=False))
    show = ["Milk Type", "Model", "Batches", "Test points (district x batch)", "MAE", "Weighted MAE", "Target", "Check (MAE)", "Batches within target", "Gain over best simple method %"]
    print("\nTarget Check Summary:")
    print(sheets["H_Target_Check"][show].round(2).to_string(index=False))
    print("\nsaved ->", outdir / "MultiHoldout_Report.xlsx")
    return sheets


def backtest_main(cfg: s1.Config, outdir: Path) -> pd.DataFrame:
    outdir.mkdir(parents=True, exist_ok=True)
    raw = s1.load_raw(cfg)
    panels, market = s1.build_panels(raw, cfg)
    last_actual = raw.loc[raw["row_valid"], s1.DATE].max()
    detail_frames = []

    for mt in s1.MILK_TYPES:
        table_path = outdir / f"training_table_{mt.lower()}.csv"
        table = pd.read_csv(table_path) if table_path.exists() else s1.build_training_table(panels[mt], market, cfg)
        table["origin"] = pd.to_datetime(table["origin"])
        table["target_month"] = pd.to_datetime(table["target_month"])
        spec = s1.spec_for(mt, forward=cfg.forward_inputs, flags=cfg.use_flags)
        recx = prepare_recx(panels[mt], market, table, cfg, spec)
        d = run_backtest(mt, panels[mt], table, cfg, recx, H=6)
        detail_frames.append(d)

    detail = pd.concat(detail_frames, ignore_index=True)
    detail.to_csv(outdir / "backtest_detail.csv", index=False)
    log.info("Saved walk-forward detail table -> %s", outdir / "backtest_detail.csv")

    ta = test_a(detail)
    tb = test_b(detail)
    tc, best, bands = comparison(detail, last_actual, cfg)
    meta = {"best_method": best, "bands": bands, "xgb_is_real": XGB_IS_REAL}
    (outdir / "backtest_meta.json").write_text(json.dumps(meta, indent=2))
    write_excel(outdir / "Backtest_Summary.xlsx", {**ta, **tb, **tc})
    log.info("Saved backtest summary -> %s", outdir / "Backtest_Summary.xlsx")
    log.info("Saved backtest metadata -> %s", outdir / "backtest_meta.json")

    holdout_main(detail, last_actual, outdir)
    return detail


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step 2: walk-forward backtest and multi-holdout report")
    ap.add_argument("--data", default=s1.Config.data_csv)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--no-forward", action="store_true")
    ap.add_argument("--no-flags", action="store_true")
    ap.add_argument("--keep-2021", action="store_true")
    ap.add_argument("--holdout-n", type=int, default=HOLDOUT_N)
    ap.add_argument("--holdout-step", type=int, default=HOLDOUT_STEP)
    ap.add_argument("--holdout-last", default=None)
    ap.add_argument("--headline-method", default=HEADLINE_METHOD)
    s1.add_swing_args(ap)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = s1.config_from_args(a)
    HOLDOUT_N, HOLDOUT_STEP, HOLDOUT_LAST, HEADLINE_METHOD = a.holdout_n, a.holdout_step, a.holdout_last, a.headline_method
    out = s1.resolve(a.outdir)
    backtest_main(cfg, out)
