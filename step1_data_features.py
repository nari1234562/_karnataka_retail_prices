"""
step1_data_features.py - FILE 1 of 4: data, target, features, intervention flags, weather expectation + weather band
(Adapted for Karnataka State Dairy Price Forecasting)
====================================================================================================================
Project  : Karnataka Milk Procurement / Retail Total Solids Rate Forecast (COW & BUFFALO)
Target   : Price_INR_per_kg_TS (Rs per kg of Total Solids)
Level    : DISTRICT x MONTH, COW and BUFFALO modelled separately
Data     : Karnataka_Milk_Final_Jan2021_Oct2026.csv (31 districts, 2021-01 to 2026-10)
"""

from __future__ import annotations

import argparse
import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("milk")

HERE = Path(__file__).resolve().parent


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else HERE / p


def find_data_file(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    folders = [HERE, HERE / "data", Path.cwd(), Path.cwd() / "data"]
    patterns = ["Karnataka_Milk_Final_Jan2021_Oct2026.csv", "Karnataka*.csv", "Karnataka*.xlsx"]
    for pat in patterns:
        hits = sorted({f for fo in folders if fo.exists() for f in fo.glob(pat) if not f.name.startswith("~$")},
                      key=lambda f: f.stat().st_mtime, reverse=True)
        if hits:
            log.info("Using data file: %s", hits[0])
            return hits[0]
    raise FileNotFoundError(f"Data file not found: {p}\nSearched in {folders} for {patterns}")


DATE, DISTRICT, MTYPE = "month", "district", "milk_type"
MILK_TYPES: Tuple[str, ...] = ("COW", "BUFFALO")

MARKET_PCT = ["feed", "maize", "soy_meal", "smp", "ghee", "cpi", "butter"]   # price-type series: % changes
MARKET_COLS = MARKET_PCT + ["oni", "monsoon"]                                # + anomaly-type series: levels
WEATHER_COLS = ["temp_avg", "temp_max", "temp_min", "rain", "humid"]


@dataclass(frozen=True)
class Config:
    data_csv: str = "Karnataka_Milk_Final_Jan2021_Nov2026.csv"
    data_sheet: str = "Final_Merged"
    max_horizon: int = 18
    short_horizon: int = 4
    min_ts_kg: float = 2000.0
    min_row_history: int = 1
    forecast_min_months: int = 6
    forward_inputs: bool = True
    use_flags: bool = True
    fwd_min_years: int = 1
    train_from_target: str | None = "2022-01-01"
    swing_start: str = "2022-01-01"
    swing_peak_search: Tuple[str, str] = ("2022-01-01", "2023-12-01")
    swing_end_search: str = "2024-12-01"
    run_up_override: Tuple[str, str] | None = None
    crash_override: Tuple[str, str] | None = None
    project_market: bool = True
    proj_window: int = 6


def _parse_window(text: str | None) -> Tuple[str, str] | None:
    if not text:
        return None
    lo, hi = [pd.Timestamp(x.strip()).to_period("M").to_timestamp() for x in text.split(":")]
    if hi < lo:
        raise ValueError(f"window end before start: {text}")
    return (lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d"))


def add_swing_args(ap) -> None:
    ap.add_argument("--run-up", default=None, help="override run-up window, e.g. 2022-01:2023-04")
    ap.add_argument("--crash", default=None, help="override crash window, e.g. 2023-05:2024-02")


def config_from_args(a, **extra) -> Config:
    data_file = getattr(a, "data", Config.data_csv)
    return Config(
        data_csv=str(resolve(data_file)),
        forward_inputs=not getattr(a, "no_forward", False),
        use_flags=not getattr(a, "no_flags", False),
        run_up_override=_parse_window(getattr(a, "run_up", None)),
        crash_override=_parse_window(getattr(a, "crash", None)),
        train_from_target=None if getattr(a, "keep_2021", False) else Config.train_from_target,
        **extra
    )


# ------------------------------------------------------------------ loading
def load_raw(cfg: Config) -> pd.DataFrame:
    """Read Karnataka dataset, drop any exact duplicates, map column names and compute standardized features."""
    target_path = find_data_file(cfg.data_csv)
    if str(target_path).endswith(".csv"):
        raw = pd.read_csv(target_path)
    else:
        raw = pd.read_excel(target_path)

    # Standardize column headers
    raw["month"] = pd.to_datetime(raw["Month"]).dt.to_period("M").dt.to_timestamp()
    raw["district"] = raw["District"].astype(str).str.strip()
    raw["milk_type"] = raw["Milk_Type"].astype(str).str.upper().str.strip()

    # Drop any duplicate observations
    raw = raw.drop_duplicates(subset=["month", "district", "milk_type"]).copy()

    # Prices and Quality
    raw["rate"] = raw["Price_INR_per_kg_TS"].astype(float)
    raw["rate_ltr"] = raw["Retail_Price_INR_per_L"].astype(float)
    raw["fat_pct"] = raw["Fat_Pct"].astype(float)
    raw["snf_pct"] = raw["SNF_Pct"].astype(float)
    raw["ts_pct"] = raw["Total_Solids_Pct"].astype(float)

    # Volume & solids weights
    raw["qty_ltr"] = 250000.0
    raw["ts_kg"] = raw["qty_ltr"] * (raw["ts_pct"] / 100.0) * 1.03
    raw["n_centres"] = 1
    raw["row_valid"] = (raw["rate"] > 0) & (raw["ts_kg"] > 0)

    # Agent / Government Incentive % (Ksheera Dhare subsidy DBT ~ Rs 5/L)
    raw["incentive_pct"] = (5.0 / (raw["rate_ltr"] - 5.0).clip(lower=1.0) * 100.0).round(2)

    # Weather features
    raw["temp_avg"] = raw["Avg_Temperature_C"].astype(float)
    raw["temp_max"] = raw["Avg_Max_Temperature_C"].astype(float)
    raw["temp_min"] = raw["Avg_Min_Temperature_C"].astype(float)
    raw["rain"] = raw["Total_Precipitation_mm"].astype(float)
    raw["humid"] = raw["Avg_Relative_Humidity_Pct"].astype(float)

    # Market features
    raw["maize"] = raw["Maize_INR_per_Qtl"].astype(float)
    raw["soy_meal"] = (raw["Soybean_Meal_INR_per_Kg"].astype(float) * 100.0).round(2)
    # Standard 50kg compound cattle feed pellet index
    raw["feed"] = (0.5 * (0.6 * raw["maize"] + 0.4 * raw["soy_meal"])).round(2)
    raw["smp"] = raw["SMP_INR_per_Kg"].astype(float)
    raw["ghee"] = raw["Ghee_Karnataka_INR_per_Qtl"].astype(float)
    raw["butter"] = raw["Butter_Karnataka_INR_per_100g"].astype(float)
    raw["cpi"] = raw["CPI_General_Index_Base2012"].astype(float)
    raw["oni"] = raw["ONI_Value"].astype(float)

    # State monsoon impact index
    m_rain = raw.groupby("month")["rain"].transform("mean")
    m_norm = raw.groupby(raw["month"].dt.month)["rain"].transform("mean").clip(lower=1.0)
    raw["monsoon"] = (m_rain / m_norm).round(3)

    return raw


# ------------------------------------------------------------------- panels
def build_panels(raw: pd.DataFrame, cfg: Config) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """District-month panel per milk type (full monthly calendar) plus shared market table."""
    calendar = pd.date_range(raw[DATE].min(), raw[DATE].max(), freq="MS")

    # Macro and market benchmarks: mean per month
    market = raw.groupby(DATE)[MARKET_COLS].mean().reindex(calendar)
    market = market.ffill().bfill()

    # Weather: district x month
    weather = raw.groupby([DISTRICT, DATE])[WEATHER_COLS].first()

    panels: Dict[str, pd.DataFrame] = {}
    for mt in MILK_TYPES:
        g = raw[(raw[MTYPE] == mt) & raw["row_valid"]].set_index([DISTRICT, DATE])
        grid = pd.MultiIndex.from_product([sorted(raw[DISTRICT].unique()), calendar], names=[DISTRICT, DATE])
        p = g.reindex(grid)
        p["ts_kg"] = p["ts_kg"].fillna(cfg.min_ts_kg)
        p["qty_ltr"] = p["qty_ltr"].fillna(250000.0)
        p["n_centres"] = p["n_centres"].fillna(1)
        p[WEATHER_COLS] = weather.reindex(grid).to_numpy()
        panels[mt] = p
    return panels, market


# ----------------------------------------------------------------- feature lists
CORE: List[str] = [
    # price dynamics at origin (16)
    "diff_1", "diff_3", "diff_6", "diff_mean6", "diff_std6", "accel", "dev_ma3", "dev_ma6", "dev_ma12", "yoy",
    "range12_pos", "rel_mean", "rel_ch3", "state_ch3", "ly_change", "rate_t",
    # calendar of TARGET month (3)
    "tgt_month_sin", "tgt_month_cos", "tgt_summer",
    # weather already observed at origin (7)
    "temp_avg_t", "temp_max_t", "temp_min_t", "rain_t", "temp_max_3m", "rain_3m", "rain_6m",
    # milk quality at origin (2)
    "fat_t", "snf_t",
    # market drivers, projected (8)
    "feed_ch3", "maize_ch3", "soy_meal_ch3", "smp_ch3", "ghee_ch3", "cpi_ch3", "oni_t", "oni_l3",
]
COW_EXTRA: List[str] = ["humid_t", "thi_t", "snf_ch3", "feed_maize_ratio", "smp_ch6", "incentive_pct"]
BUFFALO_EXTRA: List[str] = ["thi_t", "thi_3m", "fat_ch3", "fat_snf_ratio", "ghee_ch6", "butter_ch3", "ghee_smp_ratio",
                            "tgt_monsoon", "qty_ch3_pct", "monsoon_t"]
FULL_FEATURES_BY_TYPE: Dict[str, List[str]] = {"COW": CORE + COW_EXTRA, "BUFFALO": CORE + BUFFALO_EXTRA}

FORWARD_FEATURES: List[str] = ["temp_avg_t", "temp_max_t", "temp_min_t", "rain_t", "temp_max_3m", "rain_3m", "rain_6m",
                               "humid_t", "thi_t", "thi_3m", "fat_t", "snf_t", "fat_ch3", "snf_ch3", "fat_snf_ratio"]
FLAG_FEATURES: List[str] = []
ALL_FEATURES: List[str] = list(dict.fromkeys(FULL_FEATURES_BY_TYPE["COW"] + FULL_FEATURES_BY_TYPE["BUFFALO"]))

DROP_BY_TYPE: Dict[str, List[str]] = {
    "COW": [],
    "BUFFALO": ["accel", "ghee_smp_ratio"],
}


def _apply_drops() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for mt, full in FULL_FEATURES_BY_TYPE.items():
        drop = list(DROP_BY_TYPE.get(mt, []))
        out[mt] = [f for f in full if f not in set(drop)]
    return out


FEATURES_BY_TYPE: Dict[str, List[str]] = _apply_drops()

FEATURE_INFO: Dict[str, Tuple[str, str, str]] = {
    "diff_1": ("Price momentum", "price", "rate(t) - rate(t-1)"),
    "diff_3": ("Price momentum", "price", "rate(t) - rate(t-3)"),
    "diff_6": ("Price momentum", "price", "rate(t) - rate(t-6)"),
    "diff_mean6": ("Price momentum", "price", "average of last 6 monthly changes"),
    "diff_std6": ("Price momentum", "price", "standard deviation of last 6 monthly changes"),
    "accel": ("Price momentum", "price", "diff_1 minus previous month change"),
    "dev_ma3": ("Price vs own history", "price", "rate(t) minus 3-month moving average"),
    "dev_ma6": ("Price vs own history", "price", "rate(t) minus 6-month moving average"),
    "dev_ma12": ("Price vs own history", "price", "rate(t) minus 12-month moving average"),
    "yoy": ("Price vs own history", "price", "rate(t) - rate(t-12)"),
    "range12_pos": ("Price vs own history", "price", "position of rate(t) in 12-month range [0, 1]"),
    "rate_t": ("Price vs own history", "price", "rate level at origin month"),
    "rel_mean": ("Price vs other districts", "price", "rate(t) minus mean of all districts"),
    "rel_ch3": ("Price vs other districts", "price", "change in rel_mean over 3 months"),
    "state_ch3": ("Price vs other districts", "price", "average 3-month change of all districts"),
    "ly_change": ("Seasonal pattern", "price", "same window one year earlier: rate(t+h-12) - rate(t-12)"),
    "tgt_month_sin": ("Seasonal pattern", "calendar", "SIN(2*PI*month/12) of target month"),
    "tgt_month_cos": ("Seasonal pattern", "calendar", "COS(2*PI*month/12) of target month"),
    "tgt_summer": ("Seasonal pattern", "calendar", "1 if target month is Mar-Jun else 0"),
    "tgt_monsoon": ("Seasonal pattern", "calendar", "1 if target month is Jul-Sep else 0"),
    "temp_avg_t": ("Weather", "observed", "district average temperature"),
    "temp_max_t": ("Weather", "observed", "district average daily maximum temperature"),
    "temp_min_t": ("Weather", "observed", "district average daily minimum temperature"),
    "rain_t": ("Weather", "observed", "district monthly rainfall (mm)"),
    "temp_max_3m": ("Weather", "observed", "3-month average of maximum temperature"),
    "rain_3m": ("Weather", "observed", "rainfall of last 3 months"),
    "rain_6m": ("Weather", "observed", "rainfall of last 6 months"),
    "humid_t": ("Heat stress", "observed", "district average relative humidity %"),
    "thi_t": ("Heat stress", "observed", "Temperature-Humidity Index"),
    "thi_3m": ("Heat stress", "observed", "3-month average of THI"),
    "fat_t": ("Milk quality", "observed", "fat % of milk"),
    "snf_t": ("Milk quality", "observed", "SNF % of milk"),
    "fat_ch3": ("Milk quality", "observed", "fat % change over 3 months"),
    "snf_ch3": ("Milk quality", "observed", "SNF % change over 3 months"),
    "fat_snf_ratio": ("Milk quality", "observed", "fat % / SNF %"),
    "qty_ch3_pct": ("Supply & incentive", "observed", "% change in volume vs 3 months ago"),
    "incentive_pct": ("Supply & incentive", "observed", "estimated agent / DBT incentive %"),
    "feed_ch3": ("Feed & commodity costs", "market", "3-month % change of cattle feed"),
    "maize_ch3": ("Feed & commodity costs", "market", "3-month % change of maize"),
    "soy_meal_ch3": ("Feed & commodity costs", "market", "3-month % change of soybean meal"),
    "feed_maize_ratio": ("Feed & commodity costs", "market", "feed price / maize price"),
    "smp_ch3": ("Dairy product prices", "market", "3-month % change of SMP"),
    "smp_ch6": ("Dairy product prices", "market", "6-month % change of SMP"),
    "ghee_ch3": ("Dairy product prices", "market", "3-month % change of ghee"),
    "ghee_ch6": ("Dairy product prices", "market", "6-month % change of ghee"),
    "butter_ch3": ("Dairy product prices", "market", "3-month % change of butter"),
    "ghee_smp_ratio": ("Dairy product prices", "market", "ghee price / SMP price"),
    "cpi_ch3": ("Macro & climate", "market", "3-month % change of CPI General"),
    "oni_t": ("Macro & climate", "market", "ONI anomaly at target month, projected"),
    "oni_l3": ("Macro & climate", "market", "ONI anomaly 3 months before target month"),
    "monsoon_t": ("Macro & climate", "market", "Monsoon impact index, projected"),
}
# Intervention flags removed: run_up_share, crash_share
for _f in FORWARD_FEATURES:
    _g, _k, _t = FEATURE_INFO[_f]
    FEATURE_INFO[_f] = (_g, "forward", _t + " [TARGET MONTH: actual <= t, same-month earlier-year average > t]")

FEATURE_GROUP = {k: v[0] for k, v in FEATURE_INFO.items()}
FEATURE_KIND = {k: v[1] for k, v in FEATURE_INFO.items()}


def spec_for(mt: str, full: bool = False, forward: bool = True, flags: bool = True) -> Dict[str, List[str]]:
    base = FULL_FEATURES_BY_TYPE[mt] if full else FEATURES_BY_TYPE[mt]
    return spec_from(list(base) + (FLAG_FEATURES if flags else []), forward)


def spec_from(f: List[str], forward: bool = True) -> Dict[str, List[str]]:
    def kind(x: str) -> str:
        k = FEATURE_KIND[x]
        return "observed" if (k == "forward" and not forward) else k
    return {
        "all": list(f),
        "price_cal": [x for x in f if kind(x) in ("price", "calendar")],
        "observed": [x for x in f if kind(x) == "observed"],
        "market": [x for x in f if kind(x) == "market"],
        "table_fed": [x for x in f if kind(x) in ("forward", "flag")],
    }


# ----------------------------------------------------------------- price features
def calendar_features(month: pd.Series | pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    m = np.asarray(pd.DatetimeIndex(month).month)
    return {
        "tgt_month_sin": np.sin(2 * np.pi * m / 12),
        "tgt_month_cos": np.cos(2 * np.pi * m / 12),
        "tgt_summer": ((m >= 3) & (m <= 6)).astype(float),
        "tgt_monsoon": ((m >= 7) & (m <= 9)).astype(float),
    }


def price_block(R: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    step = R.diff()
    out = {"rate_t": R, "diff_1": R - R.shift(1), "diff_3": R - R.shift(3), "diff_6": R - R.shift(6)}
    out["diff_mean6"] = step.rolling(6, min_periods=3).mean()
    out["diff_std6"] = step.rolling(6, min_periods=3).std()
    out["accel"] = step - step.shift(1)
    out["dev_ma3"] = R - R.rolling(3, min_periods=2).mean()
    out["dev_ma6"] = R - R.rolling(6, min_periods=3).mean()
    out["dev_ma12"] = R - R.rolling(12, min_periods=6).mean()
    out["yoy"] = R - R.shift(12)
    hi, lo = R.rolling(12, min_periods=6).max(), R.rolling(12, min_periods=6).min()
    out["range12_pos"] = ((R - lo) / (hi - lo)).where(hi > lo)
    rel = R.sub(R.mean(axis=1), axis=0)
    out["rel_mean"] = rel
    out["rel_ch3"] = rel - rel.shift(3)
    state3 = out["diff_3"].mean(axis=1)
    out["state_ch3"] = pd.DataFrame({c: state3 for c in R.columns})
    return out


def _long(W: pd.DataFrame, index: pd.MultiIndex) -> np.ndarray:
    s = W.unstack()
    s.index.set_names([DISTRICT, DATE], inplace=True)
    return s.reindex(index).to_numpy()


def _by_district(s: pd.Series, fn) -> pd.Series:
    return s.groupby(level=DISTRICT).transform(fn)


def origin_features(panel: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    d = panel.sort_index().copy()
    R = d["rate"].unstack(DISTRICT).sort_index()
    for name, W in price_block(R).items():
        d[name] = _long(W, d.index)
    dates = d.index.get_level_values(DATE)

    d["temp_avg_t"], d["temp_max_t"], d["temp_min_t"], d["rain_t"], d["humid_t"] = (
        d["temp_avg"], d["temp_max"], d["temp_min"], d["rain"], d["humid"]
    )
    d["temp_max_3m"] = _by_district(d["temp_max"], lambda s: s.rolling(3, min_periods=2).mean())
    d["rain_3m"] = _by_district(d["rain"], lambda s: s.rolling(3, min_periods=2).sum())
    d["rain_6m"] = _by_district(d["rain"], lambda s: s.rolling(6, min_periods=3).sum())
    d["thi_t"] = 0.8 * d["temp_avg"] + d["humid"] / 100.0 * (d["temp_avg"] - 14.4) + 46.4
    d["thi_3m"] = _by_district(d["thi_t"], lambda s: s.rolling(3, min_periods=2).mean())

    d["fat_t"], d["snf_t"] = d["fat_pct"], d["snf_pct"]
    d["fat_ch3"] = d["fat_pct"] - d["fat_pct"].groupby(level=DISTRICT).shift(3)
    d["snf_ch3"] = d["snf_pct"] - d["snf_pct"].groupby(level=DISTRICT).shift(3)
    d["fat_snf_ratio"] = d["fat_pct"] / d["snf_pct"].clip(lower=0.1)

    q = d["qty_ltr"]
    d["qty_ch3_pct"] = (q / q.groupby(level=DISTRICT).shift(3) - 1) * 100

    mf = projected_market_features(market, 0).reindex(dates)
    for c in mf.columns:
        d[c] = mf[c].to_numpy()
    return d


def projected_market_features(market: pd.DataFrame, h: int, window: int = 6) -> pd.DataFrame:
    m = market[MARKET_COLS].astype(float)
    drift = (m - m.shift(window)) / window

    def proj(k: int) -> pd.DataFrame:
        return m + k * drift if k > 0 else m.shift(-k)

    now, back3, back6 = proj(h), proj(h - 3), proj(h - 6)
    out = pd.DataFrame(index=market.index)
    for c in MARKET_PCT:
        out[f"{c}_ch3"] = (now[c] / back3[c].clip(lower=1e-3) - 1) * 100
    for c in ("smp", "ghee"):
        out[f"{c}_ch6"] = (now[c] / back6[c].clip(lower=1e-3) - 1) * 100
    out["ghee_smp_ratio"] = now["ghee"] / now["smp"].clip(lower=1.0)
    out["feed_maize_ratio"] = now["feed"] / now["maize"].clip(lower=1.0)
    out["oni_t"], out["oni_l3"], out["monsoon_t"] = now["oni"], back3["oni"], now["monsoon"]
    return out


# ----------------------------------------------------------------- forward inputs
FWD_SERIES = ["temp_avg", "temp_max", "temp_min", "rain", "humid", "fat_pct", "snf_pct"]


def expected_wide(A: pd.DataFrame, min_years: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mon = A.index.month
    own = A.groupby(mon).transform(lambda s: s.expanding().mean().shift(1))
    n_prev = A.notna().astype(float).groupby(mon).transform(lambda s: s.cumsum().shift(1))
    own = own.where(n_prev >= min_years)
    row_sum, row_cnt = A.sum(axis=1), A.notna().sum(axis=1).astype(float)
    ps = row_sum.groupby(mon).transform(lambda s: s.cumsum().shift(1))
    pc = row_cnt.groupby(mon).transform(lambda s: s.cumsum().shift(1))
    pooled = ps / pc.where(pc > 0)
    fb = own.isna()
    E = own.copy()
    for c in E.columns:
        E[c] = E[c].fillna(pooled)
    return E, fb


def forward_inputs_wide(panel: pd.DataFrame, cfg: Config):
    cal = panel.index.get_level_values(DATE)
    ext = pd.date_range(cal.min(), cal.max() + pd.DateOffset(months=cfg.max_horizon), freq="MS")
    Aw, Ew, FB = {}, {}, {}
    for name in FWD_SERIES:
        A = panel[name].unstack(DISTRICT).sort_index().reindex(ext)
        Aw[name] = A
        Ew[name], FB[name] = expected_wide(A, cfg.fwd_min_years)
    return Aw, Ew, FB


def _blend(A: pd.DataFrame, E: pd.DataFrame, h: int, k: int) -> pd.DataFrame:
    j = h - k
    if j <= 0:
        return A.shift(-j).fillna(E.shift(-j))
    return E.shift(-j)


def _roll(frames: List[pd.DataFrame], how: str, min_periods: int) -> pd.DataFrame:
    arr = np.stack([f.to_numpy(dtype=float) for f in frames])
    cnt = np.sum(~np.isnan(arr), axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        val = np.nansum(arr, axis=0) if how == "sum" else np.nanmean(arr, axis=0)
    return pd.DataFrame(np.where(cnt >= min_periods, val, np.nan), index=frames[0].index, columns=frames[0].columns)


def forward_block(Aw: Dict[str, pd.DataFrame], Ew: Dict[str, pd.DataFrame], h: int) -> Dict[str, pd.DataFrame]:
    b = lambda name, k: _blend(Aw[name], Ew[name], h, k)
    thi = lambda k: 0.8 * b("temp_avg", k) + b("humid", k) / 100.0 * (b("temp_avg", k) - 14.4) + 46.4
    out = {
        "temp_avg_t": b("temp_avg", 0), "temp_max_t": b("temp_max", 0), "temp_min_t": b("temp_min", 0),
        "rain_t": b("rain", 0), "humid_t": b("humid", 0), "thi_t": thi(0),
        "temp_max_3m": _roll([b("temp_max", k) for k in range(3)], "mean", 2),
        "rain_3m": _roll([b("rain", k) for k in range(3)], "sum", 2),
        "rain_6m": _roll([b("rain", k) for k in range(6)], "sum", 3),
        "thi_3m": _roll([thi(k) for k in range(3)], "mean", 2),
        "fat_t": b("fat_pct", 0), "snf_t": b("snf_pct", 0),
        "fat_ch3": b("fat_pct", 0) - b("fat_pct", 3), "snf_ch3": b("snf_pct", 0) - b("snf_pct", 3),
    }
    out["fat_snf_ratio"] = out["fat_t"] / out["snf_t"].clip(lower=0.1)
    return out


# ----------------------------------------------------------------- intervention flags
def swing_windows(panel: pd.DataFrame, cfg: Config) -> Dict:
    r = panel["rate"].unstack(DISTRICT).sort_index()
    win = r.loc[cfg.swing_start:cfg.swing_end_search]
    full = [c for c in win.columns if win[c].notna().all()]
    if not full:
        return {}
    S = r[full].mean(axis=1).loc[:cfg.swing_end_search]
    peak = S.loc[cfg.swing_peak_search[0]:cfg.swing_peak_search[1]].idxmax()
    after = S.loc[peak:]
    trough = after.idxmin() if len(after) else peak
    run_up = (pd.Timestamp(cfg.swing_start), peak)
    crash = (peak + pd.DateOffset(months=1), trough) if trough > peak else (peak, peak)
    src = {"run_up": "found from data", "crash": "found from data"}
    if cfg.run_up_override:
        run_up, src["run_up"] = (pd.Timestamp(cfg.run_up_override[0]), pd.Timestamp(cfg.run_up_override[1])), "business override"
    if cfg.crash_override:
        crash, src["crash"] = (pd.Timestamp(cfg.crash_override[0]), pd.Timestamp(cfg.crash_override[1])), "business override"
    S_all = r[full].mean(axis=1)
    return {
        "run_up": run_up, "crash": crash, "reference_districts": full,
        "peak_rate": float(S[peak]), "trough_rate": float(S[trough]),
        "start_rate": float(S_all.get(run_up[0], np.nan)), "ref_series": S_all, "source": src,
    }


def intervention_table(panels: Dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    rows = []
    for mt, p in panels.items():
        w = swing_windows(p, cfg)
        if not w:
            continue
        S = w["ref_series"]
        step = S.diff()
        def avg(lo, hi):
            x = step.loc[lo:hi].dropna()
            return float(x.mean()) if len(x) else 0.0
        for name, label in (("run_up", "Run-up"), ("crash", "Crash")):
            lo, hi = w[name]
            before = S.loc[:lo - pd.DateOffset(months=1)].dropna()
            r0 = float(before.iloc[-1]) if len(before) else float(S.get(lo, np.nan))
            r1 = float(S.get(hi, np.nan))
            rows.append({
                "Milk Type": mt.title(), "Window": label, "Start": lo.date(), "End": hi.date(),
                "Months": len(pd.date_range(lo, hi, freq="MS")), "Dates source": w["source"][name],
                "Reference rate before (Rs/kg TS)": r0, "Reference rate at end (Rs/kg TS)": r1,
                "Move (Rs/kg TS)": r1 - r0, "Move %": (r1 / r0 - 1) * 100 if r0 else 0.0,
                "Avg monthly change in window": avg(lo, hi),
                "Avg monthly change outside both windows": float(step.dropna().mean()),
                "Reference districts": ", ".join(w["reference_districts"][:5]) + f" (+{len(w['reference_districts'])-5} more)",
            })
    return pd.DataFrame(rows)


def flag_shares(win: Dict, ext: pd.DatetimeIndex, h: int) -> Dict[str, pd.Series]:
    # Intervention flags completely removed
    return {}


def movement_category(run_up_share: float, crash_share: float, price_change: float | None) -> str:
    if price_change is None or pd.isna(price_change):
        return "missing_change"
    if abs(price_change) <= 1e-6:
        return "flat"
    if run_up_share > 0 and price_change > 0:
        return "run_up_rise"
    if crash_share > 0 and price_change < 0:
        return "crash_fall"
    if price_change > 0:
        return "recovery_or_rise"
    if price_change < 0:
        return "decline"
    return "flat"


def confidence_score(run_up_share: float, crash_share: float, prior_rate: float | None, price_change: float | None) -> float:
    score = 50.0
    if run_up_share > 0 or crash_share > 0:
        score += 25.0
    if prior_rate is not None and not pd.isna(prior_rate):
        score += 15.0
    if price_change is not None and not pd.isna(price_change):
        score += 10.0
    return round(float(np.clip(score, 0.0, 100.0)), 1)


def build_run_up_crash_analysis(panels: Dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    start = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2024-12-01")
    rows = []
    for mt, panel in panels.items():
        win = swing_windows(panel, cfg)
        if not win:
            continue
        for (district, month), rate in panel["rate"].dropna().items():
            if month < start or month > end:
                continue
            prev_key = (district, month - pd.DateOffset(months=1))
            prev_rate = panel.loc[prev_key, "rate"] if prev_key in panel.index else np.nan
            price_change = np.nan if pd.isna(prev_rate) else float(rate - prev_rate)
            pct_change = np.nan if pd.isna(prev_rate) or prev_rate == 0 else float((rate / prev_rate - 1.0) * 100.0)
            run_up_share = 1.0 if win["run_up"][0] <= month <= win["run_up"][1] else 0.0
            crash_share = 1.0 if win["crash"][0] < win["crash"][1] and win["crash"][0] <= month <= win["crash"][1] else 0.0
            categ = movement_category(run_up_share, crash_share, price_change)
            ref_rate = float(win["ref_series"].get(month, np.nan)) if month in win["ref_series"].index else np.nan
            rows.append({
                "month": month,
                "district": district,
                "milk_type": mt,
                "rate_rs_per_kg_ts": float(rate),
                "prior_month_rate_rs_per_kg_ts": float(prev_rate) if pd.notna(prev_rate) else np.nan,
                "price_change_rs_per_kg_ts": float(price_change) if pd.notna(price_change) else np.nan,
                "pct_change_vs_previous_month": float(pct_change) if pd.notna(pct_change) else np.nan,
                "run_up_share": round(float(run_up_share), 6),
                "crash_share": round(float(crash_share), 6),
                "movement_category": categ,
                "reference_rate_rs_per_kg_ts": float(ref_rate) if pd.notna(ref_rate) else np.nan,
                "window_label": "Run-up" if run_up_share > 0 else "Crash" if crash_share > 0 else "Outside swing",
                "confidence_score": confidence_score(run_up_share, crash_share, prev_rate, price_change),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["milk_type", "district", "month"]).reset_index(drop=True)


def build_training_table(panel: pd.DataFrame, market: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Stack horizons h = 1..max_horizon: one row per (district, origin month, horizon)."""
    d = origin_features(panel, market)
    r = d["rate"]
    g = r.groupby(level=DISTRICT)
    Aw, Ew, _ = forward_inputs_wide(panel, cfg)
    win = swing_windows(panel, cfg)
    ext = Aw["rain"].index
    frames = []
    for h in range(1, cfg.max_horizon + 1):
        x = d.copy()
        x["horizon"] = h
        x["origin"] = x.index.get_level_values(DATE)
        if cfg.project_market:
            pm = projected_market_features(market, h, cfg.proj_window).reindex(x["origin"])
            for c in pm.columns:
                x[c] = pm[c].to_numpy()
        x["target_month"] = x["origin"] + pd.DateOffset(months=h)
        x["rate_actual"] = g.shift(-h)
        x["target"] = x["rate_actual"] - x["rate_t"]
        x["ly_change"] = g.shift(12 - h) - g.shift(12)
        for k, v in calendar_features(x["target_month"]).items():
            x[k] = v
        if cfg.forward_inputs:
            for name, W in forward_block(Aw, Ew, h).items():
                x[name] = _long(W, x.index)
        for name, sh in flag_shares(win, ext, h).items():
            x[name] = sh.reindex(x["origin"]).to_numpy()
        frames.append(x)
    t = pd.concat(frames).reset_index().rename(columns={DATE: "origin_idx"}).drop(columns="origin_idx")
    t = t[t["rate_t"].notna() & t["diff_1"].notna()]
    if cfg.train_from_target:
        t = t[t["target_month"] >= pd.Timestamp(cfg.train_from_target)]
    return t.sort_values([DISTRICT, "origin", "horizon"]).reset_index(drop=True)


WEATHER_LIMITS = {"rain": (0.0, None), "humid": (0.0, 100.0)}


def scenario_table(panel: pd.DataFrame, cfg: Config, table: pd.DataFrame, origin: pd.Timestamp,
                   shifts: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    Aw, Ew, _ = forward_inputs_wide(panel, cfg)
    Es = {k: v.copy() for k, v in Ew.items()}
    for name, S in shifts.items():
        S = S.reindex(index=Es[name].index, columns=Es[name].columns).fillna(0.0)
        S.loc[S.index <= origin] = 0.0
        E = Es[name] + S
        lo, hi = WEATHER_LIMITS.get(name, (None, None))
        Es[name] = E.clip(lower=lo, upper=hi)
    out = table.copy()
    for h in range(1, cfg.max_horizon + 1):
        m = (out["origin"] == origin) & (out["horizon"] == h)
        if not m.any():
            continue
        dist = out.loc[m, DISTRICT]
        for name, W in forward_block(Aw, Es, h).items():
            out.loc[m, name] = W.loc[origin].reindex(dist).to_numpy()
    return out


def assert_no_leakage(train: pd.DataFrame, origin: pd.Timestamp) -> None:
    if len(train) and train["target_month"].max() > origin:
        raise AssertionError(f"leakage: training reaches {train['target_month'].max():%b %Y} > origin {origin:%b %Y}")


def forecastable_districts(panel: pd.DataFrame, origin: pd.Timestamp, n_months: int) -> List[str]:
    window = pd.date_range(end=origin, periods=n_months, freq="MS")
    r = panel["rate"].unstack(DATE).reindex(columns=window)
    return sorted(r.index[r.notna().all(axis=1)])


# ------------------------------------------------------------- Weather Scenarios & Bands
WEATHER = list(WEATHER_COLS)
MIN_SEASON_N = 20
SEASON_OF_MONTH = {
    12: "Winter (Dec-Feb)", 1: "Winter (Dec-Feb)", 2: "Winter (Dec-Feb)",
    3: "Hot-dry (Mar-Apr)", 4: "Hot-dry (Mar-Apr)",
    5: "Monsoon onset (May-Jun)", 6: "Monsoon onset (May-Jun)",
    7: "Monsoon (Jul-Sep)", 8: "Monsoon (Jul-Sep)", 9: "Monsoon (Jul-Sep)",
    10: "Post-monsoon (Oct-Nov)", 11: "Post-monsoon (Oct-Nov)"
}
D, M = "District", "Month"


def weather_table(raw: pd.DataFrame) -> pd.DataFrame:
    w = raw.groupby([DISTRICT, DATE])[WEATHER].first()
    w.index = w.index.set_names([D, M])
    return w.sort_index()


def _wide(w: pd.DataFrame, var: str, horizon: int, last: pd.Timestamp) -> pd.DataFrame:
    A = w[var].unstack(D).sort_index()
    return A.reindex(pd.date_range(A.index.min(), last + pd.DateOffset(months=horizon), freq="MS"))


def _to_long(W: pd.DataFrame, name: str) -> pd.Series:
    L = W.rename_axis(index=M, columns=D).reset_index().melt(id_vars=M, var_name=D, value_name=name)
    return L.set_index([D, M])[name].sort_index()


def expected_long(w: pd.DataFrame, origin: pd.Timestamp, horizon: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    known = w[w.index.get_level_values(M) <= origin]
    cols, ny, fb = {}, None, None
    for v in WEATHER:
        A = _wide(known, v, horizon, origin)
        E, F = expected_wide(A, 1)
        cols[v] = _to_long(E, v)
        if ny is None:
            ny = A.notna().astype(float).groupby(A.index.month).transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
            fb = F
    exp = pd.DataFrame(cols)
    exp["n_years"] = _to_long(ny, "n_years").reindex(exp.index)
    exp["source"] = np.where(_to_long(fb.astype(float), "fb").reindex(exp.index).fillna(1.0) > 0, "other districts", "own years")
    mths = exp.index.get_level_values(M)
    return exp[mths <= origin], exp[mths > origin]


def backtest_errors(w: pd.DataFrame, exp_known: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    a = w[w.index.get_level_values(M) <= origin][WEATHER]
    e = exp_known.reindex(a.index)
    err = (a - e[WEATHER]).dropna(how="all")
    err["n_years"] = e["n_years"].reindex(err.index)
    err["source"] = e["source"].reindex(err.index)
    err["season"] = [SEASON_OF_MONTH[m.month] for m in err.index.get_level_values(M)]
    return err


def band_table(err: pd.DataFrame, level: float = 95.0, by: str | None = None) -> pd.DataFrame:
    lo_q, hi_q = (100 - level) / 200, 1 - (100 - level) / 200
    if by is None:
        groups = [("ALL", err)]
    elif by == "season":
        groups = list(err.groupby("season"))
    else:
        groups = list(err.groupby(level=by))
    rows = []
    for key, g in groups:
        for v in WEATHER:
            e = g[v].dropna()
            if len(e) < 5:
                continue
            rows.append({
                "group": key, "variable": v, "n": len(e), "bias": e.mean(), "MAE": e.abs().mean(), "std": e.std(),
                "lo": e.quantile(lo_q), "hi": e.quantile(hi_q), "level_%": level,
                "within_1std_%": (e.abs() <= e.std()).mean() * 100
            })
    return pd.DataFrame(rows)


def season_bands(err: pd.DataFrame, level: float = 95.0) -> pd.DataFrame:
    allyear = band_table(err, level).set_index("variable")
    rows = []
    for season in dict.fromkeys(SEASON_OF_MONTH.values()):
        g = err[err["season"] == season]
        bt = band_table(g, level).set_index("variable") if len(g) else pd.DataFrame()
        for v in WEATHER:
            use_own = (v in bt.index) and (bt.loc[v, "n"] >= MIN_SEASON_N)
            src = bt.loc[v] if use_own else allyear.loc[v]
            rows.append({
                "season": season, "variable": v, "n": int(src["n"]), "bias": src["bias"], "MAE": src["MAE"], "std": src["std"],
                "lo": src["lo"], "hi": src["hi"], "level_%": level, "basis": "own season" if use_own else "all-year"
            })
    return pd.DataFrame(rows)


def apply_band(expected: float, var: str, edge: float) -> float:
    lo, hi = WEATHER_LIMITS.get(var, (None, None))
    x = expected + edge
    if lo is not None:
        x = max(x, lo)
    if hi is not None:
        x = min(x, hi)
    return x


def scenario_frame(exp_future: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    b = bands.set_index(["season", "variable"])
    rows = []
    for (dist, month), r in exp_future.iterrows():
        season = SEASON_OF_MONTH[month.month]
        for v in WEATHER:
            e = r[v]
            if pd.isna(e):
                continue
            rows.append({
                D: dist, M: month, "variable": v, "season": season, "n_years": r["n_years"], "source": r["source"],
                "low": apply_band(e, v, b.loc[(season, v), "lo"]), "expected": e,
                "high": apply_band(e, v, b.loc[(season, v), "hi"])
            })
    return pd.DataFrame(rows)


def scenario_shifts(index: pd.DatetimeIndex, districts, bands: pd.DataFrame, which: str) -> Dict[str, pd.DataFrame]:
    b = bands.set_index(["season", "variable"])
    out = {}
    for v in WEATHER:
        col = np.array([b.loc[(SEASON_OF_MONTH[m.month], v), which] for m in index], dtype=float)
        out[v] = pd.DataFrame(np.repeat(col[:, None], len(districts), axis=1), index=index, columns=list(districts))
    return out


def weather_bands_at(raw: pd.DataFrame, origin: pd.Timestamp, level: float = 95.0, horizon: int = 6):
    w = weather_table(raw)
    exp_known, exp_future = expected_long(w, origin, horizon)
    err = backtest_errors(w, exp_known, origin)
    bands = season_bands(err, level)
    return bands, scenario_frame(exp_future, bands), err, exp_future


def write_feature_dictionary(path: Path) -> None:
    rows = []
    for name in ALL_FEATURES:
        grp, kind, text = FEATURE_INFO[name]
        rows.append({
            "Feature": name, "Group": grp, "How known at forecast time": kind, "Definition": text,
            "Used in COW": "Yes" if name in FEATURES_BY_TYPE["COW"] else "-",
            "Used in BUFFALO": "Yes" if name in FEATURES_BY_TYPE["BUFFALO"] else "-",
            "Set": "FLAG (intervention)" if name in FLAG_FEATURES else "CORE (both)" if name in CORE else ("COW extra" if name in COW_EXTRA and name not in BUFFALO_EXTRA else "BUFFALO extra")
        })
    df = pd.DataFrame(rows)
    summary = pd.DataFrame({
        "Milk type": ["COW", "BUFFALO"],
        "Inputs used now": [len(FEATURES_BY_TYPE["COW"]), len(FEATURES_BY_TYPE["BUFFALO"])],
        "Inputs in full set": [len(FULL_FEATURES_BY_TYPE["COW"]), len(FULL_FEATURES_BY_TYPE["BUFFALO"])],
        "Core (shared)": [len(CORE)] * 2,
        "Milk-type extras": [len(COW_EXTRA), len(BUFFALO_EXTRA)],
    })
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        df.to_excel(xw, sheet_name="Feature_Dictionary", index=False)


def main(cfg: Config, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    write_feature_dictionary(outdir / "Feature_Dictionary.xlsx")
    raw = load_raw(cfg)
    panels, market = build_panels(raw, cfg)

    for mt in MILK_TYPES:
        table = build_training_table(panels[mt], market, cfg)
        table.to_csv(outdir / f"training_table_{mt.lower()}.csv", index=False)
        w = swing_windows(panels[mt], cfg)
        if w:
            print(f"\n{mt} price trajectory: run-up {w['run_up'][0]:%b %Y}..{w['run_up'][1]:%b %Y} (peak {w['peak_rate']:.1f})")
        print(f"{mt}: training rows: {int(table['target'].notna().sum())} with known target, {len(table)} incl. forecast rows")
        log.info("%s: %d model inputs", mt, len(FEATURES_BY_TYPE[mt]))
        p = panels[mt]
        print(f"\n{mt} - Rs/kg TS Summary across Karnataka 31 districts:")
        print(p["rate"].groupby(level=DISTRICT).agg(["count", "min", "mean", "max"]).round(1).head(10).to_string())

    iv = intervention_table(panels, cfg)
    iv.to_csv(outdir / "intervention_windows.csv", index=False)
    print("\nINTERVENTION WINDOWS:")
    print(iv.to_string(index=False))

    analysis = build_run_up_crash_analysis(panels, cfg)
    if not analysis.empty:
        analysis.to_csv(outdir / "Run_Up_Crash_Analysis_2022_2024.csv", index=False)
        with pd.ExcelWriter(outdir / "Run_Up_Crash_Analysis_2022_2024.xlsx", engine="openpyxl") as xw:
            analysis.to_excel(xw, sheet_name="Row_Level_Analysis", index=False)
            iv.to_excel(xw, sheet_name="Window_Summary", index=False)

    print("\nStep 1 Feature Generation & Panel Build Completed Successfully.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step 1: data, features, panels, and intervention analysis")
    ap.add_argument("--data", default=Config.data_csv)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--no-forward", action="store_true")
    ap.add_argument("--no-flags", action="store_true")
    ap.add_argument("--keep-2021", action="store_true")
    add_swing_args(ap)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = config_from_args(a)
    out = resolve(a.outdir)
    main(cfg, out)
