"""
step3_final_forecast.py - FILE 3 of 4: Final Milk Price Forecast (Point + Weather Band + Uncertainty Band)
(Adapted for Karnataka State Dairy Price Forecasting)
===========================================================================================================
Origin: Last actual month (2026-10-01), forecast Nov 2026 to Apr 2027 (N+1 to N+6).
Models:
  - COW     : "Cow Ensemble v2" (cow_model.py)
  - BUFFALO : "RecursiveX Blend" (or Direct XGB)
Weather bands:
  - Runs models under LOW, EXPECTED, and HIGH weather conditions.
  - Combines with backtest error bands to give 80% total prediction intervals.
Outputs:
  - Final_Forecast.xlsx
  - feature_importance_cow.png / feature_importance_buffalo.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import step1_data_features as s1
import step2_backtest as s2
import cow_model

log = logging.getLogger("milk")

FINAL_METHOD = "RecursiveX Blend"
FINAL_METHOD_COW = cow_model.METHOD

SCN_LOW, SCN_HIGH = "WEATHER LOW", "WEATHER HIGH"


def model_bands(detail: pd.DataFrame, origin: pd.Timestamp | None = None) -> Dict:
    d = detail.copy()
    d["target_month"] = pd.to_datetime(d["target_month"])
    if origin is not None:
        d = d[d["target_month"] <= origin]
    out: Dict = {}
    for (mt, h), g in d.groupby(["milk_type", "horizon"]):
        methods = s2.LEARNED + (cow_model.COW_COLUMNS if mt == "COW" else [])
        for m in methods:
            if m not in g.columns:
                continue
            e = (g["rate_actual"] - g[m]).dropna().to_numpy()
            if len(e) >= 5:
                out.setdefault(mt, {}).setdefault(m, {})[str(int(h))] = [float(np.percentile(e, 10)), float(np.percentile(e, 90)), len(e)]
    return out


def feature_importance(mt: str, fits: Dict, feats: List[str], H: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r, x, tr = fits[("RecursiveX", 1)]
    rec_r_signed = s2.ridge_std_coef(r, feats)
    rec_r, rec_x = s2.pct(rec_r_signed), s2.pct(s2.shap_importance(x, tr[feats]))
    dr, dx, dsign = [], [], []
    for h in range(1, H + 1):
        rh, xh, th = fits[("Direct", h)]
        s = s2.ridge_std_coef(rh, feats)
        dsign.append(s)
        dr.append(s2.pct(s))
        dx.append(s2.pct(s2.shap_importance(xh, th[feats])))
    dir_r, dir_x = sum(dr) / H, sum(dx) / H
    out = pd.DataFrame({
        "Feature": feats,
        "Group": [s1.FEATURE_GROUP[f] for f in feats],
        "How known at forecast time": [s1.FEATURE_KIND[f] for f in feats],
        "Recursive Ridge %": rec_r.to_numpy(), "Recursive XGB %": rec_x.to_numpy(),
        "Direct Ridge % (avg h1-6)": dir_r.to_numpy(), "Direct XGB % (avg h1-6)": dir_x.to_numpy(),
    })
    out["Overall importance %"] = out[["Recursive Ridge %", "Recursive XGB %", "Direct Ridge % (avg h1-6)", "Direct XGB % (avg h1-6)"]].mean(axis=1)
    sign = np.sign(rec_r_signed.to_numpy() + sum(dsign).to_numpy() / H)
    out["Effect on forecast (Ridge)"] = np.where(sign > 0, "rate rises when input rises", np.where(sign < 0, "rate falls when input rises", "-"))
    out["Meaning"] = [s1.FEATURE_INFO[f][2] for f in feats]
    out = out.sort_values("Overall importance %", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    out.insert(0, "Milk Type", mt.title())
    grp = (out.groupby("Group")[["Overall importance %", "Recursive Ridge %", "Recursive XGB %",
                                 "Direct Ridge % (avg h1-6)", "Direct XGB % (avg h1-6)"]].sum()
           .join(out.groupby("Group")["Feature"].count().rename("Inputs")).sort_values("Overall importance %", ascending=False)
           .reset_index())
    grp.insert(0, "Milk Type", mt.title())
    return out, grp


def importance_chart(mt: str, imp: pd.DataFrame, grp: pd.DataFrame, path: Path, top: int = 20) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = list(dict.fromkeys(s1.FEATURE_GROUP.values()))
        cmap = {g: plt.cm.tab20(i % 20) for i, g in enumerate(groups)}
        fig, ax = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1.5, 1]})
        t = imp.head(top).iloc[::-1]
        ax[0].barh(t["Feature"], t["Overall importance %"], color=[cmap.get(g, "steelblue") for g in t["Group"]])
        ax[0].set_title(f"{mt.title()} - top {top} inputs (overall importance %)")
        ax[0].set_xlabel("share of total importance (%)")
        for i, v in enumerate(t["Overall importance %"]):
            ax[0].text(v + 0.05, i, f"{v:.1f}", va="center", fontsize=8)
        g = grp.iloc[::-1]
        ax[1].barh(g["Group"], g["Overall importance %"], color=[cmap.get(x, "steelblue") for x in g["Group"]])
        ax[1].set_title(f"{mt.title()} - importance by group (%)")
        for i, (v, n) in enumerate(zip(g["Overall importance %"], g["Inputs"])):
            ax[1].text(v + 0.2, i, f"{v:.1f} ({int(n)} inputs)", va="center", fontsize=8)
        ax[1].set_xlim(0, g["Overall importance %"].max() * 1.35)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
    except Exception as exc:
        log.warning("Could not generate chart: %s", exc)


def final_for(mt: str, panel: pd.DataFrame, table: pd.DataFrame, cfg: s1.Config, meta: Dict, recx: Dict,
              origin_override: pd.Timestamp | None = None, scenarios: Dict[str, pd.DataFrame] | None = None):
    spec = recx["spec"]
    feats = spec["all"]
    origin = origin_override if origin_override is not None else panel["rate"].dropna().index.get_level_values(s1.DATE).max()
    districts = s1.forecastable_districts(panel, origin, cfg.forecast_min_months)
    method = meta["best_method"][mt]
    log.info("%s: origin %s | %d inputs | districts: %d | method: %s", mt, f"{origin:%b %Y}", len(feats), len(districts), method)
    d, fits = s2.forecast_origin(panel, table, origin, cfg.max_horizon, cfg, recx, districts, keep_fits=True,
                                 scenarios=scenarios, cow_extra=(mt == "COW"))

    out = pd.DataFrame({
        "Milk Type": mt.title(), "District": d[s1.DISTRICT], "Horizon (months)": d["horizon"],
        "Forecast Month": d["target_month"].dt.date, "Last Actual (Rs/kg TS)": d["rate_t"], "Forecast (Rs/kg TS)": d[method]
    })
    lo_hi = [meta["bands"].get(mt, {}).get(method, {}).get(str(int(h)), [np.nan, np.nan, np.nan]) for h in d["horizon"]]
    lo = np.minimum([x[0] for x in lo_hi], 0.0)
    hi = np.maximum([x[1] for x in lo_hi], 0.0)
    out["Lower 80% (Rs/kg TS)"], out["Upper 80% (Rs/kg TS)"] = out["Forecast (Rs/kg TS)"] + lo, out["Forecast (Rs/kg TS)"] + hi

    if scenarios and f"{method} | {SCN_LOW}" in d.columns:
        f0 = d[method].to_numpy()
        f_lo, f_hi = d[f"{method} | {SCN_LOW}"].to_numpy(), d[f"{method} | {SCN_HIGH}"].to_numpy()
        stack = np.vstack([f0, f_lo, f_hi])
        out["Forecast, LOW weather (Rs/kg TS)"], out["Forecast, HIGH weather (Rs/kg TS)"] = f_lo, f_hi
        out["Weather band low (Rs/kg TS)"], out["Weather band high (Rs/kg TS)"] = stack.min(axis=0), stack.max(axis=0)
        out["Weather band width (Rs/kg TS)"] = out["Weather band high (Rs/kg TS)"] - out["Weather band low (Rs/kg TS)"]
        out["TOTAL band low (weather + model error)"] = out["Weather band low (Rs/kg TS)"] + lo
        out["TOTAL band high (weather + model error)"] = out["Weather band high (Rs/kg TS)"] + hi

    out["Method"] = method
    if origin_override is not None and "rate_actual" in d.columns:
        out["Actual (Rs/kg TS)"] = d["rate_actual"].to_numpy()
        out["Abs error (Rs/kg TS)"] = (out["Forecast (Rs/kg TS)"] - out["Actual (Rs/kg TS)"]).abs()
    out["Status"] = np.where(d["horizon"] <= cfg.short_horizon, "Short-term (CBE)", "INDICATIVE - outside short-term horizon")

    for m in s2.ALL_METHODS + [c for c in cow_model.COW_COLUMNS if c in d.columns]:
        out[m] = d[m].to_numpy()

    eq = [s2.equation_frame(mt, f"Direct Ridge ({len(feats)} inputs)", fits[("Direct", h)][0], feats, h) for h in range(1, cfg.max_horizon + 1)]
    rx = fits[("RecursiveX", 1)]
    rec_eq = s2.equation_frame(mt, f"RecursiveX one-step Ridge ({len(feats)} inputs)", rx[0], feats, 1)
    sh = [pd.DataFrame({"Milk Type": mt.title(), "Model": "Direct XGB", "Horizon": h, "Feature": feats,
                        "Mean |SHAP|": s2.shap_importance(fits[("Direct", h)][1], fits[("Direct", h)][2][feats]).to_numpy()})
          for h in range(1, cfg.max_horizon + 1)]
    sh.append(pd.DataFrame({"Milk Type": mt.title(), "Model": "RecursiveX one-step XGB", "Horizon": 1, "Feature": feats,
                            "Mean |SHAP|": s2.shap_importance(rx[1], rx[2][feats]).to_numpy()}))
    imp, grp = feature_importance(mt, fits, feats, cfg.max_horizon)
    return out, pd.concat(eq), rec_eq, pd.concat(sh), imp, grp


def main(cfg: s1.Config, outdir: Path, override: Dict[str, str] | None = None, origin: str | None = None,
         band_level: float = 95.0, weather_band: bool = True) -> None:
    origin_ts = pd.Timestamp(origin) if origin else None
    meta_path = outdir / "backtest_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"best_method": {"BUFFALO": FINAL_METHOD, "COW": FINAL_METHOD_COW}, "bands": {}}

    override = {k: v for k, v in (override or {}).items() if v}
    for mt, m in override.items():
        meta["best_method"][mt] = m

    detail_path = outdir / "backtest_detail.csv"
    if detail_path.exists():
        meta["bands"] = model_bands(pd.read_csv(detail_path), origin_ts)

    raw = s1.load_raw(cfg)
    panels, market = s1.build_panels(raw, cfg)
    origin_eff = origin_ts if origin_ts is not None else raw.loc[raw["row_valid"], s1.DATE].max()

    wb = None
    if weather_band and cfg.forward_inputs:
        wb = s1.weather_bands_at(raw, origin_eff, band_level, cfg.max_horizon)

    fc, eq, req, sh, imps, grps = [], [], [], [], {}, {}
    for mt in s1.MILK_TYPES:
        table_path = outdir / f"training_table_{mt.lower()}.csv"
        table = pd.read_csv(table_path) if table_path.exists() else s1.build_training_table(panels[mt], market, cfg)
        table["origin"] = pd.to_datetime(table["origin"])
        table["target_month"] = pd.to_datetime(table["target_month"])
        spec = s1.spec_for(mt, forward=cfg.forward_inputs, flags=cfg.use_flags)
        recx = s2.prepare_recx(panels[mt], market, table, cfg, spec)

        scn = None
        if wb is not None:
            p = panels[mt]
            cal = p.index.get_level_values(s1.DATE)
            ext = pd.date_range(cal.min(), cal.max() + pd.DateOffset(months=cfg.max_horizon), freq="MS")
            dists = sorted(p.index.get_level_values(s1.DISTRICT).unique())
            o = origin_eff if origin_ts is not None else p["rate"].dropna().index.get_level_values(s1.DATE).max()
            scn = {
                SCN_LOW: s1.scenario_table(p, cfg, table, o, s1.scenario_shifts(ext, dists, wb[0], "lo")),
                SCN_HIGH: s1.scenario_table(p, cfg, table, o, s1.scenario_shifts(ext, dists, wb[0], "hi"))
            }

        a, b, c, e, imp, grp = final_for(mt, panels[mt], table, cfg, meta, recx, origin_ts, scn)
        fc.append(a), eq.append(b), req.append(c), sh.append(e)
        imps[mt], grps[mt] = imp, grp
        imp.to_csv(outdir / f"feature_importance_{mt.lower()}.csv", index=False)
        importance_chart(mt, imp, grp, outdir / f"feature_importance_{mt.lower()}.png")

    forecast = pd.concat(fc).sort_values(["Milk Type", "District", "Horizon (months)"]).reset_index(drop=True)
    cmp_cols = [f"{f} {k}" for k in ("Blend", "XGB", "Ridge") for f in ("Direct", "RecursiveX")]
    compare = forecast[["Milk Type", "District", "Horizon (months)", "Forecast Month", "Last Actual (Rs/kg TS)"] + cmp_cols]

    extra = {}
    if wb is not None:
        extra["Weather_Band"] = wb[1].rename(columns={"low": "Low", "expected": "Expected", "high": "High"})
        extra["Weather_Band_Rules"] = wb[0]
    extra["Intervention_Windows"] = s1.intervention_table(panels, cfg)

    sheets = {
        "Forecast": forecast, "Compare_Direct_vs_RecX": compare, **extra,
        "Feature_Importance_COW": imps["COW"], "Feature_Importance_BUFFALO": imps["BUFFALO"],
        "Importance_by_Group": pd.concat([grps["COW"], grps["BUFFALO"]]),
        "Ridge_Equation_Direct": pd.concat(eq), "Recursive_Equation": pd.concat(req),
        "XGB_SHAP_Top15": pd.concat(sh).head(30),
    }

    fname = f"Forecast_from_{origin_ts:%b%Y}_heldout.xlsx" if origin_ts is not None else "Final_Forecast.xlsx"
    s2.write_excel(outdir / fname, sheets)

    pd.set_option("display.width", 220)
    cols = ["Milk Type", "District", "Forecast Month", "Last Actual (Rs/kg TS)", "Forecast (Rs/kg TS)",
            "TOTAL band low (weather + model error)", "TOTAL band high (weather + model error)", "Method"]
    print(f"\nFINAL FORECAST (saved to {outdir / fname}):")
    print(forecast[[c for c in cols if c in forecast.columns]].head(20).round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Final milk price forecast Nov 2026 - Apr 2027, Rs/kg TS")
    ap.add_argument("--data", default=s1.Config.data_csv)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--buffalo-method", default=FINAL_METHOD)
    ap.add_argument("--cow-method", default=FINAL_METHOD_COW)
    ap.add_argument("--origin", default=None)
    ap.add_argument("--band-level", type=float, default=95.0)
    ap.add_argument("--no-weather-band", action="store_true")
    s1.add_swing_args(ap)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ov = {"BUFFALO": a.buffalo_method, "COW": a.cow_method}
    main(s1.config_from_args(a), s1.resolve(a.outdir), ov, a.origin, a.band_level, not a.no_weather_band)
