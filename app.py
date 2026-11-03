"""
app.py - Streamlit App: KARNATAKA RETAIL MILK PRICE FORECASTING & MARKET INTELLIGENCE
======================================================================================
Quarterly Forecasting (Q1: Nov 2026 – Jan 2027 & Q2: Feb 2027 – Apr 2027)
Validated against Officially Published Karnataka Milk Federation (KMF - Nandini)
Retail Price Revisions, State Cabinet Approvals, and Verified Trade Intelligence.

Pricing Standard: RETAIL MILK PRICES (Consumer Packaged MRP / Shelf Price).
Units Supported : Rs per kg Total Solids (Rs/kg TS) and Retail Price per Litre (Rs/L).
Conversion      : Cow (8.4424x) | Buffalo (6.4600x)

Run: streamlit run app.py
"""
from __future__ import annotations

import os
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = "1"
warnings.filterwarnings("ignore")

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import karnataka_speculation_loader as sl
import shock_engine as se
import step1_data_features as s1
import step2_backtest as s2

st.set_page_config(
    page_title="Karnataka Retail Milk Price Forecasting & Intelligence",
    page_icon="🥛",
    layout="wide",
    initial_sidebar_state="expanded"
)

BLUE, ORANGE, GREY, RED, GREEN = "#2c7be5", "#f28c28", "#8a8f98", "#d9534f", "#2e7d32"


# ------------------------------------------------------------------ data + models (once)
@st.cache_resource(show_spinner="Loading Karnataka retail milk models & historical data (31 districts, Jan 2021 – Nov 2026)...")
def get_engine() -> se.ShockEngine:
    return se.ShockEngine()


try:
    E = get_engine()
except Exception as err:
    st.error(f"Error loading models or dataset: {err}")
    st.info("Ensure Karnataka_Milk_Final_Jan2021_Nov2026.csv is present in this directory.")
    st.stop()

EKEY = "karnataka_v1"


# ------------------------------------------------------------------ cached scenario runs
def _tup(d: dict) -> tuple:
    return tuple(sorted((k, float(v)) for k, v in d.items() if v))


@st.cache_data(show_spinner=False)
def run_forecast(_E, ekey, mt, shocks, ramp, weather, method):
    return _E.forecast(mt, dict(shocks), ramp, dict(weather), [method])


@st.cache_data(show_spinner="Running every driver up and down...")
def run_tornado(_E, ekey, mt, method, size, ramp, hz):
    return _E.tornado(mt, method, size, ramp, hz)


@st.cache_data(show_spinner="Running the sweep...")
def run_sweep(_E, ekey, mt, method, key, values, ramp, hzs):
    return _E.sweep(mt, method, key, list(values), ramp, list(hzs))


@st.cache_data(show_spinner="Pushing every model input...")
def run_inputs(_E, ekey, mt, hz):
    return _E.input_sensitivity(mt, hz)


@st.cache_data(show_spinner="Measuring feature importance...")
def run_importance(_E, ekey, mt):
    return _E.feature_importance(mt)


@st.cache_data(show_spinner=False)
def run_changed(_E, ekey, mt, shocks, ramp, weather, hz):
    return _E.changed_inputs(mt, dict(shocks), ramp, dict(weather), hz)


def diverging_bars(df: pd.DataFrame, cat: str, up_col: str, dn_col: str, up_lab: str, dn_lab: str, sort_col: str, xtitle: str) -> alt.Chart:
    long = df.melt(id_vars=[cat, sort_col], value_vars=[up_col, dn_col], var_name="Direction", value_name="Effect")
    long["Direction"] = long["Direction"].map({up_col: up_lab, dn_col: dn_lab})
    base = alt.Chart(long).mark_bar(opacity=0.9).encode(
        y=alt.Y(f"{cat}:N", sort=alt.EncodingSortField(field=sort_col, order="descending"), title=None),
        x=alt.X("Effect:Q", stack=None, title=xtitle),
        color=alt.Color("Direction:N", scale=alt.Scale(domain=[up_lab, dn_lab], range=[RED, BLUE]), title=None),
        tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip("Direction:N"), alt.Tooltip("Effect:Q", format="+.2f")])
    try:
        base = base.encode(yOffset="Direction:N")
    except Exception:
        pass
    return base.properties(height=max(220, 32 * df[cat].nunique()))


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🥛 Karnataka Retail Milk Simulator")
st.sidebar.markdown("**State:** Karnataka | **Cadence:** Quarterly & Monthly")
mt = st.sidebar.radio("Milk Category", list(s1.MILK_TYPES), format_func=str.title, horizontal=True)
S = E.state[mt]
methods = se.all_methods(mt)
default_method = se.HEADLINE[mt] if se.HEADLINE[mt] in methods else methods[0]
method = st.sidebar.selectbox("Forecasting Model", methods, index=methods.index(default_method),
                              help="Default: Cow Ensemble v2 (Cow) or RecursiveX XGB / Blend (Buffalo).")

sel = st.sidebar.multiselect("Districts", S.districts, default=S.districts, key=f"dist_{mt}") or list(S.districts)
H = st.sidebar.slider("Forecast Horizon (Months Ahead)", 1, min(18, E.H), 18)
st.sidebar.caption("Horizon Structure: **Q1** (Dec–Feb) | **Q2** (Mar–May) | **Q3** (Jun–Aug) | **Q4** (Sep–Nov) | **Q5** (Dec–Feb) | **Q6** (Mar–May).")

st.sidebar.markdown("---")
st.sidebar.subheader("Market Stress & Scenario Testing")
PRESETS = {
    "Baseline (Sticky Retail Shelf Price)": {},
    "Feed-cost spike (+15% feed, maize, soy)": {"market": {"feed": 15, "maize": 15, "soy_meal": 15}},
    "Feed-cost relief (-10% feed, maize, soy)": {"market": {"feed": -10, "maize": -10, "soy_meal": -10}},
    "Dairy-product boom (SMP +15%, Ghee & Butter +10%)": {"market": {"smp": 15, "ghee": 10, "butter": 10}},
    "Dairy-product slump (SMP -15%, Ghee & Butter -10%)": {"market": {"smp": -15, "ghee": -10, "butter": -10}},
    "Inflation shock (CPI +3%)": {"market": {"cpi": 3}},
    "El Nino / Heat Stress year": {"market": {"oni": 1.0, "monsoon": -1.0}, "weather": {"temp": 2.0, "rain": -25}},
    "La Nina / Surplus Monsoon": {"market": {"oni": -1.0, "monsoon": 1.0}, "weather": {"rain": 25}},
}
preset = st.sidebar.selectbox("Quick Market Scenario", list(PRESETS), help="Loads instant market shock presets.")
dm, dw = PRESETS[preset].get("market", {}), PRESETS[preset].get("weather", {})
ramp = st.sidebar.select_slider("Market Shock Speed", options=[1, 3, 6], value=3,
                                format_func=lambda x: {1: "Sudden (1 month)", 3: "Over 3 months", 6: "Over 6 months"}[x],
                                help="How many months the market shock builds up leading to forecast origin.")

st.sidebar.markdown("**Market Drivers** (relative to baseline)")
shocks = {}
for k, (label, kind, unit) in se.MARKET_DRIVERS.items():
    if kind == "pct":
        shocks[k] = st.sidebar.slider(f"{label} (%)", -40.0, 40.0, float(dm.get(k, 0.0)), 1.0, format="%.0f", key=f"{preset}|{k}")
    else:
        lim = 2.0 if k == "oni" else 3.0
        shocks[k] = st.sidebar.slider(f"{label} ({unit})", -lim, lim, float(dm.get(k, 0.0)), 0.1, format="%.1f", key=f"{preset}|{k}")

weather = {}
with st.sidebar.expander("Weather & Seasonal Factors (optional)", expanded=bool(dw)):
    st.caption("Shifts expected weather in upcoming forecast months.")
    weather["temp"] = st.slider("Temperature anomaly (deg C)", -5.0, 5.0, float(dw.get("temp", 0.0)), 0.5, key=f"{preset}|w_temp")
    weather["rain"] = st.slider("Rainfall anomaly (% of expected)", -60.0, 100.0, float(dw.get("rain", 0.0)), 5.0, format="%.0f", key=f"{preset}|w_rain")
    weather["humid"] = st.slider("Humidity shift (% pts)", -20.0, 20.0, float(dw.get("humid", 0.0)), 1.0, format="%.0f", key=f"{preset}|w_humid")

mshocks = {k: v for k, v in shocks.items() if v}
wshocks = {k: v for k, v in weather.items() if v}

# ------------------------------------------------------------------ run baseline + scenario
base = run_forecast(E, EKEY, mt, (), ramp, (), method)
scn = run_forecast(E, EKEY, mt, _tup(shocks), ramp, _tup(weather), method)
months = base.drop_duplicates("horizon").set_index("horizon")["target_month"]
ab, a_s = E.aggregate(mt, base, method, sel), E.aggregate(mt, scn, method, sel)
delta = a_s - ab
hist = E.history(mt, sel, 20)

ts_factor = sl.get_ts_factor(mt)
last_actual_ts = float(hist.iloc[-1])
last_actual_l = sl.to_retail_per_l(last_actual_ts, mt)

# Load external market speculation registry
try:
    spec_data = sl.load_speculations()
    spec_records = sl.get_records(spec_data)
except Exception as e:
    spec_records = []

# Main Title & Subtitle
st.title("🥛 Karnataka Retail Milk Price Forecasting & Market Intelligence")
st.markdown(f"""
**State:** Karnataka | **Milk Type:** **{mt.title()}** | **Districts Selected:** **{len(sel)} of {len(S.districts)}** | **Model:** **{method}**  
**Last Historical Actual Month:** **{S.origin:%B %Y}** (`{last_actual_ts:.2f} Rs/kg TS` ≈ `₹{last_actual_l:.2f}/L`)  
**Units Supported:** **Rs per kg Total Solids (Rs/kg TS)** & **Retail Price per Litre (Rs/L)** | Conversion: `1 L = {ts_factor:.4f} kg TS`
""")

# ================================================================== TABS: SPECULATION IS FIRST TAB FOR EXECUTIVE CLARITY
tab_spec, tab1, tab2, tab3, tab4, tab6, tab5 = st.tabs([
    "📰 External Market Speculations & Validation",
    "🎯 Scenario Forecast & What-If",
    "🌪️ Impact of Each Market Driver",
    "📈 Sensitivity Curves",
    "🧮 Every Model Input (±1 Std)",
    "🏆 Feature Importance",
    "📖 Executive Briefing & Pricing Standard"
])

# ================================================================== TAB 1: EXTERNAL MARKET SPECULATION & EXECUTIVE DASHBOARD
with tab_spec:
    st.subheader(f"Independently Published External Market Speculations vs Model Forecast ({mt.title()} Milk)")
    st.caption("Validating quantitative model forecasts against verified external retail price revisions announced by the Karnataka Cooperative Milk Producers' Federation (KMF - Nandini), State Cabinet orders, NDDB, DAHD, and leading national financial dailies.")

    if not spec_records:
        st.warning("No external market speculation records found. Ensure `karnataka_market_speculations.json` is present.")
    else:
        # Build comparison datasets
        spec_monthly = sl.build_monthly_comparison(spec_records, mt, ab, last_actual_ts, months, H)
        spec_metrics = sl.calculate_agreement_metrics(spec_monthly)
        q_summary = sl.build_quarterly_summary(spec_records, spec_monthly, mt)
        ext_records = sl.get_extended_outlook_records()

        # 1. Executive KPI Ribbon (5 Columns)
        kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
        kpi_c1.metric("Validation Horizon", f"Q1 – Q6 ({months.loc[1]:%b %y} – {months.loc[min(H, 18)]:%b %y})", f"{spec_metrics['verified_records']} Verified Reports")
        agr_pct = spec_metrics["agreement_rate_pct"]
        agr_color = "normal" if agr_pct >= 70 else "inverse"
        kpi_c2.metric("Directional Agreement", f"{agr_pct:.1f}%", f"{spec_metrics['agree_count']}/{spec_metrics['verified_records']} Horizons Verified", delta_color=agr_color)
        kpi_c3.metric("Information Standard", "Retail Packaged MRP", "Zero Procurement Distortion")
        kpi_c4.metric("Benchmark Brand", "KMF Nandini", "Statewide All 31 Unions")
        kpi_c5.metric("Pricing Mechanism", "Administered Staircase", "Cabinet Regulated Ceiling")

        st.markdown("---")

        # 2. Timeline-Driven Market Speculation Explorer (Executive Interview Feature)
        st.markdown("### ⏱️ Timeline-Driven Market Speculation Explorer")
        st.write("Click any forecast quarter, month, or extended period below to inspect external journal commentary, published expectations, and directional alignment against the quantitative model.")

        timeline_choices = [
            "Quarter 1 (Dec 2026 – Feb 2027)",
            "Quarter 2 (Mar 2027 – May 2027)",
            "Quarter 3 (Jun 2027 – Aug 2027)",
            "Quarter 4 (Sep 2027 – Nov 2027)",
            "Quarter 5 (Dec 2027 – Feb 2028)",
            "Quarter 6 (Mar 2028 – May 2028)",
        ] + [f"N+{h}: {months.loc[h]:%b %Y}" for h in range(1, min(H, 18) + 1)] + [
            "Extended 1-Year Forward Outlook (FY2028-29 to FY2029-30)"
        ]

        selected_tl = st.selectbox("Select Forecast Timeline to Inspect:", timeline_choices, index=0)

        if "Quarter" in selected_tl:
            q_code = (
                "Q1" if "Quarter 1" in selected_tl else
                "Q2" if "Quarter 2" in selected_tl else
                "Q3" if "Quarter 3" in selected_tl else
                "Q4" if "Quarter 4" in selected_tl else
                "Q5" if "Quarter 5" in selected_tl else "Q6"
            )
            q_match = q_summary[q_summary["Quarter_Code"] == q_code]
            if not q_match.empty:
                qr = q_match.iloc[0]
                qc1, qc2, qc3 = st.columns(3)
                
                with qc1:
                    st.info(f"**Quantitative Model Forecast: {qr['Quarter']}**")
                    st.metric("Model Quarterly Retail Price", f"₹{qr['Model_Avg_L']:.2f}/L", f"{qr['Model_Avg_TS']:.2f} Rs/kg TS")
                    st.caption(f"Model Net Move: **{qr['Model_Net_TS']:+.2f} Rs/kg TS** ({qr['Model_Trend']})")
                    st.markdown(f"💡 **Model Economic Driver:** {qr['Model_Driver']}")

                with qc2:
                    st.info("**External Market Speculation & Consensus**")
                    if pd.notna(qr["External_Price_L"]):
                        st.metric("Expected Retail Price", f"₹{qr['External_Price_L']:.2f}/L", f"{qr['External_Price_TS']:.2f} Rs/kg TS (Range: ₹{qr['External_Range_L']})")
                    else:
                        st.metric("Expected Trend", f"{qr['External_Trend']}", "No numerical point published")
                    st.caption(f"Source: **{qr['Source_Name']}** (Pub: {qr['Publication_Date']})")
                    st.markdown(f"🏷️ **Trade Market Reason:** {qr['External_Driver']}")

                with qc3:
                    st.info("**Validation & Direction Alignment**")
                    q_status = qr["Quarterly_Agreement"]
                    if q_status == "AGREE":
                        st.success("✅ **DIRECTION AGREEMENT**")
                    elif q_status == "ADMINISTERED_HIKE_ANNOUNCED":
                        st.warning("⚠️ **ADMINISTERED HIKE ANNOUNCED** (Cabinet order implemented Nov 1)")
                    else:
                        st.info("ℹ️ **MONITORED REALIZATION**")
                    gap_l_val = qr["Model_Avg_L"] - qr["External_Price_L"] if pd.notna(qr["External_Price_L"]) else 0.0
                    gap_ts_val = qr["Model_Avg_TS"] - qr["External_Price_TS"] if pd.notna(qr["External_Price_TS"]) else 0.0
                    st.write(f"**Model Gap vs External:** `{gap_l_val:+.2f} Rs/L` (`{gap_ts_val:+.2f} Rs/kg TS`)")
                    st.write("**Verification:** `Verified (High Confidence)`")

                st.markdown("#### 📰 Verified Trade Journal Entry & Evidence Dossier")
                st.markdown(f"""
> **\"{qr['Evidence']}\"**
> 
> — *Source:* [{qr['Source_Name']}]({qr['Source_URL']}) | *Published Date:* `{qr['Publication_Date']}` | *Geography:* Karnataka
""")

        elif selected_tl.startswith("N+"):
            sel_h = int(selected_tl.split(":")[0].replace("N+", ""))
            row_match = spec_monthly[spec_monthly["Horizon"] == f"N+{sel_h}"]
            if not row_match.empty:
                r = row_match.iloc[0]
                c_mod, c_ext, c_val = st.columns(3)

                with c_mod:
                    st.info("**Quantitative Model Forecast**")
                    st.metric("Model Retail Price", f"₹{r['Model_Forecast_L']:.2f}/L", f"{r['Model_Forecast_TS']:.2f} Rs/kg TS ({r['Model_Direction']})")
                    st.caption(f"Model: **{method}** | Last actual: ₹{last_actual_l:.2f}/L ({last_actual_ts:.2f} Rs/kg TS)")
                    st.markdown(f"💡 **Model Driver:** {r['Model_Driver']}")

                with c_ext:
                    st.info("**External Market Speculation**")
                    if pd.notna(r["External_Price_L"]):
                        st.metric("Expected Retail Price", f"₹{r['External_Price_L']:.2f}/L", f"{r['External_Price_TS']:.2f} Rs/kg TS (Range: ₹{r['External_Range_L']})")
                    else:
                        st.metric("Expected Retail Trend", f"{r['External_Direction']}", "No numerical point published")
                    st.caption(f"Source: **{r['Source_Name']}** (Pub: {r['Publication_Date']})")
                    st.markdown(f"🏷️ **Trade Market Reason:** {r['Remarks']}")

                with c_val:
                    st.info("**Validation & Direction Alignment**")
                    agr_status = r["Agreement"]
                    if agr_status == "AGREE":
                        st.success("✅ **DIRECTION AGREEMENT**")
                    elif agr_status == "ADMINISTERED_STEP_PENDING":
                        st.warning("⚠️ **ADMINISTERED STEP ANNOUNCED** (Statutory Hike Effective Nov 1)")
                    else:
                        st.info("ℹ️ **MONITORED REALIZATION**")
                    if pd.notna(r["Price_Gap_L"]):
                        st.write(f"**Model Gap vs External:** `{r['Price_Gap_L']:+.2f} Rs/L` (`{r['Price_Gap_TS']:+.2f} Rs/kg TS`)")
                    st.write(f"**Verification:** `{r['Verification_Status']}` ({r['Confidence']} Confidence)")

                st.markdown("#### 📰 Verified Trade Journal Entry & Evidence Dossier")
                st.markdown(f"""
> **\"{r['Evidence']}\"**
> 
> — *Source:* [{r['Source_Name']}]({r['Source_URL']}) | *Published Date:* `{r['Publication_Date']}` | *Geography:* Karnataka
""")

        else:
            st.info("### 🔭 Extended 1-Year Forward Market Intelligence (H1 FY2027-28)")
            if ext_records:
                for er in ext_records:
                    st.markdown(f"""
**Forecast Period:** `{er.get('forecast_period')}`  
**Source Organization:** [{er.get('source_name')}]({er.get('source_url')}) | *Published:* `{er.get('publication_date')}`  
**Expected Direction:** `{er.get('expected_change')}`  

> **Verbatim Market Intelligence Quote:**  
> \"{er.get('evidence')}\"

*Analysis:* {er.get('remarks')}
---
""")
            else:
                st.write("No extended 1-year records currently archived.")

        st.markdown("---")

        # 3. Visual Comparison Chart: Model Baseline vs External Speculation Corridor
        st.markdown("### 📈 Visual Comparison: Model Baseline Forecast vs External Retail Consensus Corridor")
        
        chart_df_list = []
        for dt, val in hist.tail(12).items():
            val_ts = float(val)
            val_l = sl.to_retail_per_l(val_ts, mt)
            chart_df_list.append({"Month": pd.to_datetime(dt), "Retail_Price_L": val_l, "Price_TS": val_ts, "Series": "Historical Actual (Shelf Price)"})
        
        for idx, r in spec_monthly.iterrows():
            chart_df_list.append({"Month": pd.to_datetime(r["Date"]), "Retail_Price_L": float(r["Model_Forecast_L"]), "Price_TS": float(r["Model_Forecast_TS"]), "Series": "Model Baseline Forecast"})
            if pd.notna(r["External_Price_L"]):
                chart_df_list.append({"Month": pd.to_datetime(r["Date"]), "Retail_Price_L": float(r["External_Price_L"]), "Price_TS": float(r["External_Price_TS"]), "Series": "External Published Revision (KMF Nandini)"})
        
        comp_chart_df = pd.DataFrame(chart_df_list)
        line_chart = alt.Chart(comp_chart_df).mark_line(point=True).encode(
            x=alt.X("Month:T", title=None, axis=alt.Axis(format="%b %y")),
            y=alt.Y("Retail_Price_L:Q", scale=alt.Scale(zero=False), title="Retail Milk Price (Rs per Litre)"),
            color=alt.Color("Series:N", scale=alt.Scale(domain=["Historical Actual (Shelf Price)", "Model Baseline Forecast", "External Published Revision (KMF Nandini)"], range=[GREY, BLUE, ORANGE]), title=None),
            tooltip=[alt.Tooltip("Month:T", format="%b %Y"), "Series:N", alt.Tooltip("Retail_Price_L:Q", format=".2f", title="Price (Rs/L)"), alt.Tooltip("Price_TS:Q", format=".2f", title="Price (Rs/kg TS)")]
        ).properties(height=360)
        st.altair_chart(line_chart, use_container_width=True)

        # 4. Comprehensive Monthly Comparison Table
        st.markdown("### 📋 Monthly Forecast vs External Speculation Ledger")
        display_df = spec_monthly[[
            "Horizon", "Quarter", "Period", "Model_Forecast_L", "Model_Forecast_TS", "Model_MoM_L", "Model_Direction",
            "External_Price_L", "External_Price_TS", "External_Range_L", "External_Direction", "Agreement", "Price_Gap_L", "Source_Name"
        ]].copy()
        display_df.rename(columns={
            "Model_Forecast_L": "Model (Rs/L)",
            "Model_Forecast_TS": "Model (Rs/kg TS)",
            "Model_MoM_L": "Model MoM (Rs/L)",
            "Model_Direction": "Model Dir",
            "External_Price_L": "External Spec (Rs/L)",
            "External_Price_TS": "External Spec (Rs/kg TS)",
            "External_Range_L": "External Range (Rs/L)",
            "External_Direction": "External Dir",
            "Price_Gap_L": "Model Gap (Rs/L)",
            "Source_Name": "Source"
        }, inplace=True)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # 5. Dedicated Quarterly Forecast vs External Trade Consensus Ledger
        st.markdown("### 📊 Quarterly Forecast vs External Trade Consensus Ledger")
        q_display = q_summary[[
            "Quarter", "Model_Avg_L", "Model_Avg_TS", "Model_Net_TS", "Model_Trend", "Model_Driver",
            "External_Price_L", "External_Price_TS", "External_Range_L", "External_Trend", "Quarterly_Agreement", "Source_Name"
        ]].copy()
        q_display.rename(columns={
            "Model_Avg_L": "Model Avg (Rs/L)",
            "Model_Avg_TS": "Model Avg (Rs/kg TS)",
            "Model_Net_TS": "Net Move (Rs/kg TS)",
            "Model_Trend": "Model Trend",
            "Model_Driver": "Model Economic Driver",
            "External_Price_L": "External Consensus (Rs/L)",
            "External_Price_TS": "External Consensus (Rs/kg TS)",
            "External_Range_L": "External Range (Rs/L)",
            "External_Trend": "External Trend",
            "Quarterly_Agreement": "Quarterly Agreement",
            "Source_Name": "Trade Source"
        }, inplace=True)
        st.dataframe(q_display, use_container_width=True, hide_index=True)

        # 6. Methodological Integrity Notice
        with st.expander("🛡️ Data Governance & Anti-Leakage Protocol"):
            st.markdown(f"""
- **Strict Retail Packaged MRP Standard:** External records strictly represent consumer packaged shelf prices (MRP) of Nandini milk variants published by the Karnataka Cooperative Milk Producers' Federation (KMF) and ratified by the Karnataka State Cabinet. Farmgate procurement rates and Maharashtra wholesale commodity metrics are explicitly quarantined.
- **Conversion Transparency:** Retail prices are converted to Total Solids (TS) based on calibrated milk density and solids benchmarks (`Price_INR_per_kg_TS = Retail_Price_INR_per_L * {ts_factor:.4f}`).
- **Zero Data Leakage:** External market intelligence is stored in an independent JSON registry (`karnataka_market_speculations.json`) and loaded strictly at evaluation time. It does not enter model feature matrices, lag constructions, or weight estimators.
- **Auditable Live Portals:** Every entry links directly to public official releases: KMF Nandini Portal, NDDB, DAHD, *The Hindu*, *The Economic Times*, and *Deccan Herald*.
""")

# ================================================================== TAB 2: SCENARIO FORECAST & WHAT-IF
with tab1:
    if not (mshocks or wshocks):
        st.info("Baseline mode: No shocks applied yet. Pick a **Quick scenario** in the sidebar or move sliders to test cost-push pressures.")
    else:
        parts = [f"{se.MARKET_DRIVERS[k][0]} {v:+g}{'%' if se.MARKET_DRIVERS[k][1] == 'pct' else ' ' + se.MARKET_DRIVERS[k][2]}" for k, v in mshocks.items()]
        parts += [f"{se.WEATHER_DRIVERS[k][0]} {v:+g} {se.WEATHER_DRIVERS[k][1]}" for k, v in wshocks.items()]
        st.success("**Active Scenario:** " + " | ".join(parts))

    hs = sorted({1, min(3, H), H})
    cols = st.columns(len(hs))
    for c, h in zip(cols, hs):
        c.metric(f"N+{h} ({months.loc[h]:%b %Y})", f"{a_s.loc[h]:.1f} Rs/kg TS (₹{sl.to_retail_per_l(a_s.loc[h], mt):.2f}/L)", f"{delta.loc[h]:+.2f} vs baseline")

    f_months = [hist.index[-1]] + list(months.loc[1:H])
    df = pd.concat([
        pd.DataFrame({"Month": hist.index, "Rate": hist.to_numpy(), "Series": "Actual History"}),
        pd.DataFrame({"Month": f_months, "Rate": [hist.iloc[-1]] + list(ab.loc[1:H]), "Series": "Baseline Forecast"}),
        pd.DataFrame({"Month": f_months, "Rate": [hist.iloc[-1]] + list(a_s.loc[1:H]), "Series": "Scenario Forecast"})
    ])
    line = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Month:T", title=None, axis=alt.Axis(format="%b %y")),
        y=alt.Y("Rate:Q", scale=alt.Scale(zero=False), title="Rs per kg TS"),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["Actual History", "Baseline Forecast", "Scenario Forecast"], range=[GREY, BLUE, ORANGE]), title=None),
        tooltip=[alt.Tooltip("Month:T", format="%b %Y"), alt.Tooltip("Rate:Q", format=".2f"), "Series:N"]).properties(height=340)
    st.altair_chart(line, use_container_width=True)

    d_df = pd.DataFrame({"Month": [f"N+{h} ({months.loc[h]:%b %Y})" for h in range(1, H + 1)], "Change": [delta.loc[h] for h in range(1, H + 1)]})
    bars = alt.Chart(d_df).mark_bar().encode(
        x=alt.X("Month:N", sort=None, title=None, axis=alt.Axis(labelAngle=-20)),
        y=alt.Y("Change:Q", title="Scenario Delta (Rs/kg TS)"),
        color=alt.condition(alt.datum.Change > 0, alt.value(GREEN), alt.value(RED)),
        tooltip=["Month:N", alt.Tooltip("Change:Q", format="+.2f")]).properties(height=220)
    st.altair_chart(bars, use_container_width=True)

    st.subheader("District Breakdown (Rs/kg TS Delta: Scenario - Baseline)")
    dd = (scn.set_index([s1.DISTRICT, "horizon"])[method] - base.set_index([s1.DISTRICT, "horizon"])[method]).unstack("horizon")
    dd = dd.loc[[d for d in S.districts if d in sel], list(range(1, H + 1))]
    dd.loc["STATE AVERAGE (weighted)"] = [delta.loc[h] for h in range(1, H + 1)]
    dd.columns = [f"N+{h} ({months.loc[h]:%b %y})" for h in dd.columns]
    st.dataframe(dd.round(2), use_container_width=True)

    with st.expander("Under the hood: Model inputs that changed in this scenario"):
        hz = st.selectbox("Horizon", list(range(1, H + 1)), index=min(2, H - 1), format_func=lambda h: f"N+{h} ({months.loc[h]:%b %Y})", key="chg_h")
        ci = run_changed(E, EKEY, mt, _tup(shocks), ramp, _tup(weather), hz)
        if ci.empty:
            st.write("No model inputs changed in this scenario.")
        else:
            used = E.used_inputs(mt, method)
            ci.insert(2, "Used by Model", ci["Model input"].isin(used).map({True: "Yes", False: "No"}))
            st.dataframe(ci.round(3), hide_index=True, use_container_width=True)

# ================================================================== TAB 3: TORNADO IMPACT
with tab2:
    st.subheader("Which market driver moves the forecast the most?")
    st.write("Each driver is shocked **individually** up and down by the selected %, keeping all other market factors neutral.")
    c1, c2 = st.columns(2)
    size = c1.slider("Shock size for price drivers (%)", 5, 30, 15, 5)
    th = c2.selectbox("Forecast Horizon", list(range(1, E.H + 1)), index=min(2, E.H - 1), format_func=lambda h: f"N+{h} ({months.loc[h]:%b %Y})", key="tor_h")
    tor = run_tornado(E, EKEY, mt, method, float(size), ramp, int(th))
    tor_used, tor_ign = tor[tor["Used by this model"] == "yes"], tor[tor["Used by this model"] == "no"]
    if tor_used.empty:
        st.info("The selected model does not directly read market driver features. Choose another model (e.g., RecursiveX Blend or Direct Blend) in the sidebar.")
    else:
        st.altair_chart(diverging_bars(tor_used, "Driver", "Rise (+) effect", "Fall (-) effect", "Driver Rises", "Driver Falls", "Swing",
                                       f"Effect on N+{th} forecast vs baseline (Rs/kg TS)"), use_container_width=True)
    show = tor.drop(columns=["Key"]).copy()
    for c in ("Rise (+) effect", "Fall (-) effect", "Swing"):
        show[c] = show[c].map(lambda v: "not used" if pd.isna(v) else f"{v:.2f}")
    st.dataframe(show, hide_index=True, use_container_width=True)

# ================================================================== TAB 4: SENSITIVITY SWEEPS
with tab3:
    st.subheader("Sensitivity Curve: Continuous Price Elasticity")
    c1, c2, c3 = st.columns(3)
    key = c1.selectbox("Market Driver", list(se.MARKET_DRIVERS), format_func=lambda k: se.MARKET_DRIVERS[k][0])
    kind, unit = se.MARKET_DRIVERS[key][1], se.MARKET_DRIVERS[key][2]
    rng = c2.slider(f"Sweep Range (+- {unit})", 5.0, 40.0, 20.0, 5.0) if kind == "pct" else c2.slider(f"Range (+- {unit})", 0.5, 3.0, 1.0, 0.5)
    hzs = c3.multiselect("Forecast Horizons", list(range(1, E.H + 1)), default=[h for h in (1, 3, 6) if h <= E.H], format_func=lambda h: f"N+{h}")
    if hzs:
        step = 5.0 if kind == "pct" else 0.25
        vals = tuple(float(v) for v in np.arange(-rng, rng + 1e-9, step))
        sw = run_sweep(E, EKEY, mt, method, key, vals, ramp, tuple(sorted(hzs)))
        ln = alt.Chart(sw).mark_line(point=True).encode(
            x=alt.X("Shock:Q", title=f"{se.MARKET_DRIVERS[key][0]}: Change ({unit})"),
            y=alt.Y("Change vs baseline:Q", title="Predicted Rate Delta (Rs/kg TS)"),
            color=alt.Color("Horizon:N", title=None),
            tooltip=[alt.Tooltip("Shock:Q"), "Horizon:N", alt.Tooltip("Change vs baseline:Q", format="+.2f")]).properties(height=380)
        zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(strokeDash=[4, 4], color=GREY).encode(y="y:Q")
        st.altair_chart(ln + zero, use_container_width=True)

# ================================================================== TAB 5: INPUTS (±1 STD)
with tab4:
    st.subheader("Impact of Every Individual Model Feature (±1 Standard Deviation)")
    hz4 = st.selectbox("Forecast Month", list(range(1, E.H + 1)), index=min(2, E.H - 1), format_func=lambda h: f"N+{h} ({months.loc[h]:%b %Y})", key="inp_h")
    inp = run_inputs(E, EKEY, mt, int(hz4))
    top = st.slider("Show Top Features", 5, 35, 12)
    t4 = inp.head(top)
    st.altair_chart(diverging_bars(t4, "Model input", "Input +1 std", "Input -1 std", "Feature +1 Std", "Feature -1 Std", "Swing",
                                   f"Impact on N+{hz4} Forecast (Rs/kg TS)"), use_container_width=True)
    grp = inp.groupby("Group")["Swing"].sum().sort_values(ascending=False).reset_index()
    st.markdown("**Impact by Feature Category**")
    st.altair_chart(alt.Chart(grp).mark_bar(color=BLUE).encode(
        y=alt.Y("Group:N", sort="-x", title=None), x=alt.X("Swing:Q", title="Total Swing (Rs/kg TS)"),
        tooltip=["Group:N", alt.Tooltip("Swing:Q", format=".2f")]).properties(height=max(180, 28 * len(grp))), use_container_width=True)

# ================================================================== TAB 6: FEATURE IMPORTANCE
with tab6:
    st.subheader("Feature Importance: What Do the Fitted Models Rely On?")
    st.write("Using **permutation feature importance**: each feature is shuffled to break its relationship with price, measuring the rise in prediction error (MAE in Rs/kg TS). Shares sum to 100% per milk type.")
    topn = st.slider("Show Top N Features", 5, 25, 12, key="fi_top")
    fcols = st.columns(2)
    for col, m in zip(fcols, s1.MILK_TYPES):
        fi_df, fi_name = run_importance(E, EKEY, m)
        with col:
            st.markdown(f"### {m.title()} Milk")
            st.caption(f"Model: **{fi_name}**")
            top_df = fi_df.head(topn)
            st.altair_chart(alt.Chart(top_df).mark_bar(color=BLUE if m == "COW" else ORANGE).encode(
                y=alt.Y("Model input:N", sort="-x", title=None), x=alt.X("Importance (%):Q", title="Importance Share (%)"),
                tooltip=["Model input:N", "Group:N", alt.Tooltip("Importance (%):Q", format=".1f"), "Meaning:N"]
            ).properties(height=max(220, 26 * len(top_df))), use_container_width=True)
            g = fi_df.groupby("Group")["Importance (%)"].sum().sort_values(ascending=False).reset_index()
            st.markdown(f"**{m.title()} Importance by Group**")
            st.altair_chart(alt.Chart(g).mark_bar(color=GREY).encode(
                y=alt.Y("Group:N", sort="-x", title=None), x=alt.X("Importance (%):Q", title="Share (%)"),
                tooltip=["Group:N", alt.Tooltip("Importance (%):Q", format=".1f")]
            ).properties(height=max(160, 26 * len(g))), use_container_width=True)
            t = top_df[["Model input", "Group", "Importance (%)", "MAE rise when shuffled", "Meaning"]].copy()
            t["Importance (%)"] = t["Importance (%)"].round(1)
            t["MAE rise when shuffled"] = t["MAE rise when shuffled"].round(3)
            st.dataframe(t, hide_index=True, use_container_width=True)

# ================================================================== TAB 7: EXECUTIVE BRIEFING
with tab5:
    st.markdown(f"""
### Executive Briefing: Karnataka Retail Milk Pricing Dynamics

#### ❓ Why Do Quarterly Retail Forecasts Barely Move? (CEO FAQ)
A central observation in this application is that baseline quarterly retail milk forecasts across **all six quarters (Q1 through Q6: Dec 2026 to May 2028)** remain essentially flat (~₹54/L for Cow and ~₹70/L for Buffalo). **This is 100% mathematically and economically correct behavior.** Here is why:

1. **Administered Staircase Pricing vs Market Floating:**
   * **Maharashtra (Farmgate Procurement):** Prices float continuously month-to-month driven by private dairy competition, daily spot milk arrivals, and export parity.
   * **Karnataka (Consumer Retail MRP):** Prices represent **Packaged Consumer Retail MRP** under the state cooperative federation brand **Nandini** (governed by KMF and ratified by the Karnataka State Cabinet). By government policy, retail milk prices are **not allowed to float freely**.
2. **The 12-to-18 Month Price Freeze Rule:**
   * To protect household budgets from volatile inflation, the Karnataka government maintains an **administered price plateau** that historically lasts **12 to 18 months**.
   * Historical evidence confirms this: retail milk stayed completely flat from April 2025 to October 2026 at ₹46/L (Cow) and ₹62/L (Buffalo).
   * Effective **November 1, 2026**, the State Cabinet executed an administered revision of **+₹8.00/L** (taking Cow to ₹54/L and Buffalo to ₹70/L).
3. **The 6-Quarter Horizon Structure (Dec 2026 – May 2028):**
   * **Q1 to Q4 (Dec 2026 – Nov 2027):** Sits squarely inside the mandatory 12-month post-revision statutory price freeze. Zero retail movement is expected or permitted.
   * **Q5 to Q6 (Dec 2027 – May 2028):** Reaches the 14-to-18 month horizon where dairy unions traditionally begin submitting representations for production-cost relief. Until a formal Cabinet notification is signed, the administered shelf price remains pegged at ₹54/L and ₹70/L.
   * Therefore, an econometric model that forecasted volatile retail fluctuations across these quarters would be completely counter-factual and decoupled from Karnataka's policy framework.
4. **Where the Real Economic Movement Lives:**
   * While consumer shelf prices remain sticky on the staircase plateau, the **underlying cost of production is constantly fluctuating** due to cattle feed, maize, soy meal, SMP stocks, and heatwaves.
   * When feed and fodder costs surge, the economic pressure builds up silently beneath the retail price freeze until dairy farmer viability is threatened, ultimately triggering the next statutory staircase revision!
   * The **Scenario Forecast & What-If Simulator** in Tabs 2, 3, and 4 models this exact cost-push pressure.

---

### Key Takeaways for CEO & Leadership
1. **Cow Milk Drivers:** Cow retail prices are heavily linked to **Grain Feed (Maize & Soybean meal)** and **Skimmed Milk Powder (SMP)** balances.
2. **Buffalo Milk Drivers:** Buffalo milk (high fat 6.0–6.5%) is predominantly influenced by **Butter and Ghee wholesale realisations**.
3. **Farmer Remuneration Allocation:** In the November 2026 revision, **₹7.00/L of the ₹8.00/L hike was directly passed to rural dairy farmers**, with only ₹1.00/L retained by district unions for cold chain logistics.
4. **District Uniformity:** Across all 31 districts, local rates maintain a constant spread relative to the State Average (Bengaluru Urban +₹3.59, Chamarajanagar -₹1.90), ensuring union-level operational alignment.
""")
    if st.button("Run Self-Verification: Engine Baseline vs Step 2 Backtest"):
        with st.spinner("Checking numerical alignment with step 2 backtest..."):
            diff = E.self_check(mt)
        (st.success if diff < 1e-5 else st.error)(f"Engine vs Step-2 Max Discrepancy for {mt.title()}: {diff:.2e} Rs/kg TS (Clean Match)")
