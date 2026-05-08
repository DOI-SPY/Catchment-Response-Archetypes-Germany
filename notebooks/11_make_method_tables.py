from __future__ import annotations

from pathlib import Path
import pandas as pd


# =========================================================
# 0. PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

OUT_TABLE4 = TABLE_DIR / "Table4_operational_definitions_of_monthly_hydrological_states.csv"
OUT_TABLE5 = TABLE_DIR / "Table5_response_metrics_used_in_this_study.csv"
OUT_TABLE_S3 = TABLE_DIR / "TableS3_robustness_analysis_settings.csv"
OUT_EXCEL = TABLE_DIR / "Method_chapter_tables.xlsx"


# =========================================================
# 1. TABLE 4: HYDROLOGICAL STATE DEFINITIONS
# =========================================================
def make_table4() -> pd.DataFrame:
    rows = [
        {
            "State": "drought_like",
            "Chinese label": "干旱型月份",
            "Operational rule": "Q_{i,t} <= Q20_{i,m}",
            "Threshold / temporal condition": (
                "Monthly mean discharge is lower than or equal to the site- and calendar-month-specific "
                "20th percentile."
            ),
            "Hydrological meaning": (
                "The month represents a low-flow condition relative to the long-term seasonal distribution "
                "of the same station and calendar month."
            ),
            "Role in analysis": (
                "Used to characterize low-flow background conditions and to identify subsequent "
                "post-drought rewetting months."
            ),
            "Output field": "hydro_state = drought_like",
        },
        {
            "State": "post_drought_rewetting",
            "Chinese label": "后干旱再润湿型月份",
            "Operational rule": (
                "S_{i,t+r}=R if S_{i,t}=D, r in {1,2}, and Q_{i,t+r} > Q50_{i,m+r}"
            ),
            "Threshold / temporal condition": (
                "The first or second month after a drought-like month is classified as post-drought "
                "rewetting if monthly mean discharge recovers above the site- and calendar-month-specific median."
            ),
            "Hydrological meaning": (
                "The month represents hydrological recovery following a prior low-flow condition, "
                "distinguished from ordinary high-flow months without a preceding drought-like background."
            ),
            "Role in analysis": (
                "Used as the key compound-state category to test whether post-drought recovery produces "
                "distinct water-quality, flux, and c-Q responses."
            ),
            "Output field": "hydro_state = post_drought_rewetting",
        },
        {
            "State": "non_drought_highflow",
            "Chinese label": "非干旱高流量月份",
            "Operational rule": "Q_{i,t} >= Q80_{i,m} and S_{i,t} != R",
            "Threshold / temporal condition": (
                "Monthly mean discharge is higher than or equal to the site- and calendar-month-specific "
                "80th percentile, and the month is not classified as post-drought rewetting."
            ),
            "Hydrological meaning": (
                "The month represents high-flow conditions not directly constrained by a preceding "
                "drought-like month."
            ),
            "Role in analysis": (
                "Used as a high-flow reference state to distinguish ordinary high-flow responses from "
                "post-drought rewetting responses."
            ),
            "Output field": "hydro_state = non_drought_highflow",
        },
        {
            "State": "normal",
            "Chinese label": "正常月份",
            "Operational rule": "S_{i,t} not in {D,R,H}",
            "Threshold / temporal condition": (
                "All remaining months that are neither drought-like, post-drought rewetting, "
                "nor non-drought high-flow months."
            ),
            "Hydrological meaning": (
                "The month represents background hydrological conditions outside the low-flow, recovery, "
                "and high-flow state classes."
            ),
            "Role in analysis": (
                "Used as the baseline state for state comparison and for calculating state-specific "
                "c-Q slope differences."
            ),
            "Output field": "hydro_state = normal",
        },
    ]

    return pd.DataFrame(rows)


# =========================================================
# 2. TABLE 5: RESPONSE METRICS
# =========================================================
def make_table5() -> pd.DataFrame:
    rows = [
        {
            "Metric": "Concentration anomaly",
            "Symbol": "CA_{i,t,s}",
            "Equation": "CA_{i,t,s}=ln(mean_C_{i,t,s}/mean_FNC_{i,t,s})",
            "Source variables": "mean_C; mean_FNC",
            "Scale": "Station-month-constituent",
            "Interpretation": (
                "Positive values indicate that observed monthly concentration is higher than the "
                "flow-normalized concentration baseline; negative values indicate lower-than-baseline concentration."
            ),
            "Used in": "State comparison; response signatures; archetype extraction",
        },
        {
            "Metric": "Flux anomaly",
            "Symbol": "FA_{i,t,s}",
            "Equation": "FA_{i,t,s}=ln(mean_Flux_{i,t,s}/mean_FNFlux_{i,t,s})",
            "Source variables": "mean_Flux; mean_FNFlux",
            "Scale": "Station-month-constituent",
            "Interpretation": (
                "Positive values indicate enhanced monthly constituent export relative to the "
                "flow-normalized flux baseline; negative values indicate weaker export."
            ),
            "Used in": "State comparison; response signatures; archetype extraction",
        },
        {
            "Metric": "State-specific c-Q slope",
            "Symbol": "beta_{i,s,k}",
            "Equation": "ln(C_{i,t,s})=alpha_{i,s,k}+beta_{i,s,k}ln(Q_{i,t})+epsilon_{i,t,s,k}",
            "Source variables": "mean_C; mean_Q; hydro_state",
            "Scale": "Station-constituent-state",
            "Interpretation": (
                "Positive slopes indicate enrichment with increasing discharge; negative slopes indicate "
                "dilution; near-zero slopes indicate weak concentration sensitivity to discharge."
            ),
            "Used in": "State-dependent c-Q relationship analysis",
        },
        {
            "Metric": "c-Q slope difference",
            "Symbol": "Delta_beta_{i,s,k}",
            "Equation": "Delta_beta_{i,s,k}=beta_{i,s,k}-beta_{i,s,N}",
            "Source variables": "beta_{i,s,k}; beta_{i,s,N}",
            "Scale": "Station-constituent-state",
            "Interpretation": (
                "Measures how the c-Q relationship under a given hydrological state differs from "
                "the normal-month baseline."
            ),
            "Used in": "Coupling-change analysis; response signatures; archetype extraction",
        },
        {
            "Metric": "State-level concentration signature",
            "Symbol": "S_CA_{i,s,k}",
            "Equation": "S_CA_{i,s,k}=median(CA_{i,t,s} | S_{i,t}=k)",
            "Source variables": "CA; hydro_state",
            "Scale": "Station-constituent-state",
            "Interpretation": (
                "Summarizes the typical concentration anomaly of a station-constituent pair under "
                "a specific hydrological state."
            ),
            "Used in": "Response signature matrix",
        },
        {
            "Metric": "State-level flux signature",
            "Symbol": "S_FA_{i,s,k}",
            "Equation": "S_FA_{i,s,k}=median(FA_{i,t,s} | S_{i,t}=k)",
            "Source variables": "FA; hydro_state",
            "Scale": "Station-constituent-state",
            "Interpretation": (
                "Summarizes the typical flux anomaly of a station-constituent pair under "
                "a specific hydrological state."
            ),
            "Used in": "Response signature matrix",
        },
        {
            "Metric": "Rewetting-versus-high-flow concentration contrast",
            "Symbol": "Delta_CA_{i,s,R-H}",
            "Equation": "Delta_CA_{i,s,R-H}=S_CA_{i,s,R}-S_CA_{i,s,H}",
            "Source variables": "S_CA under post_drought_rewetting and non_drought_highflow",
            "Scale": "Station-constituent",
            "Interpretation": (
                "Compares whether post-drought rewetting produces stronger or weaker concentration "
                "anomaly than ordinary non-drought high-flow conditions."
            ),
            "Used in": "Testing the distinctiveness of post-drought rewetting responses",
        },
        {
            "Metric": "Rewetting-versus-high-flow flux contrast",
            "Symbol": "Delta_FA_{i,s,R-H}",
            "Equation": "Delta_FA_{i,s,R-H}=S_FA_{i,s,R}-S_FA_{i,s,H}",
            "Source variables": "S_FA under post_drought_rewetting and non_drought_highflow",
            "Scale": "Station-constituent",
            "Interpretation": (
                "Compares whether post-drought rewetting produces stronger or weaker export anomaly "
                "than ordinary non-drought high-flow conditions."
            ),
            "Used in": "Testing the distinctiveness of post-drought rewetting export responses",
        },
    ]

    return pd.DataFrame(rows)


# =========================================================
# 3. TABLE S3: ROBUSTNESS SETTINGS
# =========================================================
def make_table_s3() -> pd.DataFrame:
    rows = [
        {
            "Robustness component": "Low-flow threshold",
            "Main setting": "Q20",
            "Alternative setting(s)": "Q15; Q25",
            "Purpose": (
                "Test whether drought-like state identification and subsequent response patterns "
                "depend on the selected low-flow percentile."
            ),
            "Expected output": "State counts; CA/FA summaries; c-Q slope differences under alternative thresholds",
        },
        {
            "Robustness component": "High-flow threshold",
            "Main setting": "Q80",
            "Alternative setting(s)": "Q75; Q85",
            "Purpose": (
                "Test whether non-drought high-flow responses are sensitive to the high-flow definition."
            ),
            "Expected output": "High-flow state counts; rewetting-versus-high-flow contrasts",
        },
        {
            "Robustness component": "Post-drought rewetting window",
            "Main setting": "1–2 months after drought-like month",
            "Alternative setting(s)": "1 month; 1–3 months",
            "Purpose": (
                "Evaluate whether the distinctiveness of post-drought rewetting responses depends "
                "on the temporal recovery window."
            ),
            "Expected output": "Rewetting state counts; CA/FA contrasts; Delta_beta summaries",
        },
        {
            "Robustness component": "Hydrological-state input variable",
            "Main setting": "mean_Q",
            "Alternative setting(s)": "median_Q where available",
            "Purpose": (
                "Check whether hydrological-state classification depends on the discharge summary metric."
            ),
            "Expected output": "Agreement rate of hydro_state labels; response summaries under alternative Q metric",
        },
        {
            "Robustness component": "Anomaly baseline",
            "Main setting": "WRTDS flow-normalized baseline",
            "Alternative setting(s)": "Normal-month median baseline",
            "Purpose": (
                "Test whether concentration and flux anomaly conclusions depend on the chosen baseline."
            ),
            "Expected output": "CA/FA distributions and state contrasts under alternative baselines",
        },
        {
            "Robustness component": "Sample composition",
            "Main setting": "Constituent-specific main samples",
            "Alternative setting(s)": "Common-station subset; longer-record subset",
            "Purpose": (
                "Test whether conclusions are driven by unequal station coverage among NO3N, PO4P, and DOC."
            ),
            "Expected output": "Main state-response patterns under restricted samples",
        },
        {
            "Robustness component": "Archetype identification",
            "Main setting": "Hierarchical clustering on standardized response signatures",
            "Alternative setting(s)": "k-means; HDBSCAN; different feature subsets",
            "Purpose": (
                "Test whether response archetypes are stable across clustering algorithms and feature sets."
            ),
            "Expected output": "Cluster stability metrics; archetype membership agreement",
        },
        {
            "Robustness component": "Explainable attribution",
            "Main setting": "Tree-based model with SHAP interpretation",
            "Alternative setting(s)": "Random forest; XGBoost; LightGBM; repeated splits",
            "Purpose": (
                "Evaluate whether important controls are stable across model choices and random partitions."
            ),
            "Expected output": "Variable-importance stability; SHAP ranking consistency",
        },
    ]

    return pd.DataFrame(rows)


# =========================================================
# 4. SAVE
# =========================================================
def main():
    table4 = make_table4()
    table5 = make_table5()
    table_s3 = make_table_s3()

    table4.to_csv(OUT_TABLE4, index=False, encoding="utf-8-sig")
    table5.to_csv(OUT_TABLE5, index=False, encoding="utf-8-sig")
    table_s3.to_csv(OUT_TABLE_S3, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
        table4.to_excel(writer, sheet_name="Table4_state_definitions", index=False)
        table5.to_excel(writer, sheet_name="Table5_response_metrics", index=False)
        table_s3.to_excel(writer, sheet_name="TableS3_robustness", index=False)

    print("\n=== Method chapter tables generated successfully ===")
    print(f"- {OUT_TABLE4}")
    print(f"- {OUT_TABLE5}")
    print(f"- {OUT_TABLE_S3}")
    print(f"- {OUT_EXCEL}")

    print("\n=== Table 4 preview ===")
    print(table4.to_string(index=False))

    print("\n=== Table 5 preview ===")
    print(table5.to_string(index=False))

    print("\n=== Table S3 preview ===")
    print(table_s3.to_string(index=False))


if __name__ == "__main__":
    main()