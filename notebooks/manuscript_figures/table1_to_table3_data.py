from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


# =========================================================
# 0. PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data_raw" / "QUADICA_v2"
DATA_DIR = RAW_DIR / "data" / "contents" / "data"

INTER_DIR = PROJECT_ROOT / "data_intermediate"
FINAL_DIR = PROJECT_ROOT / "data_final"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ATTR_FILE = DATA_DIR / "attributes.csv"
META_Q_FILE = DATA_DIR / "metadata_q.csv"
META_C_FILE = DATA_DIR / "metadata_c.csv"
WRTDS_SUMMARY_FILE = DATA_DIR / "wrtds_summary.csv"
INPUT_NP_FILE = DATA_DIR / "input_N_P.csv"
CLIMATE_MONTHLY_FILE = DATA_DIR / "climate_monthly.csv"

STATION_SOLUTE_MASTER_FILE = INTER_DIR / "station_solute_master.csv"
MONTHLY_PANEL_FILE = FINAL_DIR / "analysis_panel_station_month_main.csv"
MONTHLY_STATE_FILE = FINAL_DIR / "analysis_panel_station_month_states.csv"

OUT_TABLE1 = TABLE_DIR / "Table1_datasets_and_variables.csv"
OUT_TABLE2 = TABLE_DIR / "Table2_primary_constituents.csv"
OUT_TABLE3 = TABLE_DIR / "Table3_screening_and_final_sample.csv"
OUT_TABLE4 = TABLE_DIR / "Table4_final_sample_by_constituent.csv"
OUT_PREDICTOR_INVENTORY = TABLE_DIR / "predictor_inventory_for_attribution.csv"
OUT_EXCEL = TABLE_DIR / "Data_chapter_tables.xlsx"


TARGET_SOLUTES = ["NO3N", "PO4P", "DOC"]


# =========================================================
# 1. HELPERS
# =========================================================
def read_csv_fallback(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to read {path}: {last_error}")


def has_file(path: Path) -> str:
    return "Available" if path.exists() else "Missing"


def simplify_list(items, max_n=12):
    items = [str(x) for x in items if pd.notna(x)]
    if len(items) <= max_n:
        return "; ".join(items)
    return "; ".join(items[:max_n]) + f"; ... (+{len(items) - max_n} more)"


# =========================================================
# 2. TABLE 1: DATASETS AND VARIABLES
# =========================================================
def make_table1() -> pd.DataFrame:
    rows = [
        {
            "Data category": "Water-quality and WRTDS monthly outputs",
            "Source file(s)": "wrtds_monthly.csv; wrtds_summary.csv",
            "Main variables / fields": "solute; mean_C; mean_FNC; mean_Flux; mean_FNFlux; mean_Q",
            "Temporal resolution": "Monthly",
            "Role in this study": "Quantify concentration anomaly, flux anomaly, and state-specific c-Q relationships",
            "Availability in local project": has_file(DATA_DIR / "time_series" / "wrtds_monthly.csv"),
        },
        {
            "Data category": "Discharge information",
            "Source file(s)": "metadata_q.csv; q_annual.csv; WRTDS monthly outputs",
            "Main variables / fields": "Q_flag; Station_Q; mean_Q; median_Q",
            "Temporal resolution": "Monthly / annual metadata",
            "Role in this study": "Identify monthly hydrological compound states and support c-Q analysis",
            "Availability in local project": has_file(META_Q_FILE),
        },
        {
            "Data category": "Climate variables",
            "Source file(s)": "climate_monthly.csv",
            "Main variables / fields": "tavg; pre; pet",
            "Temporal resolution": "Monthly",
            "Role in this study": "Describe hydroclimatic background and support explainable attribution",
            "Availability in local project": has_file(CLIMATE_MONTHLY_FILE),
        },
        {
            "Data category": "Catchment attributes",
            "Source file(s)": "attributes.csv",
            "Main variables / fields": "Area_km2; land use fractions; topography; soils; hydrogeology; hydrological indices",
            "Temporal resolution": "Catchment-scale static attributes",
            "Role in this study": "Explain catchment response signatures and response archetypes",
            "Availability in local project": has_file(ATTR_FILE),
        },
        {
            "Data category": "Nutrient input indicators",
            "Source file(s)": "input_N_P.csv; attributes.csv",
            "Main variables / fields": "N surplus; P surplus; point-source N and P inputs; WWTP-related inputs",
            "Temporal resolution": "Annual / aggregated indicators",
            "Role in this study": "Represent nutrient-source pressure in explainable attribution",
            "Availability in local project": has_file(INPUT_NP_FILE),
        },
        {
            "Data category": "GIS layers",
            "Source file(s)": "catchments.shp; stations.shp; stations_mod.shp; riversegments_mod.shp",
            "Main variables / fields": "catchment polygons; station points; river segments",
            "Temporal resolution": "Spatial layers",
            "Role in this study": "Support study-domain visualization and spatial interpretation",
            "Availability in local project": has_file(DATA_DIR / "gis" / "catchments.shp"),
        },
    ]

    return pd.DataFrame(rows)


# =========================================================
# 3. TABLE 2: PRIMARY CONSTITUENTS
# =========================================================
def make_table2() -> pd.DataFrame:
    rows = [
        {
            "Constituent": "NO3N",
            "Full name": "Nitrate nitrogen",
            "Constituent group": "Inorganic nitrogen",
            "Hydrological-biogeochemical relevance": (
                "Mobile nitrogen form related to agricultural inputs, groundwater contributions, "
                "nitrogen retention, and transport under changing flow conditions"
            ),
            "Role in this study": (
                "Represents nitrogen response across drought-like, post-drought rewetting, "
                "non-drought high-flow, and normal months"
            ),
        },
        {
            "Constituent": "PO4P",
            "Full name": "Orthophosphate phosphorus",
            "Constituent group": "Reactive phosphorus",
            "Hydrological-biogeochemical relevance": (
                "Sensitive to source availability, point inputs, surface transport, and flow-path connectivity"
            ),
            "Role in this study": (
                "Represents phosphorus export response under rewetting and high-flow conditions"
            ),
        },
        {
            "Constituent": "DOC",
            "Full name": "Dissolved organic carbon",
            "Constituent group": "Organic carbon",
            "Hydrological-biogeochemical relevance": (
                "Linked to organic matter mobilization, soil-water connectivity, and wetness recovery processes"
            ),
            "Role in this study": (
                "Represents carbon mobilization response under monthly hydrological compound states"
            ),
        },
    ]

    return pd.DataFrame(rows)


# =========================================================
# 4. TABLE 3 AND TABLE 4: SAMPLE SCREENING
# =========================================================
def make_sample_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    attr = read_csv_fallback(ATTR_FILE)
    meta_q = read_csv_fallback(META_Q_FILE)

    n_total_stations = attr["OBJECTID"].nunique()

    if "Q_flag" in meta_q.columns:
        n_qflag = pd.to_numeric(meta_q["Q_flag"], errors="coerce").fillna(0).astype(int).sum()
    else:
        n_qflag = np.nan

    if MONTHLY_PANEL_FILE.exists():
        monthly = read_csv_fallback(MONTHLY_PANEL_FILE)
    else:
        raise FileNotFoundError(f"Missing monthly panel: {MONTHLY_PANEL_FILE}")

    final_by_solute = (
        monthly[monthly["solute"].isin(TARGET_SOLUTES)]
        .groupby("solute")
        .agg(
            n_monthly_records=("solute", "size"),
            n_stations=("OBJECTID", pd.Series.nunique),
            year_min=("Year", "min"),
            year_max=("Year", "max"),
            n_with_mean_Q=("mean_Q", lambda x: pd.to_numeric(x, errors="coerce").notna().sum()),
            n_with_mean_C=("mean_C", lambda x: pd.to_numeric(x, errors="coerce").notna().sum()),
            n_with_mean_Flux=("mean_Flux", lambda x: pd.to_numeric(x, errors="coerce").notna().sum()),
            n_with_mean_FNC=("mean_FNC", lambda x: pd.to_numeric(x, errors="coerce").notna().sum()),
            n_with_mean_FNFlux=("mean_FNFlux", lambda x: pd.to_numeric(x, errors="coerce").notna().sum()),
        )
        .reset_index()
    )

    final_by_solute["solute"] = pd.Categorical(
        final_by_solute["solute"],
        categories=TARGET_SOLUTES,
        ordered=True
    )
    final_by_solute = final_by_solute.sort_values("solute").reset_index(drop=True)

    n_final_records = int(final_by_solute["n_monthly_records"].sum())
    n_final_unique_stations = monthly["OBJECTID"].nunique()
    n_station_constituent_initial = n_total_stations * len(TARGET_SOLUTES)

    table3_rows = [
        {
            "Screening step": "Initial monitoring network",
            "Criterion": "All water-quality stations in QUADICA v2 attributes table",
            "Remaining sample": f"{n_total_stations} stations",
            "Purpose": "Define the full catchment monitoring network",
        },
        {
            "Screening step": "Discharge matching",
            "Criterion": "Stations with paired or matchable discharge information (Q_flag = 1)",
            "Remaining sample": f"{int(n_qflag)} stations" if pd.notna(n_qflag) else "Not available",
            "Purpose": "Support hydrological-state identification and c-Q analysis",
        },
        {
            "Screening step": "Target-constituent selection",
            "Criterion": "NO3N, PO4P, and DOC selected as representative N, P, and C constituents",
            "Remaining sample": f"{n_station_constituent_initial} possible station-constituent combinations",
            "Purpose": "Focus the analysis on contrasting nutrient and carbon responses",
        },
        {
            "Screening step": "WRTDS monthly-output availability",
            "Criterion": "Station-constituent combinations with valid WRTDS monthly outputs",
            "Remaining sample": (
                "NO3N: {0} stations; PO4P: {1} stations; DOC: {2} stations"
                .format(
                    int(final_by_solute.loc[final_by_solute["solute"] == "NO3N", "n_stations"].iloc[0]),
                    int(final_by_solute.loc[final_by_solute["solute"] == "PO4P", "n_stations"].iloc[0]),
                    int(final_by_solute.loc[final_by_solute["solute"] == "DOC", "n_stations"].iloc[0]),
                )
            ),
            "Purpose": "Define the final constituent-specific analytical samples",
        },
        {
            "Screening step": "Monthly panel construction",
            "Criterion": "OBJECTID-Year-Month alignment across WRTDS outputs, climate data, nutrient inputs, and attributes",
            "Remaining sample": f"{n_final_records:,} station-month-constituent records; {n_final_unique_stations} unique stations",
            "Purpose": "Support monthly compound-state, anomaly, and c-Q analyses",
        },
    ]

    table3 = pd.DataFrame(table3_rows)

    table4 = final_by_solute.rename(
        columns={
            "solute": "Constituent",
            "n_stations": "Number of stations",
            "n_monthly_records": "Number of station-month records",
            "year_min": "Start year",
            "year_max": "End year",
            "n_with_mean_Q": "Records with mean_Q",
            "n_with_mean_C": "Records with mean_C",
            "n_with_mean_Flux": "Records with mean_Flux",
            "n_with_mean_FNC": "Records with mean_FNC",
            "n_with_mean_FNFlux": "Records with mean_FNFlux",
        }
    )

    return table3, table4


# =========================================================
# 5. PREDICTOR INVENTORY
# =========================================================
def classify_attribute(col: str) -> str:
    c = col.lower()

    if col in ["OBJECTID", "Station"]:
        return "Identifier"

    if col in ["Area_km2", "f_AreaGer", "strahler_order", "id_downstream", "n_upstream"]:
        return "Catchment size and network position"

    if c.startswith("dem") or c.startswith("slo") or c.startswith("twi") or c in ["ddhad", "draindens", "flashi", "bfi"]:
        return "Topography and hydrological indices"

    if c.startswith("f_agric") or c.startswith("f_artif") or c.startswith("f_forest") or c.startswith("f_wetl") or c.startswith("f_water") or c.startswith("f_urban") or c.startswith("f_industry") or c.startswith("f_mine") or c.startswith("f_arable") or c.startswith("f_pastures") or c.startswith("f_fores") or c.startswith("f_scrub") or c.startswith("f_open"):
        return "Land use and land cover"

    if c in ["pdens", "new", "n_uwwtp"] or "point" in c or "wwtp" in c or "wttp" in c:
        return "Population and point-source pressure"

    if c.startswith("n_") or c.startswith("p_") or "surp" in c or "fert" in c or "manure" in c or "uptake" in c or "removal" in c or "input" in c or "output" in c or "depo" in c:
        return "Nutrient inputs and nutrient pressure"

    if c.startswith("soil") or c.startswith("theta") or c.startswith("waterroots") or c.startswith("dtb") or c.startswith("f_sand") or c.startswith("f_silt") or c.startswith("f_clay") or c.startswith("f_gwsoils"):
        return "Soils and subsurface properties"

    if c.startswith("f_calc") or c.startswith("f_magma") or c.startswith("f_metam") or c.startswith("f_sedim") or c.startswith("f_silic") or c.startswith("f_consol") or c.startswith("f_porous") or c.startswith("f_fissured") or c.startswith("f_hard"):
        return "Geology and hydrogeology"

    if c in ["p_mm", "pet_mm", "ai", "t_mean"] or c.startswith("p_si") or c.startswith("p_lambda") or c.startswith("p_alpha"):
        return "Climate and aridity"

    if c.startswith("q_"):
        return "Discharge statistics"

    if c.startswith("lai_"):
        return "Vegetation seasonality"

    if c.startswith("het") or c.startswith("r2_het") or c.startswith("sdist"):
        return "Spatial heterogeneity and drainage structure"

    return "Other attributes"


def make_predictor_inventory() -> pd.DataFrame:
    attr = read_csv_fallback(ATTR_FILE)

    rows = []
    for col in attr.columns:
        group = classify_attribute(col)

        if group == "Identifier":
            use_in_attribution = "No"
        elif col in ["Q_StartDate", "Q_EndDate", "Q_gaps", "Q_nNAs"]:
            use_in_attribution = "No"
        else:
            use_in_attribution = "Candidate"

        non_missing = attr[col].notna().sum()
        missing_rate = 1 - non_missing / len(attr)

        if pd.api.types.is_numeric_dtype(attr[col]):
            example_range = f"{pd.to_numeric(attr[col], errors='coerce').min():.4g} to {pd.to_numeric(attr[col], errors='coerce').max():.4g}"
        else:
            example_range = simplify_list(attr[col].dropna().unique().tolist(), max_n=5)

        rows.append(
            {
                "Variable": col,
                "Predictor group": group,
                "Use in attribution": use_in_attribution,
                "Non-missing count": int(non_missing),
                "Missing rate": round(missing_rate, 4),
                "Example range / values": example_range,
            }
        )

    inventory = pd.DataFrame(rows)

    group_order = [
        "Catchment size and network position",
        "Topography and hydrological indices",
        "Climate and aridity",
        "Land use and land cover",
        "Soils and subsurface properties",
        "Geology and hydrogeology",
        "Nutrient inputs and nutrient pressure",
        "Population and point-source pressure",
        "Vegetation seasonality",
        "Spatial heterogeneity and drainage structure",
        "Discharge statistics",
        "Other attributes",
        "Identifier",
    ]

    inventory["Predictor group"] = pd.Categorical(
        inventory["Predictor group"],
        categories=group_order,
        ordered=True
    )

    inventory = inventory.sort_values(["Predictor group", "Variable"]).reset_index(drop=True)

    return inventory


# =========================================================
# 6. SAVE
# =========================================================
def main():
    table1 = make_table1()
    table2 = make_table2()
    table3, table4 = make_sample_tables()
    predictor_inventory = make_predictor_inventory()

    table1.to_csv(OUT_TABLE1, index=False, encoding="utf-8-sig")
    table2.to_csv(OUT_TABLE2, index=False, encoding="utf-8-sig")
    table3.to_csv(OUT_TABLE3, index=False, encoding="utf-8-sig")
    table4.to_csv(OUT_TABLE4, index=False, encoding="utf-8-sig")
    predictor_inventory.to_csv(OUT_PREDICTOR_INVENTORY, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
        table1.to_excel(writer, sheet_name="Table1_datasets", index=False)
        table2.to_excel(writer, sheet_name="Table2_constituents", index=False)
        table3.to_excel(writer, sheet_name="Table3_screening", index=False)
        table4.to_excel(writer, sheet_name="Table4_final_sample", index=False)
        predictor_inventory.to_excel(writer, sheet_name="Predictor_inventory", index=False)

    print("\n=== Data chapter tables generated successfully ===")
    print(f"- {OUT_TABLE1}")
    print(f"- {OUT_TABLE2}")
    print(f"- {OUT_TABLE3}")
    print(f"- {OUT_TABLE4}")
    print(f"- {OUT_PREDICTOR_INVENTORY}")
    print(f"- {OUT_EXCEL}")

    print("\n=== Table 3 preview ===")
    print(table3.to_string(index=False))

    print("\n=== Table 4 preview ===")
    print(table4.to_string(index=False))

    print("\n=== Predictor group counts ===")
    print(
        predictor_inventory
        .groupby("Predictor group", observed=False)
        .size()
        .reset_index(name="n_variables")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()