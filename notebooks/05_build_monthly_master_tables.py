from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data_raw" / "QUADICA_v2"
INTERMEDIATE_DIR = PROJECT_ROOT / "data_intermediate"
FINAL_DIR = PROJECT_ROOT / "data_final"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SOLUTES = ["NO3N", "PO4P", "DOC"]


def find_file_by_name(root: Path, filename: str) -> Optional[Path]:
    matches = list(root.rglob(filename))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[WARN] 多个同名文件 {filename}，默认使用第一个: {matches[0]}")
    return matches[0]


def read_table_with_fallback(path: Path, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, nrows=nrows)
        return df, "excel"

    if suffix == ".parquet":
        df = pd.read_parquet(path)
        if nrows is not None:
            df = df.head(nrows)
        return df, "parquet"

    if suffix in {".csv", ".txt"}:
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        last_error = None
        for enc in encodings:
            try:
                if suffix == ".csv":
                    df = pd.read_csv(path, encoding=enc, low_memory=False, nrows=nrows)
                else:
                    df = pd.read_csv(path, sep=None, engine="python", encoding=enc, nrows=nrows)
                return df, enc
            except Exception as e:
                last_error = e
        raise RuntimeError(f"所有编码尝试都失败: {last_error}")

    raise RuntimeError(f"不支持的文件类型: {suffix}")


def parse_date_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    out["Year"] = out[col].dt.year
    out["Month"] = out[col].dt.month
    out["YearMonth"] = out[col].dt.to_period("M").astype(str)
    return out

def safe_log(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s > 0)
    return np.log(s)

def load_raw_tables():
    raw_files = {
        "wrtds_monthly": "wrtds_monthly.csv",
        "climate_monthly": "climate_monthly.csv",
        "input_N_P": "input_N_P.csv",
    }

    tables = {}
    for key, fname in raw_files.items():
        fpath = find_file_by_name(RAW_DIR, fname)
        if fpath is None:
            raise FileNotFoundError(f"未找到原始文件: {fname}")
        df, enc = read_table_with_fallback(fpath, nrows=None)
        tables[key] = df
        print(f"[OK] raw {key} loaded, encoding={enc}, shape={df.shape}")

    return tables


def load_master_tables():
    station_master_core_path = INTERMEDIATE_DIR / "station_master_core.csv"
    station_solute_master_path = INTERMEDIATE_DIR / "station_solute_master.csv"

    if not station_master_core_path.exists():
        raise FileNotFoundError(f"未找到底表: {station_master_core_path}")
    if not station_solute_master_path.exists():
        raise FileNotFoundError(f"未找到底表: {station_solute_master_path}")

    station_master_core, _ = read_table_with_fallback(station_master_core_path, nrows=None)
    station_solute_master, _ = read_table_with_fallback(station_solute_master_path, nrows=None)

    print(f"[OK] station_master_core loaded, shape={station_master_core.shape}")
    print(f"[OK] station_solute_master loaded, shape={station_solute_master.shape}")

    return station_master_core, station_solute_master


def prepare_wrtds_monthly(wrtds_monthly: pd.DataFrame) -> pd.DataFrame:
    required = {"OBJECTID", "Date", "solute"}
    missing = required - set(wrtds_monthly.columns)
    if missing:
        raise KeyError(f"wrtds_monthly 缺少必要字段: {missing}")

    df = wrtds_monthly.copy()
    df = df[df["solute"].isin(TARGET_SOLUTES)].copy()
    df = parse_date_column(df, "Date")

    # 删除日期无法解析的行
    df = df[df["Year"].notna() & df["Month"].notna()].copy()
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)

    return df


def prepare_climate_monthly(climate_monthly: pd.DataFrame) -> pd.DataFrame:
    required = {"OBJECTID", "Date"}
    missing = required - set(climate_monthly.columns)
    if missing:
        raise KeyError(f"climate_monthly 缺少必要字段: {missing}")

    df = climate_monthly.copy()
    df = parse_date_column(df, "Date")
    df = df[df["Year"].notna() & df["Month"].notna()].copy()
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)

    keep_cols = [c for c in ["OBJECTID", "Date", "Year", "Month", "YearMonth", "tavg", "pre", "pet"] if c in df.columns]
    df = df[keep_cols].drop_duplicates(subset=["OBJECTID", "Year", "Month"])

    return df


def prepare_input_np(input_np: pd.DataFrame) -> pd.DataFrame:
    required = {"OBJECTID", "Year"}
    missing = required - set(input_np.columns)
    if missing:
        raise KeyError(f"input_N_P 缺少必要字段: {missing}")

    df = input_np.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"].notna()].copy()
    df["Year"] = df["Year"].astype(int)
    df = df.drop_duplicates(subset=["OBJECTID", "Year"])

    return df


def build_wrtds_monthly_master(
    wrtds_monthly: pd.DataFrame,
    station_solute_master: pd.DataFrame,
    station_master_core: pd.DataFrame
) -> pd.DataFrame:
    # 从 station_solute_master 取变量级主键信息和样本筛选标记
    solute_cols = [
        "OBJECTID", "solute", "n", "StartYear", "EndYear",
        "wrtds_n", "wrtds_ts_length", "wrtds_gap_max", "wrtds_R2", "wrtds_pbias",
        "wrtds_gap_fraction", "has_wrtds", "has_q_match",
        "wrtds_primary_pass", "main_sample_candidate"
    ]
    solute_cols = [c for c in solute_cols if c in station_solute_master.columns]
    solute_key = station_solute_master[solute_cols].drop_duplicates(subset=["OBJECTID", "solute"])

    master = wrtds_monthly.merge(
        solute_key,
        on=["OBJECTID", "solute"],
        how="left"
    )

    master = master.merge(
        station_master_core,
        on="OBJECTID",
        how="left"
    )

    return master


def build_analysis_panel(
    wrtds_monthly_master: pd.DataFrame,
    climate_monthly: pd.DataFrame,
    input_np: pd.DataFrame
) -> pd.DataFrame:
    panel = wrtds_monthly_master.merge(
        climate_monthly,
        on=["OBJECTID", "Year", "Month"],
        how="left",
        suffixes=("", "_clim")
    )

    panel = panel.merge(
        input_np,
        on=["OBJECTID", "Year"],
        how="left",
        suffixes=("", "_inp")
    )

    panel["has_climate_match"] = panel["tavg"].notna() if "tavg" in panel.columns else False
    panel["has_input_match"] = panel["Year"].notna()

    if "mean_Q" in panel.columns:
        panel["log_mean_Q"] = safe_log(panel["mean_Q"])

    if "mean_C" in panel.columns:
        panel["log_mean_C"] = safe_log(panel["mean_C"])

    if "median_C" in panel.columns:
        panel["log_median_C"] = safe_log(panel["median_C"])

    # 排序
    sort_cols = [c for c in ["OBJECTID", "solute", "Year", "Month"] if c in panel.columns]
    panel = panel.sort_values(sort_cols).reset_index(drop=True)

    return panel


def make_summary_report(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for sol in TARGET_SOLUTES:
        sub = panel[panel["solute"] == sol].copy()

        rows.append({
            "solute": sol,
            "n_rows": len(sub),
            "n_unique_stations": sub["OBJECTID"].nunique(),
            "n_main_sample_rows": int(sub["main_sample_candidate"].fillna(False).sum()),
            "n_main_sample_stations": sub.loc[sub["main_sample_candidate"].fillna(False), "OBJECTID"].nunique(),
            "year_min": pd.to_numeric(sub["Year"], errors="coerce").min(),
            "year_max": pd.to_numeric(sub["Year"], errors="coerce").max(),
            "n_with_climate": int(sub["has_climate_match"].fillna(False).sum()) if "has_climate_match" in sub.columns else None,
            "n_with_mean_Q": int(sub["mean_Q"].notna().sum()) if "mean_Q" in sub.columns else None,
            "n_with_mean_C": int(sub["mean_C"].notna().sum()) if "mean_C" in sub.columns else None,
            "n_with_mean_Flux": int(sub["mean_Flux"].notna().sum()) if "mean_Flux" in sub.columns else None,
            "n_with_mean_FNC": int(sub["mean_FNC"].notna().sum()) if "mean_FNC" in sub.columns else None,
            "n_with_mean_FNFlux": int(sub["mean_FNFlux"].notna().sum()) if "mean_FNFlux" in sub.columns else None,
        })

    return pd.DataFrame(rows)


def main():
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("RAW_DIR =", RAW_DIR)

    raw_tables = load_raw_tables()
    station_master_core, station_solute_master = load_master_tables()

    wrtds_monthly = prepare_wrtds_monthly(raw_tables["wrtds_monthly"])
    climate_monthly = prepare_climate_monthly(raw_tables["climate_monthly"])
    input_np = prepare_input_np(raw_tables["input_N_P"])

    wrtds_monthly_target = wrtds_monthly.copy()

    wrtds_monthly_master = build_wrtds_monthly_master(
        wrtds_monthly=wrtds_monthly_target,
        station_solute_master=station_solute_master,
        station_master_core=station_master_core
    )

    analysis_panel_station_month = build_analysis_panel(
        wrtds_monthly_master=wrtds_monthly_master,
        climate_monthly=climate_monthly,
        input_np=input_np
    )

    analysis_panel_station_month_main = analysis_panel_station_month[
        analysis_panel_station_month["main_sample_candidate"].fillna(False)
    ].copy()

    summary_report = make_summary_report(analysis_panel_station_month)

    # 输出文件
    wrtds_monthly_target.to_csv(
        INTERMEDIATE_DIR / "wrtds_monthly_target.csv",
        index=False,
        encoding="utf-8-sig"
    )

    wrtds_monthly_master.to_csv(
        INTERMEDIATE_DIR / "wrtds_monthly_master.csv",
        index=False,
        encoding="utf-8-sig"
    )

    analysis_panel_station_month.to_csv(
        FINAL_DIR / "analysis_panel_station_month.csv",
        index=False,
        encoding="utf-8-sig"
    )

    analysis_panel_station_month_main.to_csv(
        FINAL_DIR / "analysis_panel_station_month_main.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary_report.to_csv(
        REPORT_DIR / "monthly_panel_summary_report.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== 月尺度主表已生成 ===")
    print(f"wrtds_monthly_target: {wrtds_monthly_target.shape}")
    print(f"wrtds_monthly_master: {wrtds_monthly_master.shape}")
    print(f"analysis_panel_station_month: {analysis_panel_station_month.shape}")
    print(f"analysis_panel_station_month_main: {analysis_panel_station_month_main.shape}")

    print("\n=== 月尺度主表摘要 ===")
    print(summary_report.to_string(index=False))

    print("\n输出文件位置：")
    print(f"- {INTERMEDIATE_DIR / 'wrtds_monthly_target.csv'}")
    print(f"- {INTERMEDIATE_DIR / 'wrtds_monthly_master.csv'}")
    print(f"- {FINAL_DIR / 'analysis_panel_station_month.csv'}")
    print(f"- {FINAL_DIR / 'analysis_panel_station_month_main.csv'}")
    print(f"- {REPORT_DIR / 'monthly_panel_summary_report.csv'}")


if __name__ == "__main__":
    main()