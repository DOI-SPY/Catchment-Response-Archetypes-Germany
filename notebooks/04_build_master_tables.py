from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data_raw" / "QUADICA_v2"
INTERMEDIATE_DIR = PROJECT_ROOT / "data_intermediate"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
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


def load_required_tables():
    file_map = {
        "attributes": "attributes.csv",
        "metadata_q": "metadata_q.csv",
        "metadata_c": "metadata_c.csv",
        "wrtds_summary": "wrtds_summary.csv",
        "input_N_P": "input_N_P.csv",
    }

    tables = {}
    for key, fname in file_map.items():
        fpath = find_file_by_name(RAW_DIR, fname)
        if fpath is None:
            raise FileNotFoundError(f"未找到文件: {fname}")
        df, enc = read_table_with_fallback(fpath, nrows=None)
        tables[key] = df
        print(f"[OK] {key} loaded with encoding={enc}, shape={df.shape}")

    return tables


def build_station_master(attributes: pd.DataFrame, metadata_q: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 全量站点主表
    station_master_full = attributes.merge(
        metadata_q,
        on="OBJECTID",
        how="left",
        suffixes=("_attr", "_q")
    )

    # 核心站点主表（保留最重要字段，后面写论文和画图更方便）
    candidate_cols = [
        "OBJECTID",
        "Station_attr", "Station_q", "Station", "Station_Q", "River",
        "Area_km2_attr", "Area_km2_q", "Area_km2",
        "Q_flag",
        "f_agric", "f_forest", "f_artif", "f_urban",
        "AI", "T_mean", "P_mm", "PET_mm",
        "Q_StartDate", "Q_EndDate", "Q_mean", "Q_median", "Q_spec", "BFI", "flashi",
        "DrainDens", "strahler_order",
        "N_T_YKM2", "P_T_YKM2",
        "P_input", "P_surplus",
        "Npoint_Pop_from1991", "Npoint_WWTP_from1991", "Nsurp_Batool_from1991",
        "Ppoint_Pop_from1991", "Ppoint_WWTP_from1991", "Psurp_Batool_from1991",
        "id_downstream", "n_upstream"
    ]

    existing_cols = [c for c in candidate_cols if c in station_master_full.columns]
    station_master_core = station_master_full[existing_cols].copy()

    return station_master_full, station_master_core


def build_station_solute_master(
    metadata_c: pd.DataFrame,
    wrtds_summary: pd.DataFrame,
    station_master_core: pd.DataFrame
) -> pd.DataFrame:
    # 只保留主文目标变量
    c_target = metadata_c[metadata_c["solute"].isin(TARGET_SOLUTES)].copy()
    w_target = wrtds_summary[wrtds_summary["solute"].isin(TARGET_SOLUTES)].copy()

    # 合并元数据和 WRTDS 质量信息
    station_solute_master = c_target.merge(
        w_target,
        on=["OBJECTID", "solute"],
        how="left",
        suffixes=("", "_wrtds")
    )

    # 再拼接站点信息
    station_solute_master = station_solute_master.merge(
        station_master_core,
        on="OBJECTID",
        how="left"
    )

    # 计算 WRTDS gap 占总序列比例
    if {"wrtds_gap_max", "wrtds_ts_length"}.issubset(station_solute_master.columns):
        station_solute_master["wrtds_gap_fraction"] = (
            station_solute_master["wrtds_gap_max"] /
            (station_solute_master["wrtds_ts_length"] * 365.25)
        )

    # 是否存在 WRTDS 结果
    station_solute_master["has_wrtds"] = station_solute_master["wrtds_n"].notna()

    # 第一版主文高质量 WRTDS 标准
    station_solute_master["wrtds_primary_pass"] = (
        station_solute_master["has_wrtds"].fillna(False)
        & (station_solute_master["wrtds_n"] >= 150)
        & (station_solute_master["wrtds_ts_length"] >= 20)
        & (station_solute_master["wrtds_gap_fraction"] <= 0.20)
    )

    # 是否有 discharge 匹配
    if "Q_flag" in station_solute_master.columns:
        station_solute_master["has_q_match"] = station_solute_master["Q_flag"] == 1
    else:
        station_solute_master["has_q_match"] = False

    # 主分析推荐样本：有 WRTDS + 有 Q 匹配
    station_solute_master["main_sample_candidate"] = (
        station_solute_master["wrtds_primary_pass"]
        & station_solute_master["has_q_match"]
    )

    return station_solute_master


def inspect_time_series_folder() -> pd.DataFrame:
    ts_dir_candidates = list(RAW_DIR.rglob("time_series"))
    if not ts_dir_candidates:
        raise FileNotFoundError("未找到 time_series 文件夹")

    ts_dir = ts_dir_candidates[0]
    rows = []

    for p in ts_dir.rglob("*"):
        if p.is_file():
            try:
                df_preview, enc = read_table_with_fallback(p, nrows=3)
                cols = [str(c) for c in df_preview.columns]
                rows.append({
                    "relative_path": str(p.relative_to(PROJECT_ROOT)),
                    "filename": p.name,
                    "encoding": enc,
                    "n_columns": len(cols),
                    "columns_preview": " | ".join(cols[:20]),
                    "preview_first_row": df_preview.head(1).to_dict(orient="records")
                })
            except Exception as e:
                rows.append({
                    "relative_path": str(p.relative_to(PROJECT_ROOT)),
                    "filename": p.name,
                    "encoding": "ERROR",
                    "n_columns": "ERROR",
                    "columns_preview": str(e),
                    "preview_first_row": ""
                })

    return pd.DataFrame(rows)


def make_summary_report(station_solute_master: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for sol in TARGET_SOLUTES:
        sub = station_solute_master[station_solute_master["solute"] == sol].copy()

        rows.append({
            "solute": sol,
            "n_station_solute_rows": len(sub),
            "n_with_q_match": int(sub["has_q_match"].fillna(False).sum()),
            "n_with_wrtds": int(sub["has_wrtds"].fillna(False).sum()),
            "n_wrtds_primary_pass": int(sub["wrtds_primary_pass"].fillna(False).sum()),
            "n_main_sample_candidate": int(sub["main_sample_candidate"].fillna(False).sum()),
            "median_n": pd.to_numeric(sub["n"], errors="coerce").median(),
            "median_wrtds_n": pd.to_numeric(sub["wrtds_n"], errors="coerce").median(),
            "median_wrtds_ts_length": pd.to_numeric(sub["wrtds_ts_length"], errors="coerce").median(),
        })

    return pd.DataFrame(rows)


def main() -> None:
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("RAW_DIR =", RAW_DIR)

    tables = load_required_tables()

    station_master_full, station_master_core = build_station_master(
        tables["attributes"], tables["metadata_q"]
    )

    station_solute_master = build_station_solute_master(
        tables["metadata_c"], tables["wrtds_summary"], station_master_core
    )

    time_series_file_summary = inspect_time_series_folder()
    summary_report = make_summary_report(station_solute_master)

    # 输出底表
    station_master_full.to_csv(
        INTERMEDIATE_DIR / "station_master_full.csv",
        index=False,
        encoding="utf-8-sig"
    )

    station_master_core.to_csv(
        INTERMEDIATE_DIR / "station_master_core.csv",
        index=False,
        encoding="utf-8-sig"
    )

    station_solute_master.to_csv(
        INTERMEDIATE_DIR / "station_solute_master.csv",
        index=False,
        encoding="utf-8-sig"
    )

    time_series_file_summary.to_csv(
        INTERMEDIATE_DIR / "time_series_file_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary_report.to_csv(
        REPORT_DIR / "target_solute_summary_report.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== 第一批底表已生成 ===")
    print(f"station_master_full: {station_master_full.shape}")
    print(f"station_master_core: {station_master_core.shape}")
    print(f"station_solute_master: {station_solute_master.shape}")
    print(f"time_series_file_summary: {time_series_file_summary.shape}")

    print("\n=== 目标变量样本概况 ===")
    print(summary_report.to_string(index=False))

    print("\n输出文件位置：")
    print(f"- {INTERMEDIATE_DIR / 'station_master_full.csv'}")
    print(f"- {INTERMEDIATE_DIR / 'station_master_core.csv'}")
    print(f"- {INTERMEDIATE_DIR / 'station_solute_master.csv'}")
    print(f"- {INTERMEDIATE_DIR / 'time_series_file_summary.csv'}")
    print(f"- {REPORT_DIR / 'target_solute_summary_report.csv'}")


if __name__ == "__main__":
    main()