from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data_raw" / "QUADICA_v2"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTS = {".csv", ".txt", ".xlsx", ".xls", ".parquet"}

TARGET_FILES = {
    "attributes": "attributes.csv",
    "metadata_q": "metadata_q.csv",
    "metadata_c": "metadata_c.csv",
    "wrtds_summary": "wrtds_summary.csv",
    "input_N_P": "input_N_P.csv",
}


def find_file_by_name(root: Path, filename: str) -> Optional[Path]:
    matches = list(root.rglob(filename))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[WARN] 多个同名文件 {filename}，默认使用第一个: {matches[0]}")
    return matches[0]


def read_table_with_fallback(path: Path, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    """
    尝试多种编码读取 CSV/TXT；Excel/Parquet 直接读。
    返回: (DataFrame, 使用的编码或说明)
    """
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


def inspect_named_table(table_key: str, filename: str) -> None:
    fpath = find_file_by_name(RAW_DIR, filename)
    if fpath is None:
        print(f"[MISSING] {filename}")
        return

    print(f"\n===== {table_key} =====")
    print(f"PATH: {fpath}")

    df_preview, used_encoding = read_table_with_fallback(fpath, nrows=5)
    cols = [str(c) for c in df_preview.columns]

    print(f"ENCODING_USED: {used_encoding}")
    print("COLUMNS:")
    for i, c in enumerate(cols, start=1):
        print(f"  {i:02d}. {c}")

    print("\nPREVIEW:")
    print(df_preview.head(3).to_string(index=False))

    pd.DataFrame({
        "column_order": range(1, len(cols) + 1),
        "column_name": cols
    }).to_csv(REPORT_DIR / f"{table_key}_columns_v2.csv", index=False, encoding="utf-8-sig")

    df_preview.to_csv(REPORT_DIR / f"{table_key}_preview_v2.csv", index=False, encoding="utf-8-sig")


def inspect_solutes() -> None:
    metadata_c_path = find_file_by_name(RAW_DIR, "metadata_c.csv")
    wrtds_summary_path = find_file_by_name(RAW_DIR, "wrtds_summary.csv")

    if metadata_c_path is not None:
        metadata_c, enc = read_table_with_fallback(metadata_c_path, nrows=None)
        if "solute" in metadata_c.columns:
            solutes = sorted(metadata_c["solute"].dropna().astype(str).unique().tolist())
            pd.DataFrame({"solute": solutes}).to_csv(
                REPORT_DIR / "metadata_c_unique_solutes.csv",
                index=False,
                encoding="utf-8-sig"
            )
            print("\n===== metadata_c unique solutes =====")
            print(solutes[:50])

            target_mask = metadata_c["solute"].astype(str).str.contains(
                r"NO3|Nitrate|PO4|Phosphate|DOC", case=False, regex=True, na=False
            )
            metadata_c[target_mask].to_csv(
                REPORT_DIR / "metadata_c_target_solutes_rows.csv",
                index=False,
                encoding="utf-8-sig"
            )

    if wrtds_summary_path is not None:
        wrtds_summary, enc = read_table_with_fallback(wrtds_summary_path, nrows=None)
        if "solute" in wrtds_summary.columns:
            solutes = sorted(wrtds_summary["solute"].dropna().astype(str).unique().tolist())
            pd.DataFrame({"solute": solutes}).to_csv(
                REPORT_DIR / "wrtds_summary_unique_solutes.csv",
                index=False,
                encoding="utf-8-sig"
            )

            target_mask = wrtds_summary["solute"].astype(str).str.contains(
                r"NO3|Nitrate|PO4|Phosphate|DOC", case=False, regex=True, na=False
            )
            wrtds_summary[target_mask].to_csv(
                REPORT_DIR / "wrtds_summary_target_solutes_rows.csv",
                index=False,
                encoding="utf-8-sig"
            )


def inspect_time_series_folder() -> None:
    ts_dir_candidates = list(RAW_DIR.rglob("time_series"))
    if not ts_dir_candidates:
        print("\n[MISSING] 未找到 time_series 文件夹")
        return

    ts_dir = ts_dir_candidates[0]
    print(f"\n===== time_series folder =====")
    print(f"PATH: {ts_dir}")

    rows = []
    for p in ts_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            rows.append({
                "relative_path": str(p.relative_to(PROJECT_ROOT)),
                "filename": p.name,
                "suffix": p.suffix.lower(),
                "parent": str(p.parent.relative_to(PROJECT_ROOT)),
                "size_kb": round(p.stat().st_size / 1024, 2),
            })

    ts_inventory = pd.DataFrame(rows).sort_values(by=["parent", "filename"])
    ts_inventory.to_csv(REPORT_DIR / "time_series_inventory.csv", index=False, encoding="utf-8-sig")

    print(f"time_series 文件数: {len(ts_inventory)}")

    # 对前若干文件做预览，看看哪些最像主分析表
    preview_rows = []
    for _, row in ts_inventory.head(30).iterrows():
        fpath = PROJECT_ROOT / row["relative_path"]
        try:
            df_preview, used_encoding = read_table_with_fallback(fpath, nrows=3)
            cols = [str(c) for c in df_preview.columns]
            preview_rows.append({
                "relative_path": row["relative_path"],
                "encoding_used": used_encoding,
                "n_columns": len(cols),
                "columns_preview": " | ".join(cols[:20]),
            })
        except Exception as e:
            preview_rows.append({
                "relative_path": row["relative_path"],
                "encoding_used": "ERROR",
                "n_columns": "ERROR",
                "columns_preview": str(e),
            })

    pd.DataFrame(preview_rows).to_csv(
        REPORT_DIR / "time_series_preview_top30.csv",
        index=False,
        encoding="utf-8-sig"
    )


def main() -> None:
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("RAW_DIR =", RAW_DIR)

    for key, fname in TARGET_FILES.items():
        inspect_named_table(key, fname)

    inspect_solutes()
    inspect_time_series_folder()

    print("\n=== 第三轮检查完成 ===")
    print(f"输出目录: {REPORT_DIR}")


if __name__ == "__main__":
    main()