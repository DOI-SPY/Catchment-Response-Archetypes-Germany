from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data_raw" / "QUADICA_v2"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


TARGET_FILES = {
    "attributes": "attributes.csv",
    "climate_monthly": "climate_monthly.csv",
    "input_N_P": "input_N_P.csv",
    "metadata_q": "metadata_q.csv",
    "metadata_c": "metadata_c.csv",
    "wrtds_summary": "wrtds_summary.csv",
}


def find_file_by_name(root: Path, filename: str) -> Optional[Path]:
    matches = list(root.rglob(filename))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[WARN] 找到多个 {filename}，默认使用第一个: {matches[0]}")
    return matches[0]


def read_preview(path: Path, nrows: int = 5) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def inspect_core_tables() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"未找到数据目录: {RAW_DIR}")

    summary_rows = []

    for key, fname in TARGET_FILES.items():
        fpath = find_file_by_name(RAW_DIR, fname)

        if fpath is None:
            print(f"[MISSING] {fname}")
            summary_rows.append({
                "table_key": key,
                "filename": fname,
                "found": False,
                "path": "",
                "n_columns": "",
                "columns": "",
            })
            continue

        print(f"\n===== {key} =====")
        print(f"PATH: {fpath}")

        try:
            df_preview = read_preview(fpath, nrows=5)
            cols = [str(c) for c in df_preview.columns]

            print("COLUMNS:")
            for i, c in enumerate(cols, start=1):
                print(f"  {i:02d}. {c}")

            print("\nPREVIEW:")
            print(df_preview.head(3).to_string(index=False))

            # 保存字段名
            col_df = pd.DataFrame({
                "column_order": range(1, len(cols) + 1),
                "column_name": cols
            })
            col_df.to_csv(
                REPORT_DIR / f"{key}_columns.csv",
                index=False,
                encoding="utf-8-sig"
            )

            # 保存预览
            df_preview.to_csv(
                REPORT_DIR / f"{key}_preview.csv",
                index=False,
                encoding="utf-8-sig"
            )

            summary_rows.append({
                "table_key": key,
                "filename": fname,
                "found": True,
                "path": str(fpath.relative_to(PROJECT_ROOT)),
                "n_columns": len(cols),
                "columns": " | ".join(cols[:20]),
            })

        except Exception as e:
            print(f"[ERROR] 读取 {fname} 失败: {e}")
            summary_rows.append({
                "table_key": key,
                "filename": fname,
                "found": True,
                "path": str(fpath.relative_to(PROJECT_ROOT)),
                "n_columns": "ERROR",
                "columns": str(e),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        REPORT_DIR / "core_tables_inspection_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== 核心表检查完成 ===")
    print(f"结果已输出到: {REPORT_DIR}")


if __name__ == "__main__":
    inspect_core_tables()