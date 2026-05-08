from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# =========================
# 0. CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data_raw" / "QUADICA_v2"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 需要重点查找的关键词
TARGET_KEYWORDS = {
    "discharge": [
        "discharge", "q", "flow", "streamflow"
    ],
    "constituents": [
        "no3", "nitrate", "po4", "phosphate", "orthophosphate", "doc",
        "dissolved organic carbon"
    ],
    "wrtds_outputs": [
        "flow_normalized", "flow-normalized", "fnc", "flux", "mean_flux",
        "monthly_median", "median_concentration"
    ],
    "attributes": [
        "catchment", "basin", "attribute", "climate", "soil", "landuse",
        "topography", "lithology"
    ],
    "nutrient_inputs": [
        "nutrient", "nitrogen", "phosphorus", "input", "surplus", "point_source",
        "diffuse", "n_input", "p_input"
    ],
}

SUPPORTED_EXTS = {".csv", ".txt", ".xlsx", ".xls", ".parquet"}


# =========================
# 1. HELPER FUNCTIONS
# =========================
def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.strip().lower())


def list_data_files(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return sorted(files)


def safe_read_table(path: Path, nrows: int = 20) -> Tuple[pd.DataFrame | None, str]:
    """
    只读取前几行做字段普查，避免一上来把超大文件全读入。
    """
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path, nrows=nrows, low_memory=False)
        elif suffix == ".txt":
            df = pd.read_csv(path, sep=None, engine="python", nrows=nrows)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path, nrows=nrows)
        elif suffix == ".parquet":
            df = pd.read_parquet(path).head(nrows)
        else:
            return None, f"Unsupported extension: {suffix}"
        return df, ""
    except Exception as e:
        return None, str(e)


def detect_keywords_in_columns(columns: List[str]) -> Dict[str, List[str]]:
    normalized_cols = [normalize_text(c) for c in columns]
    hits: Dict[str, List[str]] = {}
    for group, words in TARGET_KEYWORDS.items():
        matched = []
        for col_raw, col_norm in zip(columns, normalized_cols):
            if any(word in col_norm for word in words):
                matched.append(col_raw)
        hits[group] = matched
    return hits


def score_table(columns: List[str]) -> Dict[str, int]:
    hits = detect_keywords_in_columns(columns)
    return {k: len(v) for k, v in hits.items()}


# =========================
# 2. AUDIT
# =========================
def audit_dataset() -> None:
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("RAW_DIR =", RAW_DIR)
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw data folder not found: {RAW_DIR}\n"
            "请先把 QUADICA v2 数据解压到 data_raw/QUADICA_v2/ 下。"
        )

    files = list_data_files(RAW_DIR)

    inventory_rows = []
    failed_rows = []

    for file_path in files:
        df, err = safe_read_table(file_path, nrows=20)
        if df is None:
            failed_rows.append({
                "file": str(file_path.relative_to(PROJECT_ROOT)),
                "error": err
            })
            continue

        columns = [str(c) for c in df.columns]
        hits = detect_keywords_in_columns(columns)
        scores = score_table(columns)

        inventory_rows.append({
            "file": str(file_path.relative_to(PROJECT_ROOT)),
            "extension": file_path.suffix.lower(),
            "n_preview_rows": len(df),
            "n_columns": len(columns),
            "columns_preview": " | ".join(columns[:20]),
            "discharge_cols": "; ".join(hits["discharge"]),
            "constituent_cols": "; ".join(hits["constituents"]),
            "wrtds_cols": "; ".join(hits["wrtds_outputs"]),
            "attribute_cols": "; ".join(hits["attributes"]),
            "nutrient_input_cols": "; ".join(hits["nutrient_inputs"]),
            "score_discharge": scores["discharge"],
            "score_constituents": scores["constituents"],
            "score_wrtds": scores["wrtds_outputs"],
            "score_attributes": scores["attributes"],
            "score_nutrient_inputs": scores["nutrient_inputs"],
        })

    inventory_df = pd.DataFrame(inventory_rows)
    failed_df = pd.DataFrame(failed_rows)

    if inventory_df.empty:
        raise RuntimeError("未成功读取任何支持的数据文件，请检查数据解压是否完整。")

    inventory_path = REPORT_DIR / "quadica_file_inventory.csv"
    inventory_df.sort_values(
        by=["score_constituents", "score_discharge", "score_wrtds", "score_attributes"],
        ascending=False
    ).to_csv(inventory_path, index=False, encoding="utf-8-sig")

    failed_path = REPORT_DIR / "quadica_file_read_failures.csv"
    failed_df.to_csv(failed_path, index=False, encoding="utf-8-sig")

    candidates = {
        "likely_discharge_tables": inventory_df[inventory_df["score_discharge"] > 0]
        .sort_values(by=["score_discharge", "score_constituents"], ascending=False)
        .head(10),
        "likely_quality_tables": inventory_df[inventory_df["score_constituents"] > 0]
        .sort_values(by=["score_constituents", "score_wrtds"], ascending=False)
        .head(10),
        "likely_wrtds_tables": inventory_df[inventory_df["score_wrtds"] > 0]
        .sort_values(by=["score_wrtds", "score_constituents"], ascending=False)
        .head(10),
        "likely_attribute_tables": inventory_df[inventory_df["score_attributes"] > 0]
        .sort_values(by=["score_attributes"], ascending=False)
        .head(10),
        "likely_nutrient_input_tables": inventory_df[inventory_df["score_nutrient_inputs"] > 0]
        .sort_values(by=["score_nutrient_inputs"], ascending=False)
        .head(10),
    }

    summary = {
        "n_files_detected": len(files),
        "n_files_read_success": int(len(inventory_df)),
        "n_files_read_failed": int(len(failed_df)),
        "acceptance_check": {
            "has_discharge_candidate": not candidates["likely_discharge_tables"].empty,
            "has_quality_candidate": not candidates["likely_quality_tables"].empty,
            "has_wrtds_candidate": not candidates["likely_wrtds_tables"].empty,
            "has_attribute_candidate": not candidates["likely_attribute_tables"].empty,
            "has_nutrient_input_candidate": not candidates["likely_nutrient_input_tables"].empty,
        },
    }

    summary_path = REPORT_DIR / "quadica_audit_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for name, df in candidates.items():
        out_path = REPORT_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n=== QUADICA v2 数据普查完成 ===")
    print(f"文件总数: {summary['n_files_detected']}")
    print(f"成功读取: {summary['n_files_read_success']}")
    print(f"读取失败: {summary['n_files_read_failed']}")
    print("\n=== 验收检查 ===")
    for k, v in summary["acceptance_check"].items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")

    print(f"\n详细清单已输出到: {REPORT_DIR}")


if __name__ == "__main__":
    audit_dataset()