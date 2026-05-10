from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

# =========================================================
# 1. 动态路径解析 (适配 PyCharm 环境)
# =========================================================
# 当前文件在 src/preprocessing/ 下，向上 2 层即为项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 【核心修正】：原始大数据库保持在原位置，使用绝对路径外部挂载读取
RAW_DIR = Path("D:/quadica_project/data_raw/QUADICA_v2/data/contents/data")

FINAL_DIR = PROJECT_ROOT / "data_final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SOLUTES = ["NO3N", "PO4P", "DOC"]


# =========================================================
# 2. 辅助函数：安全读取与对数计算
# =========================================================
def read_csv_fallback(path: Path, **kwargs) -> pd.DataFrame:
    """
    尝试使用多种编码读取 CSV 文件，防止出现 UnicodeDecodeError (如德文特殊字符)。
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_error = e
    raise RuntimeError(f"无法读取文件 {path}，已尝试所有编码。最后报错: {last_error}")


def safe_log(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return np.log(s.where(s > 0))


# =========================================================
# 3. 核心合并逻辑
# =========================================================
def main():
    print("开始构建综合月尺度基表 (Monthly Master Panel)...")

    # 1. 读取基础表 (使用 fallback 机制解决特殊字符编码问题)
    attr = read_csv_fallback(RAW_DIR / "attributes.csv", low_memory=False)
    meta_q = read_csv_fallback(RAW_DIR / "metadata_q.csv", low_memory=False)
    meta_c = read_csv_fallback(RAW_DIR / "metadata_c.csv", low_memory=False)
    wrtds_summary = read_csv_fallback(RAW_DIR / "wrtds_summary.csv", low_memory=False)

    # 2. 提取核心站点属性
    station_core = attr.merge(meta_q[["OBJECTID", "Station_Q", "Q_flag"]], on="OBJECTID", how="left")

    # 3. 提取目标变量质量信息 (筛选具备 Q 匹配且 WRTDS 质量过关的站点)
    solute_info = meta_c[meta_c["solute"].isin(TARGET_SOLUTES)].merge(
        wrtds_summary[wrtds_summary["solute"].isin(TARGET_SOLUTES)],
        on=["OBJECTID", "solute"], how="left"
    )
    solute_info = solute_info.merge(station_core[["OBJECTID", "Q_flag"]], on="OBJECTID", how="left")

    solute_info["has_wrtds"] = solute_info["wrtds_n"].notna()
    solute_info["wrtds_primary_pass"] = (
            solute_info["has_wrtds"] &
            (solute_info["wrtds_n"] >= 150) &
            (solute_info["wrtds_ts_length"] >= 20)
    )
    solute_info["main_sample_candidate"] = solute_info["wrtds_primary_pass"] & (solute_info["Q_flag"] == 1)

    # 4. 读取动态时间序列数据
    wrtds_monthly = read_csv_fallback(RAW_DIR / "time_series" / "wrtds_monthly.csv", low_memory=False)
    wrtds_monthly = wrtds_monthly[wrtds_monthly["solute"].isin(TARGET_SOLUTES)]
    wrtds_monthly["Date"] = pd.to_datetime(wrtds_monthly["Date"], errors="coerce")
    wrtds_monthly["Year"] = wrtds_monthly["Date"].dt.year
    wrtds_monthly["Month"] = wrtds_monthly["Date"].dt.month

    climate = read_csv_fallback(RAW_DIR / "climate_monthly.csv", low_memory=False)
    climate["Date"] = pd.to_datetime(climate["Date"], errors="coerce")
    climate["Year"] = climate["Date"].dt.year
    climate["Month"] = climate["Date"].dt.month
    climate = climate[["OBJECTID", "Year", "Month", "tavg", "pre", "pet"]].drop_duplicates()

    # 5. 构建总面板 (Panel)
    panel = wrtds_monthly.merge(solute_info[["OBJECTID", "solute", "main_sample_candidate"]], on=["OBJECTID", "solute"],
                                how="left")
    panel = panel[panel["main_sample_candidate"] == True].copy()  # 仅保留主样本

    panel = panel.merge(climate, on=["OBJECTID", "Year", "Month"], how="left")
    panel["log_mean_Q"] = safe_log(panel["mean_Q"])
    panel["log_mean_C"] = safe_log(panel["mean_C"])

    panel = panel.sort_values(["OBJECTID", "solute", "Year", "Month"]).reset_index(drop=True)

    # 输出结果
    output_path = FINAL_DIR / "analysis_panel_station_month_main.csv"
    panel.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[完成] 面板数据已生成，共 {len(panel)} 行，包含气象与水质信息。保存在: {output_path}")


if __name__ == "__main__":
    main()