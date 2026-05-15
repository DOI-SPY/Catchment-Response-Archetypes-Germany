from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np

# =========================================================
# 1. 动态路径解析
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = PROJECT_ROOT / "data_final"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

IN_PANEL = FINAL_DIR / "analysis_panel_with_anomalies.csv"
IN_CENTERS = OUTPUT_TABLES_DIR / "archetype_cluster_centers.csv"
IN_CAUSAL = OUTPUT_TABLES_DIR / "causal_network_edges_physically_corrected.csv"

OUT_TABLE1 = OUTPUT_TABLES_DIR / "Manuscript_Table1_Anomaly_Summary.csv"
OUT_TABLE2 = OUTPUT_TABLES_DIR / "Manuscript_Table2_Archetype_Profiles.csv"
OUT_TABLE3 = OUTPUT_TABLES_DIR / "Manuscript_Table3_Causal_Links.csv"


# =========================================================
# 2. 辅助函数：安全读取
# =========================================================
def read_csv_fallback(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except:
            pass
    raise RuntimeError(f"无法读取文件 {path}")


# =========================================================
# 3. 表格生成逻辑
# =========================================================

def generate_table1_anomaly_summary(panel: pd.DataFrame):
    """
    Table 1: Summary statistics of concentration and flux anomalies across hydrological states.
    目标：展示不同水文状态下，各类溶质异常的统计中位数和四分位距 (IQR)。
    """
    print("正在生成 Table 1 (异常值全景统计)...")

    # 仅保留核心状态
    states = ["drought_like", "post_drought_rewetting", "non_drought_highflow", "normal"]
    df = panel[panel["hydro_state"].isin(states)].copy()

    results = []
    for solute in ["NO3N", "PO4P", "DOC"]:
        sub_df = df[df["solute"] == solute]
        for state in states:
            state_df = sub_df[sub_df["hydro_state"] == state]
            if len(state_df) == 0: continue

            row = {
                "Solute": solute,
                "Hydrological State": state.replace("_", " ").title(),
                "N (Solute-Station-Months)": len(state_df),
                "CA Median": f"{state_df['CA'].median():.3f}",
                "CA (25th-75th)": f"[{state_df['CA'].quantile(0.25):.3f}, {state_df['CA'].quantile(0.75):.3f}]",
                "FA Median": f"{state_df['FA'].median():.3f}",
                "FA (25th-75th)": f"[{state_df['FA'].quantile(0.25):.3f}, {state_df['FA'].quantile(0.75):.3f}]",
            }
            results.append(row)

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_TABLE1, index=False, encoding="utf-8-sig")
    print(f"  -> 已保存至: {OUT_TABLE1.name}")


def generate_table2_archetype_profiles(centers: pd.DataFrame):
    """
    Table 2: Multivariate characterization of catchment response archetypes.
    目标：将聚类中心提取出来，并加上物理定名，作为论文解释流域分类的基石。
    """
    print("正在生成 Table 2 (流域响应原型多维特征矩阵)...")

    names = {
        "NO3N": {0: "Buffered Chemostatic", 1: "Source-Depleted Dilution"},
        "PO4P": {0: "Surface-Pathway Activation", 1: "Point-Source Dilution"},
        "DOC": {0: "Non-responsive", 1: "Accumulation-Flush Pulse"}
    }

    centers = centers.copy()

    # 动态插入物理命名
    def get_name(row):
        return names.get(row["solute"], {}).get(row["Archetype"], "Unknown")

    centers.insert(2, "Archetype Name (Mechanism)", centers.apply(get_name, axis=1))

    # 重命名列以符合学术规范
    centers.rename(columns={
        "solute": "Solute",
        "Archetype": "Class ID",
        "CA_rewet": "CA during Rewetting (Median)",
        "FA_rewet": "FA Export (Median)",
        "Delta_Beta": "Δβ (c-Q Shift)",
        "Delta_CV_Ratio": "ΔCVc/CVq (ΔCV Ratio)"
    }, inplace=True)

    # 保留 3 位小数
    for col in ["CA during Rewetting (Median)", "FA Export (Median)", "Δβ (c-Q Shift)", "ΔCVc/CVq (ΔCV Ratio)"]:
        centers[col] = centers[col].apply(lambda x: f"{x:.3f}")

    centers.to_csv(OUT_TABLE2, index=False, encoding="utf-8-sig")
    print(f"  -> 已保存至: {OUT_TABLE2.name}")


def generate_table3_causal_links(causal_edges: pd.DataFrame):
    """
    Table 3: Causal drivers of catchment response archetypes discovered by PC-Algorithm.
    目标：将复杂网络图转化为易于阅读的文本表格。
    """
    print("正在生成 Table 3 (结构因果链路解析表)...")

    # 清洗变量名
    var_map = {
        "Elevation": "Elevation (Topography)",
        "Aridity": "Aridity Index (Climate)",
        "Agri_Frac": "Agricultural Fraction (Land Use)",
        "Avg_Drought_Memory": "Drought Memory (CMD_3m)",
        "Archetype": "Response Archetype (Terminal)"
    }

    causal_edges = causal_edges.copy()
    causal_edges["Cause Variable"] = causal_edges["Cause"].map(var_map)
    causal_edges["Effect Variable"] = causal_edges["Effect"].map(var_map)

    out_df = causal_edges[["Solute", "Cause Variable", "Edge_Type", "Effect Variable"]].copy()
    out_df.sort_values(by=["Solute", "Cause Variable"], inplace=True)

    out_df.to_csv(OUT_TABLE3, index=False, encoding="utf-8-sig")
    print(f"  -> 已保存至: {OUT_TABLE3.name}")


# =========================================================
# 4. 主控程序
# =========================================================
def main():
    if not IN_PANEL.exists():
        raise FileNotFoundError("找不到面板数据，请先跑通 03 脚本。")

    panel = read_csv_fallback(IN_PANEL)
    centers = read_csv_fallback(IN_CENTERS)
    causal = read_csv_fallback(IN_CAUSAL)

    print("\n[开始构建论文附件表格体系]")
    generate_table1_anomaly_summary(panel)
    generate_table2_archetype_profiles(centers)
    generate_table3_causal_links(causal)
    print("\n[全部表格生成完毕] 现可直接导入 Word 或 LaTeX。")


if __name__ == "__main__":
    main()