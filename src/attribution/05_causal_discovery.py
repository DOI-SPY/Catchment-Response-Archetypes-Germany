from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

# 导入因果推断核心库
from causallearn.search.ConstraintBased.PC import pc

# =========================================================
# 1. 动态路径解析
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path("D:/quadica_project/data_raw/QUADICA_v2/data/contents/data")
FINAL_DIR = PROJECT_ROOT / "data_final"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IN_ARCHETYPES = FINAL_DIR / "catchment_response_archetypes.csv"
IN_PANEL = FINAL_DIR / "analysis_panel_with_anomalies.csv"
OUT_CAUSAL_EDGES = OUTPUT_DIR / "causal_network_edges_physically_corrected.csv"

TARGET_SOLUTES = ["NO3N", "PO4P", "DOC"]

# =========================================================
# 2. 地学先验层级定义 (Geophysical Tiers)
# =========================================================
# 数值越小，在自然界中越基础，因果链越靠前
GEOPHYSICAL_TIERS = {
    "Elevation": 1,
    "Aridity": 1,
    "Agri_Frac": 2,
    "Avg_Drought_Memory": 3,
    "Archetype": 4
}


# =========================================================
# 3. 辅助函数：安全读取与智能列名匹配
# =========================================================
def read_csv_fallback(path: Path, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_error = e
    raise RuntimeError(f"无法读取文件 {path}，最后报错: {last_error}")


def get_actual_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_cols = {str(c).lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    raise KeyError(f"未找到候选列名: {candidates}")


# =========================================================
# 4. 构建因果网络数据集
# =========================================================
def build_causal_dataset(solute: str) -> pd.DataFrame:
    archetypes = read_csv_fallback(IN_ARCHETYPES)
    solute_arch = archetypes[archetypes["solute"] == solute][["OBJECTID", "Archetype"]]

    attr = read_csv_fallback(RAW_DIR / "attributes.csv", low_memory=False)

    col_dem = get_actual_col(attr, ["dem.mean", "dem_mean", "elevation"])
    col_agric = get_actual_col(attr, ["f_agric", "f_agric_18", "agri_frac"])
    col_ai = get_actual_col(attr, ["ai", "aridity_index", "aridity"])

    attr_sub = attr[["OBJECTID", col_dem, col_agric, col_ai]].copy()
    attr_sub.rename(columns={
        col_dem: "Elevation",
        col_agric: "Agri_Frac",
        col_ai: "Aridity"
    }, inplace=True)

    panel = read_csv_fallback(IN_PANEL)
    solute_panel = panel[panel["solute"] == solute]

    rewet_events = solute_panel[solute_panel["hydro_state"] == "post_drought_rewetting"]
    memory_feature = rewet_events.groupby("OBJECTID")["CMD_3m"].mean().reset_index()
    memory_feature.rename(columns={"CMD_3m": "Avg_Drought_Memory"}, inplace=True)

    causal_df = solute_arch.merge(attr_sub, on="OBJECTID", how="inner")
    causal_df = causal_df.merge(memory_feature, on="OBJECTID", how="left")

    return causal_df.dropna()


# =========================================================
# 5. 执行 PC 算法 + 地学先验校正
# =========================================================
def run_pc_algorithm_with_priors(causal_df: pd.DataFrame, solute: str) -> pd.DataFrame:
    node_names = ["Elevation", "Agri_Frac", "Aridity", "Avg_Drought_Memory", "Archetype"]
    for col in node_names:
        causal_df[col] = pd.to_numeric(causal_df[col], errors='coerce')
    causal_df = causal_df.dropna(subset=node_names)

    data_matrix = causal_df[node_names].values

    # 1. 运行纯数据驱动的 PC 算法
    cg = pc(data_matrix, alpha=0.05, indep_test='fisherz', show_progress=False)
    adj_matrix = cg.G.graph

    edges = []
    num_nodes = len(node_names)

    # 2. 解析边缘并进行地学先验校正 (Geophysical Prior Correction)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i >= j: continue  # 避免重复处理无向边或自环

            node_i = node_names[i]
            node_j = node_names[j]
            tier_i = GEOPHYSICAL_TIERS[node_i]
            tier_j = GEOPHYSICAL_TIERS[node_j]

            # 判断是否有连接 (无论是 -1 还是 1，只要不为0说明有联系)
            if adj_matrix[i, j] != 0 or adj_matrix[j, i] != 0:

                # 强制运用物理先验定出因果方向：低 Tier -> 高 Tier
                if tier_i < tier_j:
                    cause, effect = node_i, node_j
                elif tier_i > tier_j:
                    cause, effect = node_j, node_i
                else:
                    # 同一 Tier 的情况 (如 Elevation 和 Aridity)，保留其无向属性或由算法决定的方向
                    if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                        cause, effect = node_i, node_j
                    elif adj_matrix[i, j] == 1 and adj_matrix[j, i] == -1:
                        cause, effect = node_j, node_i
                    else:
                        edges.append({"Solute": solute, "Cause": node_i, "Effect": node_j,
                                      "Edge_Type": "Correlated (Tier equivalent)"})
                        continue

                edges.append({
                    "Solute": solute,
                    "Cause": cause,
                    "Effect": effect,
                    "Edge_Type": "Directed Causation (->)"
                })

    return pd.DataFrame(edges)


# =========================================================
# 6. 主控程序
# =========================================================
def main():
    if not IN_ARCHETYPES.exists():
        raise FileNotFoundError(f"找不到聚类原型结果，请先运行 04 脚本。")

    all_edges = []

    print("\n[开始因果推断] 算法架构: 数据驱动(PC-Algorithm) + 物理定律(Geophysical Priors)")
    for solute in TARGET_SOLUTES:
        causal_df = build_causal_dataset(solute)
        if len(causal_df) < 50:
            continue

        edges_df = run_pc_algorithm_with_priors(causal_df, solute)
        if not edges_df.empty:
            all_edges.append(edges_df)

    if all_edges:
        final_causal_network = pd.concat(all_edges, ignore_index=True)
        # 去重，防止同一有向边被算法多次录入
        final_causal_network = final_causal_network.drop_duplicates()
        final_causal_network.to_csv(OUT_CAUSAL_EDGES, index=False, encoding="utf-8-sig")

        print("\n[物理约束因果模型构建完成]")
        print("\n[地学规律校正后的真实因果链预览]")
        print(final_causal_network.to_string(index=False))


if __name__ == "__main__":
    main()