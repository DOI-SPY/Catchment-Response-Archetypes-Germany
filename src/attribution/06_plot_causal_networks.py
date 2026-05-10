from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches

# =========================================================
# 1. 动态路径解析
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_EDGES = PROJECT_ROOT / "outputs" / "tables" / "causal_network_edges_physically_corrected.csv"
OUT_FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG_PNG = OUT_FIG_DIR / "fig7_causal_networks.png"
OUT_FIG_PDF = OUT_FIG_DIR / "fig7_causal_networks.pdf"

TARGET_SOLUTES = ["NO3N", "PO4P", "DOC"]

# =========================================================
# 2. 节点坐标布局 (基于地学先验 Tiers 强制定制)
# =========================================================
# 让图表呈现自上而下的物理控制逻辑
FIXED_POSITIONS = {
    "Elevation": (0.2, 3.0),  # Tier 1 (左上)
    "Aridity": (0.8, 3.0),  # Tier 1 (右上)
    "Agri_Frac": (0.2, 2.0),  # Tier 2 (左中)
    "Avg_Drought_Memory": (0.8, 2.0),  # Tier 3 (右中)
    "Archetype": (0.5, 0.5)  # Tier 4 (底部中央, 终端响应)
}

# 节点别名展示 (为了图表美观)
NODE_LABELS = {
    "Elevation": "Elevation\n(Topography)",
    "Aridity": "Aridity Index\n(Climate)",
    "Agri_Frac": "Agri. Fraction\n(Land Use)",
    "Avg_Drought_Memory": "Drought Memory\n(Hydrol. Forcing)",
    "Archetype": "Response\nArchetype"
}


# =========================================================
# 3. 绘图主逻辑
# =========================================================
def plot_causal_dags(edges_df: pd.DataFrame):
    print("正在绘制结构因果模型 (SCM) 拓扑图...")

    # 全局字体设置
    plt.rcParams["font.family"] = "Times New Roman"

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.1)

    colors = {"NO3N": "#2f7f73", "PO4P": "#c48a3a", "DOC": "#7b5ea7"}

    for ax, solute in zip(axes, TARGET_SOLUTES):
        ax.set_title(f"Causal Network: {solute}", fontsize=16, fontweight="bold", pad=20)

        # 筛选该溶质的边
        solute_edges = edges_df[edges_df["Solute"] == solute].copy()

        # 构建有向图
        G = nx.DiGraph()
        # 强制添加所有节点（即使孤立）
        for node in FIXED_POSITIONS.keys():
            G.add_node(node)

        # 添加边
        for _, row in solute_edges.iterrows():
            G.add_edge(row["Cause"], row["Effect"])

        # 绘制节点
        nx.draw_networkx_nodes(
            G, FIXED_POSITIONS, ax=ax,
            node_size=4000,
            node_color="white",
            edgecolors=colors[solute],
            linewidths=2.5
        )

        # 绘制标签
        nx.draw_networkx_labels(
            G, FIXED_POSITIONS, ax=ax,
            labels=NODE_LABELS,
            font_size=11,
            font_family="Times New Roman",
            font_weight="bold"
        )

        # 绘制有向边
        nx.draw_networkx_edges(
            G, FIXED_POSITIONS, ax=ax,
            node_size=4000,
            arrowstyle="-|>",
            arrowsize=25,
            edge_color="#555555",
            width=2.0,
            connectionstyle="arc3,rad=0.05"  # 微弱的弧线使双向或交叉边更清晰
        )

        # 移除坐标轴外框
        ax.axis("off")

        # 绘制虚线层级背景以辅助视觉理解
        ax.axhline(y=2.5, color="gray", linestyle="--", alpha=0.3, zorder=-1)
        ax.axhline(y=1.5, color="gray", linestyle="--", alpha=0.3, zorder=-1)

        ax.text(0.5, 3.3, "Tier 1: Intrinsic Boundaries", ha="center", fontsize=10, color="gray", style="italic")
        ax.text(0.5, 2.3, "Tier 2 & 3: Human & Hydrological Forcings", ha="center", fontsize=10, color="gray",
                style="italic")
        ax.text(0.5, 0.1, "Tier 4: Catchment Response", ha="center", fontsize=10, color="gray", style="italic")

    # 整体保存
    plt.tight_layout()
    fig.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表绘制完成] 已保存至: \n{OUT_FIG_PNG}\n{OUT_FIG_PDF}")


def main():
    if not INPUT_EDGES.exists():
        raise FileNotFoundError(f"找不到因果边文件: {INPUT_EDGES}，请先运行 05 脚本。")

    df = pd.read_csv(INPUT_EDGES, encoding="utf-8-sig")
    plot_causal_dags(df)


if __name__ == "__main__":
    main()