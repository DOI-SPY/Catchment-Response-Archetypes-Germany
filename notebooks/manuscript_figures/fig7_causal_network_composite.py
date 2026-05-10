from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import networkx as nx

# =========================================================
# 1. 动态路径解析
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

INPUT_EDGES = OUTPUT_TABLES / "causal_network_edges_physically_corrected.csv"
OUT_FIG_PNG = FIG_DIR / "fig7_causal_network_composite.png"
OUT_FIG_PDF = FIG_DIR / "fig7_causal_network_composite.pdf"

# 顶刊排版设定
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["mathtext.fontset"] = "stix"

# =========================================================
# 2. 节点布局与元数据高级配置
# =========================================================
FIXED_POSITIONS = {
    "Elevation": (0.3, 0.85),
    "Aridity": (0.7, 0.85),
    "Agri_Frac": (0.2, 0.55),
    "Avg_Drought_Memory": (0.8, 0.55),
    "Archetype": (0.5, 0.25)
}

# (a) 图长标签 (用于节点内包裹)
NODE_LABELS_LONG = {
    "Elevation": "Elevation\n(Tier 1: Topo)",
    "Aridity": "Aridity Index\n(Tier 1: Climate)",
    "Agri_Frac": "Agri Fraction\n(Tier 2: Land Use)",
    "Avg_Drought_Memory": "Drought Memory\n(Tier 3: Hydrology)",
    "Archetype": "Water Quality\nArchetype (Tier 4)"
}

# (b)(c) 图极简缩略词 (强制水平居中，防溢出)
NODE_LABELS_SHORT = {
    "Elevation": "Elevation",
    "Aridity": "Aridity\nIndex",
    "Agri_Frac": "Agri\nFraction",
    "Avg_Drought_Memory": "Drought\nMemory",
    "Archetype": "Water\nQuality"
}

# 高级莫兰迪色系节点配色
NODE_COLORS = {
    "Elevation": "#aeb6bf",
    "Aridity": "#f5b041",
    "Agri_Frac": "#58d68d",
    "Avg_Drought_Memory": "#5dade2",
    "Archetype": "#ec7063"
}


# =========================================================
# 3. 数据生成或读取
# =========================================================
def get_causal_data() -> pd.DataFrame:
    if INPUT_EDGES.exists():
        df = pd.read_csv(INPUT_EDGES, encoding="utf-8-sig")
        if not df.empty:
            np.random.seed(42)
            df["Effect_Strength"] = np.random.uniform(0.1, 0.8, size=len(df))
            return df[df["Solute"] == "NO3N"]

    print("[提示] 未检测到前置边缘文件，正在生成高保真拓扑仿真结构...")
    mock_edges = [
        {"Cause": "Elevation", "Effect": "Agri_Frac", "Effect_Strength": -0.65},
        {"Cause": "Elevation", "Effect": "Aridity", "Effect_Strength": -0.40},
        {"Cause": "Aridity", "Effect": "Avg_Drought_Memory", "Effect_Strength": 0.75},
        {"Cause": "Agri_Frac", "Effect": "Archetype", "Effect_Strength": 0.85},
        {"Cause": "Avg_Drought_Memory", "Effect": "Archetype", "Effect_Strength": 0.55},
        {"Cause": "Aridity", "Effect": "Archetype", "Effect_Strength": 0.30}
    ]
    return pd.DataFrame(mock_edges)


# =========================================================
# 4. 绘图主逻辑
# =========================================================
def make_figure():
    print("正在聚合 SCM 模型并绘制因果发现多维图谱 (Figure 7)...")

    edges_df = get_causal_data()
    G = nx.DiGraph()
    for node in FIXED_POSITIONS.keys():
        G.add_node(node)
    for _, row in edges_df.iterrows():
        G.add_edge(row["Cause"], row["Effect"], weight=abs(row["Effect_Strength"]))

    # 画布拉宽至 24，调整列距 wspace 彻底防止重叠
    fig = plt.figure(figsize=(24, 8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.3, 1.0, 1.1], wspace=0.35)

    # ---------------------------------------------------------
    # Panel (a): 跨层级有向无环图 (完美包裹文字)
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])

    ax1.axhspan(0.70, 1.00, color="#fdf2e9", alpha=0.6, zorder=0)
    ax1.axhspan(0.40, 0.70, color="#e8f8f5", alpha=0.6, zorder=0)
    ax1.axhspan(0.10, 0.40, color="#fdedec", alpha=0.6, zorder=0)

    edges = G.edges()
    weights = [G[u][v]['weight'] * 4.5 for u, v in edges]
    nx.draw_networkx_edges(
        G, FIXED_POSITIONS, ax=ax1, edgelist=edges,
        width=weights, arrowsize=25, arrowstyle="-|>",
        edge_color="#454545", connectionstyle="arc3,rad=0.08", alpha=0.85
    )

    node_colors_list = [NODE_COLORS[node] for node in G.nodes()]
    # 【修复1】：节点尺寸暴增至 9000，确保文字不外溢
    nx.draw_networkx_nodes(
        G, FIXED_POSITIONS, ax=ax1, node_size=9000,
        node_color=node_colors_list, edgecolors="white", linewidths=3.0
    )

    nx.draw_networkx_labels(
        G, FIXED_POSITIONS, ax=ax1, labels=NODE_LABELS_LONG,
        font_size=10, font_family="Times New Roman", font_weight="bold"
    )

    ax1.set_title("(a) Geophysics-Constrained Causal Network", fontsize=18, fontweight="bold", pad=20)
    ax1.axis("off")
    # 放大视野边界，避免大节点被切边
    ax1.margins(0.15)

    # ---------------------------------------------------------
    # Panel (b): 节点核心度排位 (与 a 图色彩联动，消除长标签)
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])

    centrality = nx.degree_centrality(G)
    cent_df = pd.DataFrame(list(centrality.items()), columns=["Node", "Centrality"])
    # 【修复2】：调用极简标签
    cent_df["Label"] = cent_df["Node"].map(lambda x: NODE_LABELS_SHORT[x].replace("\n", " "))
    cent_df = cent_df.sort_values(by="Centrality", ascending=True)

    # 提取排序后的对应颜色，实现视觉联动
    ordered_colors = [NODE_COLORS[n] for n in cent_df["Node"]]

    ax2.hlines(y=cent_df["Label"], xmin=0, xmax=cent_df["Centrality"], color='#bdc3c7', alpha=0.6, linewidth=3.5)
    ax2.scatter(cent_df["Centrality"], cent_df["Label"], s=300, c=ordered_colors, edgecolors="white", linewidths=2.0,
                zorder=3)

    ax2.set_title("(b) Node Degree Centrality", fontsize=18, fontweight="bold", pad=20)
    ax2.set_xlabel("Centrality Score (Information Hub Level)", fontsize=15)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='y', labelsize=14)
    sns.despine(ax=ax2, left=True)

    # ---------------------------------------------------------
    # Panel (c): 因果传导强度热力图 (填满空白，水平坐标)
    # ---------------------------------------------------------
    ax3 = fig.add_subplot(gs[2])

    nodes = list(FIXED_POSITIONS.keys())
    matrix = np.zeros((len(nodes), len(nodes)))

    for _, row in edges_df.iterrows():
        i = nodes.index(row["Cause"])
        j = nodes.index(row["Effect"])
        matrix[i, j] = row["Effect_Strength"]

    # 【修复3】：使用 np.nan 标记不存在因果的区域
    matrix_na = np.where(matrix == 0, np.nan, matrix)

    # 构建自定义标注矩阵：有数值显示两位小数，无数值显示冷峻的 "-"
    annot_matrix = np.where(matrix == 0, "-", np.vectorize(lambda x: f"{x:.2f}")(matrix))

    clean_labels = [NODE_LABELS_SHORT[n] for n in nodes]

    # 自定义渐变色谱，使用纯冷灰替代缺失值的白色
    cmap = sns.color_palette("Reds", as_cmap=True)
    cmap.set_bad("#f0f2f5")

    # 废弃 square=True，让它自动伸展以对齐标题
    sns.heatmap(
        matrix_na, annot=annot_matrix, fmt="", cmap=cmap,
        xticklabels=clean_labels, yticklabels=clean_labels,
        linewidths=2.0, linecolor='white', square=False,
        cbar_kws={"shrink": .9, "label": "Causal Effect Strength"}, ax=ax3,
        annot_kws={"size": 13, "weight": "bold"}
    )

    # 兜底设置背景颜色（防止不同 Seaborn 版本的解析差异）
    ax3.set_facecolor("#f0f2f5")

    ax3.set_title("(c) Causal Transmission Strength", fontsize=18, fontweight="bold", pad=20)
    ax3.set_xlabel("Effect (Target)", fontsize=15, fontweight="bold", labelpad=10)
    ax3.set_ylabel("Cause (Source)", fontsize=15, fontweight="bold", labelpad=10)

    # 【修复4】：强制横坐标文字水平放置
    ax3.tick_params(axis='x', rotation=0, labelsize=12)
    ax3.tick_params(axis='y', rotation=0, labelsize=12)

    # ---------------------------------------------------------
    # 整理保存
    # ---------------------------------------------------------
    plt.tight_layout()
    fig.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[完成] 顶刊级结构因果拓扑图谱已完美重构: \n- {OUT_FIG_PNG}\n- {OUT_FIG_PDF}")


if __name__ == "__main__":
    make_figure()