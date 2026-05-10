from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.lines as mlines
import matplotlib.patches as patches

# =========================================================
# 1. 动态路径解析
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG_PNG = FIG_DIR / "fig9_seasonal_hierarchical_clustering.png"
OUT_FIG_PDF = FIG_DIR / "fig9_seasonal_hierarchical_clustering.pdf"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["mathtext.fontset"] = "stix"


# =========================================================
# 2. 高保真时空协同仿真数据生成
# =========================================================
def generate_seasonal_clustering_data(n_catchments=150):
    np.random.seed(42)

    # 模拟三种流域机制 (引入适度噪声模拟真实世界的轨迹发散)
    n1 = int(n_catchments * 0.4)
    c1_w = np.random.normal(2.5, 0.4, n1)
    c1_sp = np.random.normal(1.2, 0.3, n1)
    c1_su = np.random.normal(0.2, 0.1, n1)
    c1_a = np.random.normal(0.8, 0.2, n1)

    n2 = int(n_catchments * 0.35)
    c2_w = np.random.normal(0.5, 0.2, n2)
    c2_sp = np.random.normal(0.4, 0.2, n2)
    c2_su = np.random.normal(2.0, 0.5, n2)
    c2_a = np.random.normal(1.8, 0.4, n2)

    n3 = n_catchments - n1 - n2
    c3_w = np.random.normal(0.3, 0.15, n3)
    c3_sp = np.random.normal(0.4, 0.15, n3)
    c3_su = np.random.normal(0.3, 0.15, n3)
    c3_a = np.random.normal(0.5, 0.2, n3)

    winter = np.concatenate([c1_w, c2_w, c3_w])
    spring = np.concatenate([c1_sp, c2_sp, c3_sp])
    summer = np.concatenate([c1_su, c2_su, c3_su])
    autumn = np.concatenate([c1_a, c2_a, c3_a])

    df = pd.DataFrame({"Winter": winter, "Spring": spring, "Summer": summer, "Autumn": autumn})
    df = df.sample(frac=1, random_state=10).reset_index(drop=True)  # 打乱模拟聚类过程
    df.index = [f"Basin_{i + 1}" for i in range(len(df))]
    df = df.clip(lower=0)

    return df


# =========================================================
# 3. 绘图主逻辑 (顶刊轨迹流形与嵌套架构重构)
# =========================================================
def make_figure():
    print("正在计算 Ward 层次聚类并绘制高精时空流形图谱 (Figure 9 终极版)...")

    data = generate_seasonal_clustering_data()
    Z = linkage(data.values, method='ward', metric='euclidean')
    cluster_labels = fcluster(Z, t=3, criterion='maxclust')
    data['Cluster'] = cluster_labels

    # ---------------------------------------------------------
    # 【排版重构】：绝对安全的嵌套网格，彻底阻断拉伸
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(22, 8.5))
    # 主网格划分为左右两部分，wspace 适度推开
    gs_main = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.1], wspace=0.25)

    # 左半部分再次切分给 (a)树状图 和 (b)热力图，必须极度贴合 (wspace=0.01)
    gs_left = gs_main[0].subgridspec(1, 2, width_ratios=[0.3, 0.7], wspace=0.01)

    ax1 = fig.add_subplot(gs_left[0])  # (a) Dendrogram
    ax2 = fig.add_subplot(gs_left[1])  # (b) Heatmap
    ax3 = fig.add_subplot(gs_main[1])  # (c) Trajectory Flow

    # 高级莫兰迪/地球科学配色
    hex_colors = ["#2980B9", "#E67E22", "#27AE60"]  # 蓝(冬季型), 橘(夏秋型), 绿(稳定型)
    cluster_names_map = {1: "Winter-Dominated (Flushing)", 2: "Summer/Autumn (Storm-Driven)",
                         3: "Chemostatic (Resistant)"}

    # =========================================================
    # Panel (a): 层次聚类树状图 - 精修留白与阈值线
    # =========================================================
    from scipy.cluster.hierarchy import set_link_color_palette
    set_link_color_palette(hex_colors)
    cut_distance = Z[-3, 2]

    dendro = dendrogram(
        Z, orientation='left', ax=ax1,
        color_threshold=cut_distance,
        above_threshold_color='#7F8C8D',
        no_labels=True
    )

    # 树枝加粗
    for coll in ax1.collections:
        coll.set_linewidth(2.0)

    ax1.axvline(cut_distance, color="#C0392B", linestyle="--", linewidth=2.5, alpha=0.8, zorder=0)

    # 【修复1】：动态智能裁切冗长的左侧“废根”，彻底消除留白
    ax1.set_xlim(cut_distance * 1.5, 0)

    # 完善坐标轴与标题
    ax1.set_title("(a) Hierarchical Clustering", fontsize=18, fontweight="bold", pad=20)
    ax1.set_xlabel("Ward Linkage Distance", fontsize=15, fontweight="bold")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.set_yticks([])

    # 确保 X 轴刻度文本加粗
    ax1.tick_params(axis='x', labelsize=12)
    for label in ax1.get_xticklabels():
        label.set_fontweight("bold")

    # =========================================================
    # Panel (b): 重排序季节响应热力图 - 极简贴合
    # =========================================================
    ordered_indices = dendro['leaves']
    heatmap_data = data.drop(columns=['Cluster']).iloc[ordered_indices]
    ordered_clusters = data['Cluster'].iloc[ordered_indices].values

    cmap_hm = sns.color_palette("rocket_r", as_cmap=True)

    sns.heatmap(
        heatmap_data, ax=ax2, cmap=cmap_hm, cbar=False,
        xticklabels=True, yticklabels=False, linewidths=0, rasterized=True
    )

    ax2.set_title("(b) Seasonal Anomaly Probability Matrix", fontsize=18, fontweight="bold", pad=20)
    ax2.set_xlabel("Hydrological Season", fontsize=15, fontweight="bold")
    ax2.tick_params(axis='x', labelsize=14, labelrotation=0)
    for label in ax2.get_xticklabels():
        label.set_fontweight("bold")
    ax2.set_ylabel("")

    # 【美学机制】：在热力图左侧(贴合树状图的位置) 绘制一根彩色分类彩带 (Cluster Color Bar)
    # 这能极大地增强审稿人对“哪些行属于哪棵树”的认知映射
    y_min, y_max = ax2.get_ylim()
    total_rows = len(ordered_indices)

    for i in range(total_rows):
        cluster_id = ordered_clusters[i]
        c_color = hex_colors[cluster_id - 1]
        # 绘制极细的矩形作为左侧色带
        rect = patches.Rectangle((-0.15, i), 0.15, 1, clip_on=False, facecolor=c_color, edgecolor='none')
        ax2.add_patch(rect)

    # 绘制内部白色分割线
    changes = np.where(ordered_clusters[:-1] != ordered_clusters[1:])[0]
    for y_line in changes:
        ax2.axhline(y_line + 1, color="white", linewidth=2.5, linestyle="-")

    # 安全的 Colorbar (挂载在热力图右侧内部边界，绝不导致撑破画布)
    cbar_ax = ax2.inset_axes([1.02, 0.05, 0.03, 0.9])
    sm = plt.cm.ScalarMappable(cmap=cmap_hm, norm=plt.Normalize(vmin=0, vmax=heatmap_data.values.max()))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Anomaly Amplitude (CA)", fontsize=13, fontweight="bold", labelpad=15, rotation=270)
    cbar.outline.set_visible(False)

    # =========================================================
    # Panel (c): 意面图+流形叠合 (Spaghetti Trajectories & Mean Flow)
    # =========================================================
    plot_data = data.copy()
    plot_data['Basin_ID'] = plot_data.index
    plot_data['Cluster_Name'] = plot_data['Cluster'].map(cluster_names_map)
    melted_df = plot_data.melt(id_vars=['Basin_ID', 'Cluster', 'Cluster_Name'],
                               value_vars=['Winter', 'Spring', 'Summer', 'Autumn'],
                               var_name='Season', value_name='Anomaly')

    season_mapping = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}
    melted_df['Season_Num'] = melted_df['Season'].map(season_mapping)

    # 1. 绘制背景微观轨迹 (Spaghetti Lines)
    # 这带来了极致的信息密度和物理流淌感
    for basin in plot_data['Basin_ID'].unique():
        basin_data = melted_df[melted_df['Basin_ID'] == basin].sort_values('Season_Num')
        c_id = basin_data['Cluster'].iloc[0]
        color = hex_colors[c_id - 1]
        ax3.plot(basin_data['Season_Num'], basin_data['Anomaly'],
                 color=color, alpha=0.15, linewidth=1.0, zorder=1)

    # 2. 绘制前景宏观统计流形 (Mean + 95% CI)
    sns.lineplot(
        data=melted_df, x="Season_Num", y="Anomaly", hue="Cluster_Name",
        palette={cluster_names_map[i + 1]: hex_colors[i] for i in range(3)},
        marker="o", markersize=14, linewidth=4.5, err_style="band", errorbar=("ci", 95),
        ax=ax3, sort=False, zorder=5, legend=False, markeredgecolor="white", markeredgewidth=2
    )

    # 辅助基线
    ax3.axhline(0, color="#7F8C8D", linestyle="--", linewidth=1.5, zorder=0)

    # 图表修饰
    ax3.set_title("(c) Catchment Longitudinal Trajectories", fontsize=18, fontweight="bold", pad=20)
    ax3.set_ylabel("Concentration Anomaly Amplitude (CA)", fontsize=15, fontweight="bold")
    ax3.set_xlabel("Hydrological Season", fontsize=15, fontweight="bold")

    ax3.set_xticks(range(4))
    ax3.set_xticklabels(["Winter", "Spring", "Summer", "Autumn"], fontsize=14, fontweight="bold")
    ax3.tick_params(axis='y', labelsize=13)
    for label in ax3.get_yticklabels():
        label.set_fontweight("bold")

    ax3.set_xlim(-0.2, 3.2)

    # 优雅的内置图例 (绝对安全防撑破)
    legend_elements = [
        mlines.Line2D([0], [0], color=hex_colors[0], lw=4, marker='o', markersize=10, label=cluster_names_map[1]),
        mlines.Line2D([0], [0], color=hex_colors[1], lw=4, marker='o', markersize=10, label=cluster_names_map[2]),
        mlines.Line2D([0], [0], color=hex_colors[2], lw=4, marker='o', markersize=10, label=cluster_names_map[3])
    ]

    # 【修复】：移除 kwargs 中的 zorder，分离对象属性赋值
    leg = ax3.legend(handles=legend_elements, title="Archetype / Cluster", loc="upper center",
                     bbox_to_anchor=(0.5, 0.98), frameon=True, fontsize=12, title_fontsize=13,
                     edgecolor="#cccccc", facecolor="white", framealpha=0.9)
    leg.set_zorder(10)

    sns.despine(ax=ax3)

    # ---------------------------------------------------------
    # 保存输出
    # ---------------------------------------------------------
    plt.tight_layout()
    plt.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight")
    plt.savefig(OUT_FIG_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[完成] 季节特征树状流形图谱已完成颠覆性重构: \n- {OUT_FIG_PNG}\n- {OUT_FIG_PDF}")


if __name__ == "__main__":
    make_figure()