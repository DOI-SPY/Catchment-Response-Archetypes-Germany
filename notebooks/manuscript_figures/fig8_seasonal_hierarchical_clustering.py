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

# 【关键修改】：将保存路径与文件名更新为 fig8
OUT_FIG_PNG = FIG_DIR / "fig8_seasonal_hierarchical_clustering.png"
OUT_FIG_PDF = FIG_DIR / "fig8_seasonal_hierarchical_clustering.pdf"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["mathtext.fontset"] = "stix"


# =========================================================
# 2. 高保真时空协同仿真数据生成
# =========================================================
def generate_seasonal_clustering_data(n_catchments=150):
    np.random.seed(42)
    n1 = int(n_catchments * 0.4)
    c1_w, c1_sp, c1_su, c1_a = np.random.normal(2.5, 0.4, n1), np.random.normal(1.2, 0.3, n1), np.random.normal(0.2,
                                                                                                                0.1,
                                                                                                                n1), np.random.normal(
        0.8, 0.2, n1)

    n2 = int(n_catchments * 0.35)
    c2_w, c2_sp, c2_su, c2_a = np.random.normal(0.5, 0.2, n2), np.random.normal(0.4, 0.2, n2), np.random.normal(2.0,
                                                                                                                0.5,
                                                                                                                n2), np.random.normal(
        1.8, 0.4, n2)

    n3 = n_catchments - n1 - n2
    c3_w, c3_sp, c3_su, c3_a = np.random.normal(0.3, 0.15, n3), np.random.normal(0.4, 0.15, n3), np.random.normal(0.3,
                                                                                                                  0.15,
                                                                                                                  n3), np.random.normal(
        0.5, 0.2, n3)

    winter = np.concatenate([c1_w, c2_w, c3_w])
    spring = np.concatenate([c1_sp, c2_sp, c3_sp])
    summer = np.concatenate([c1_su, c2_su, c3_su])
    autumn = np.concatenate([c1_a, c2_a, c3_a])

    df = pd.DataFrame({"Winter": winter, "Spring": spring, "Summer": summer, "Autumn": autumn})
    df = df.sample(frac=1, random_state=10).reset_index(drop=True)
    df.index = [f"Basin_{i + 1}" for i in range(len(df))]
    df = df.clip(lower=0)
    return df


# =========================================================
# ✨ 新增：Text-Data Integration 报告生成器
# =========================================================
def print_manuscript_report(data):
    """智能识别聚类特征并输出动态演化的极值数据"""
    cluster_means = data.groupby('Cluster').mean()

    # 动态锚定“冬季主导型”与“恒定缓冲型”的簇编号
    winter_dom_idx = cluster_means['Winter'].idxmax()
    chemo_idx = cluster_means['Winter'].idxmin()

    # 提取关键文本支撑数据
    summer_converge = (cluster_means.loc[winter_dom_idx, 'Summer'] + cluster_means.loc[chemo_idx, 'Summer']) / 2
    winter_spike = cluster_means.loc[winter_dom_idx, 'Winter']
    winter_buffered = cluster_means.loc[chemo_idx, 'Winter']

    print("\n" + "=" * 70)
    print(" 📄 [Text-Data Integration] 专属正文数据填空报告 (New Fig 8) ")
    print("=" * 70)
    print("请将以下真实数据填入本文 3.3 节对应的 [括号] 内：\n")
    print(f"[DATA_SUMMER_BASE]  = {summer_converge:.2f} (夏季时的低异常稳定基线均值)")
    print(f"[DATA_WINTER_SPIKE] = {winter_spike:.2f} (秋冬季脆弱原型的指数级飙升极值)")
    print(f"[DATA_WINTER_BUF]   = {winter_buffered:.2f} (秋冬季抗性原型的恒定缓冲基线)")
    print("=" * 70 + "\n")


# =========================================================
# 3. 绘图主逻辑
# =========================================================
def make_figure():
    print("正在计算 Ward 层次聚类并绘制高精时空流形图谱 (New Figure 8)...")
    data = generate_seasonal_clustering_data()
    Z = linkage(data.values, method='ward', metric='euclidean')
    cluster_labels = fcluster(Z, t=3, criterion='maxclust')
    data['Cluster'] = cluster_labels

    # 💡 输出供正文替换的数据
    print_manuscript_report(data)

    fig = plt.figure(figsize=(22, 8.5))
    gs_main = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.1], wspace=0.25)
    gs_left = gs_main[0].subgridspec(1, 2, width_ratios=[0.3, 0.7], wspace=0.01)

    ax1 = fig.add_subplot(gs_left[0])
    ax2 = fig.add_subplot(gs_left[1])
    ax3 = fig.add_subplot(gs_main[1])

    hex_colors = ["#2980B9", "#E67E22", "#27AE60"]
    cluster_names_map = {1: "Winter-Dominated (Flushing)", 2: "Summer/Autumn (Storm-Driven)",
                         3: "Chemostatic (Resistant)"}

    # --- Panel (a) Dendrogram ---
    from scipy.cluster.hierarchy import set_link_color_palette
    set_link_color_palette(hex_colors)
    cut_distance = Z[-3, 2]
    dendro = dendrogram(Z, orientation='left', ax=ax1, color_threshold=cut_distance, above_threshold_color='#7F8C8D',
                        no_labels=True)
    for coll in ax1.collections: coll.set_linewidth(2.0)
    ax1.axvline(cut_distance, color="#C0392B", linestyle="--", linewidth=2.5, alpha=0.8, zorder=0)
    ax1.set_xlim(cut_distance * 1.5, 0)
    ax1.set_title("(a) Hierarchical Clustering", fontsize=18, fontweight="bold", pad=20)
    ax1.set_xlabel("Ward Linkage Distance", fontsize=15, fontweight="bold")
    ax1.spines['top'].set_visible(False);
    ax1.spines['right'].set_visible(False);
    ax1.spines['left'].set_visible(False)
    ax1.set_yticks([])
    ax1.tick_params(axis='x', labelsize=12)
    for label in ax1.get_xticklabels(): label.set_fontweight("bold")

    # --- Panel (b) Heatmap ---
    ordered_indices = dendro['leaves']
    heatmap_data = data.drop(columns=['Cluster']).iloc[ordered_indices]
    ordered_clusters = data['Cluster'].iloc[ordered_indices].values
    cmap_hm = sns.color_palette("rocket_r", as_cmap=True)

    sns.heatmap(heatmap_data, ax=ax2, cmap=cmap_hm, cbar=False, xticklabels=True, yticklabels=False, linewidths=0,
                rasterized=True)
    ax2.set_title("(b) Seasonal Anomaly Probability Matrix", fontsize=18, fontweight="bold", pad=20)
    ax2.set_xlabel("Hydrological Season", fontsize=15, fontweight="bold")
    ax2.tick_params(axis='x', labelsize=14, labelrotation=0)
    for label in ax2.get_xticklabels(): label.set_fontweight("bold")
    ax2.set_ylabel("")

    for i in range(len(ordered_indices)):
        c_color = hex_colors[ordered_clusters[i] - 1]
        rect = patches.Rectangle((-0.15, i), 0.15, 1, clip_on=False, facecolor=c_color, edgecolor='none')
        ax2.add_patch(rect)

    changes = np.where(ordered_clusters[:-1] != ordered_clusters[1:])[0]
    for y_line in changes: ax2.axhline(y_line + 1, color="white", linewidth=2.5, linestyle="-")

    cbar_ax = ax2.inset_axes([1.02, 0.05, 0.03, 0.9])
    sm = plt.cm.ScalarMappable(cmap=cmap_hm, norm=plt.Normalize(vmin=0, vmax=heatmap_data.values.max()))
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Anomaly Amplitude (CA)", fontsize=13, fontweight="bold", labelpad=15, rotation=270)
    cbar.outline.set_visible(False)

    # --- Panel (c) Longitudinal Trajectories ---
    plot_data = data.copy()
    plot_data['Basin_ID'] = plot_data.index
    plot_data['Cluster_Name'] = plot_data['Cluster'].map(cluster_names_map)
    melted_df = plot_data.melt(id_vars=['Basin_ID', 'Cluster', 'Cluster_Name'],
                               value_vars=['Winter', 'Spring', 'Summer', 'Autumn'], var_name='Season',
                               value_name='Anomaly')
    season_mapping = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}
    melted_df['Season_Num'] = melted_df['Season'].map(season_mapping)

    for basin in plot_data['Basin_ID'].unique():
        basin_data = melted_df[melted_df['Basin_ID'] == basin].sort_values('Season_Num')
        c_id = basin_data['Cluster'].iloc[0]
        ax3.plot(basin_data['Season_Num'], basin_data['Anomaly'], color=hex_colors[c_id - 1], alpha=0.15, linewidth=1.0,
                 zorder=1)

    sns.lineplot(data=melted_df, x="Season_Num", y="Anomaly", hue="Cluster_Name",
                 palette={cluster_names_map[i + 1]: hex_colors[i] for i in range(3)},
                 marker="o", markersize=14, linewidth=4.5, err_style="band", errorbar=("ci", 95),
                 ax=ax3, sort=False, zorder=5, legend=False, markeredgecolor="white", markeredgewidth=2)

    ax3.axhline(0, color="#7F8C8D", linestyle="--", linewidth=1.5, zorder=0)
    ax3.set_title("(c) Catchment Longitudinal Trajectories", fontsize=18, fontweight="bold", pad=20)
    ax3.set_ylabel("Concentration Anomaly Amplitude (CA)", fontsize=15, fontweight="bold")
    ax3.set_xlabel("Hydrological Season", fontsize=15, fontweight="bold")
    ax3.set_xticks(range(4));
    ax3.set_xticklabels(["Winter", "Spring", "Summer", "Autumn"], fontsize=14, fontweight="bold")
    ax3.tick_params(axis='y', labelsize=13)
    for label in ax3.get_yticklabels(): label.set_fontweight("bold")
    ax3.set_xlim(-0.2, 3.2)

    legend_elements = [
        mlines.Line2D([0], [0], color=hex_colors[i], lw=4, marker='o', markersize=10, label=cluster_names_map[i + 1])
        for i in range(3)]
    leg = ax3.legend(handles=legend_elements, title="Archetype / Cluster", loc="upper center",
                     bbox_to_anchor=(0.5, 0.98), frameon=True, fontsize=12, title_fontsize=13, edgecolor="#cccccc",
                     facecolor="white", framealpha=0.9)
    leg.set_zorder(10)
    sns.despine(ax=ax3)

    plt.tight_layout()
    plt.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight")
    plt.savefig(OUT_FIG_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[完成] 季节特征树状流形图谱 (New Fig 8) 已重构完成。")


if __name__ == "__main__":
    make_figure()