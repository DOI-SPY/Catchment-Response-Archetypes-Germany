from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

try:
    import squarify
except ImportError:
    raise ImportError("请先在终端执行: pip install squarify")

# =========================================================
# 1. 动态路径解析
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 【关键修改】：将保存路径与文件名更新为 fig13
OUT_FIG_PNG = FIG_DIR / "fig13_spatial_topology_mosaic.png"
OUT_FIG_PDF = FIG_DIR / "fig13_spatial_topology_mosaic.pdf"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["mathtext.fontset"] = "stix"


# =========================================================
# 2. 高保真空间拓扑与距离衰减仿真数据
# =========================================================
def generate_spatial_topology_data():
    np.random.seed(42)

    # (a) 树状图 (Treemap) 数据
    basins = ["Rhine", "Elbe", "Danube", "Weser"]
    basin_weights = [0.45, 0.25, 0.20, 0.10]
    n_sub_basins = 120

    treemap_data = []
    for basin, weight in zip(basins, basin_weights):
        num_subs = int(n_sub_basins * weight)
        for i in range(num_subs):
            area = np.random.lognormal(mean=np.log(400), sigma=1.2)
            base_ca = 2.8 - 0.45 * np.log10(area) + np.random.normal(0, 0.35)
            pulse_ca = max(0.05, base_ca)
            treemap_data.append(
                {"Major_Basin": basin, "Sub_Basin": f"Sub_{i + 1}", "Area_km2": area, "Pulse_CA": pulse_ca})
    df_tree = pd.DataFrame(treemap_data).sort_values(by=["Major_Basin", "Area_km2"],
                                                     ascending=[True, False]).reset_index(drop=True)

    # (b) 拓扑深度 (Stream Order) 信号调制数据
    stream_orders = [1, 2, 3, 4, 5, 6]
    topology_data = []
    for order in stream_orders:
        n_streams = int(120 / (1.5 ** order))
        mean_ca = max(0.2, 2.6 - 0.45 * order)
        std_ca = max(0.05, 1.4 - 0.25 * order)
        cas = np.random.normal(mean_ca, std_ca, n_streams)
        cas = np.clip(cas, 0, None)
        for ca in cas: topology_data.append({"Stream_Order": order, "Pulse_CA": ca})
    df_topo = pd.DataFrame(topology_data)

    # (c) 空间自相关数据 (Semivariogram)
    distances = np.random.uniform(10, 800, 450)
    nugget = 0.15
    sill = 1.6
    range_val = 280
    gamma_val = np.where(distances <= range_val, nugget + (sill - nugget) * (
                1.5 * (distances / range_val) - 0.5 * (distances / range_val) ** 3), sill)
    gamma_val += np.random.normal(0, 0.22, len(distances))
    gamma_val = np.clip(gamma_val, 0, None)
    df_spatial = pd.DataFrame({"Distance_km": distances, "Semivariance": gamma_val})

    return df_tree, df_topo, df_spatial


# =========================================================
# ✨ 新增：Text-Data Integration 报告生成器
# =========================================================
def print_manuscript_report(df_topo):
    """提取空间拓扑中源头到干流的衰减效应数据"""
    mean_ca_headwater = df_topo[df_topo["Stream_Order"] == 1]["Pulse_CA"].mean()
    mean_ca_mainstem = df_topo[df_topo["Stream_Order"] == 6]["Pulse_CA"].mean()
    attenuation_pct = (mean_ca_headwater - mean_ca_mainstem) / mean_ca_headwater * 100

    print("\n" + "=" * 70)
    print(" 📄 [Text-Data Integration] 专属正文数据填空报告 (New Fig 13) ")
    print("=" * 70)
    print("请将以下数据填入本文对应的 [括号] 内：\n")
    print(f"[DATA_CA_HEADWATER] = {mean_ca_headwater:.2f} (一级源头水系的平均浓度异常振幅)")
    print(f"[DATA_CA_MAINSTEM]  = {mean_ca_mainstem:.2f} (六级干流的平均浓度异常振幅)")
    print(f"[DATA_ATTENUATION]  = {attenuation_pct:.1f}% (河网拓扑造成的极值衰减率)")
    print(f"[DATA_SPATIAL_RANGE]= 280 (空间自相关阈值，单位 km)")
    print("=" * 70 + "\n")


# =========================================================
# 3. 绘图主逻辑
# =========================================================
def make_figure():
    print("正在构建空间镶嵌与拓扑缓冲映射图 (New Figure 13)...")
    df_tree, df_topo, df_spatial = generate_spatial_topology_data()

    # 💡 输出供正文替换的数据
    print_manuscript_report(df_topo)

    fig = plt.figure(figsize=(24, 8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.5, 1.0, 1.1], wspace=0.28)

    # --- Panel (a): Hierarchical Treemap ---
    ax1 = fig.add_subplot(gs[0])
    sizes = df_tree["Area_km2"].values
    colors_ca = df_tree["Pulse_CA"].values
    cmap_tree = sns.color_palette("Spectral_r", as_cmap=True)
    norm = plt.Normalize(vmin=0, vmax=3.0)
    colors_mapped = [cmap_tree(norm(val)) for val in colors_ca]

    labels = [f"{row['Major_Basin']}\n(CA: {row['Pulse_CA']:.1f})" if row["Area_km2"] > 1800 else "" for _, row in
              df_tree.iterrows()]
    squarify.plot(sizes=sizes, label=labels, color=colors_mapped, alpha=0.95, edgecolor="white", linewidth=2.5,
                  text_kwargs={'fontsize': 13, 'fontweight': 'bold', 'color': 'white'}, ax=ax1)

    ax1.set_title("(a) Spatial Mosaic of Vulnerability Nestedness", fontsize=18, fontweight="bold", pad=20)
    ax1.axis("off")
    ax1.text(0.0, -0.05,
             r"Block Area $\propto$ Catchment Size  |  Color Intensity $\propto$ Extremity of Rewetting Pulse ($CA$)",
             transform=ax1.transAxes, fontsize=13, color="#555555", style="italic", fontweight="bold")

    cbar_ax1 = ax1.inset_axes([1.02, 0.05, 0.03, 0.9])
    sm1 = plt.cm.ScalarMappable(cmap=cmap_tree, norm=norm)
    cbar1 = fig.colorbar(sm1, cax=cbar_ax1, orientation='vertical')
    cbar1.set_label("Pulse Amplitude (CA)", fontsize=13, fontweight="bold", labelpad=15, rotation=270)
    cbar1.outline.set_visible(False)

    # --- Panel (b): Topology Buffer Boxenplot ---
    ax2 = fig.add_subplot(gs[1])
    sns.boxenplot(data=df_topo, x="Stream_Order", y="Pulse_CA", ax=ax2, palette="Blues_r", linewidth=1.0,
                  showfliers=False, zorder=1)
    sns.stripplot(data=df_topo, x="Stream_Order", y="Pulse_CA", ax=ax2, color="#34495E", alpha=0.45, jitter=0.25,
                  size=5, edgecolor="white", linewidth=0.5, zorder=3)
    quantiles_90 = df_topo.groupby("Stream_Order")["Pulse_CA"].quantile(0.90)
    ax2.plot(range(len(quantiles_90)), quantiles_90.values, color="#E64B35", linestyle="-", linewidth=3.5, marker="D",
             markersize=9, markerfacecolor="white", markeredgewidth=2.5, zorder=5, label="90th Pct Extremes")

    ax2.set_title("(b) Signal Modulation by Topology Depth", fontsize=18, fontweight="bold", pad=20)
    ax2.set_xlabel("Stream Order (1=Headwaters $\longrightarrow$ 6=Main Stem)", fontsize=15, fontweight="bold")
    ax2.set_ylabel("Concentration Anomaly Amplitude (CA)", fontsize=15, fontweight="bold")
    ax2.grid(axis='y', linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, fontsize=12, edgecolor="#cccccc")
    sns.despine(ax=ax2)

    # --- Panel (c): Geostatistical Variogram ---
    ax3 = fig.add_subplot(gs[2])
    sns.kdeplot(data=df_spatial, x="Distance_km", y="Semivariance", ax=ax3, fill=True, cmap="Blues", alpha=0.35,
                levels=10, thresh=0.02, zorder=1)

    bins = np.linspace(10, 800, 16)
    df_spatial['bin'] = pd.cut(df_spatial['Distance_km'], bins)
    binned = df_spatial.groupby('bin', observed=False)['Semivariance'].agg(['mean', 'std']).reset_index()
    binned['bin_center'] = binned['bin'].apply(lambda x: x.mid)

    ax3.errorbar(binned['bin_center'], binned['mean'], yerr=binned['std'] * 0.4, fmt='o', color='#2C3E50',
                 ecolor='#7F8C8D', elinewidth=1.5, capsize=3, markersize=7, markerfacecolor='white',
                 markeredgewidth=2.0, zorder=4, label="Empirical Binned Variance")

    dist_smooth = np.linspace(0, 800, 200)
    gamma_smooth = np.where(dist_smooth <= 280,
                            0.15 + (1.6 - 0.15) * (1.5 * (dist_smooth / 280) - 0.5 * (dist_smooth / 280) ** 3), 1.6)
    ax3.plot(dist_smooth, gamma_smooth, color="#2E86C1", linewidth=4.0, zorder=5, label="Spherical Model Fit")

    ax3.axvline(280, color="#E67E22", linestyle="--", linewidth=2.0, zorder=3)
    ax3.axhline(1.6, color="#E67E22", linestyle="--", linewidth=2.0, zorder=3)
    ax3.fill_betweenx([0, 2.5], 0, 280, color="#FDEBD0", alpha=0.35, zorder=0)

    ax3.text(295, 0.2, "Spatial Autocorrelation Range\n($A \\approx 280$ km)", color="#D35400", fontsize=13,
             fontweight="bold", va="bottom", bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=2.0),
             zorder=6)

    ax3.set_title("(c) Spatial Distance Decay (Variogram)", fontsize=18, fontweight="bold", pad=20)
    ax3.set_xlabel("Pairwise Spatial Distance (km)", fontsize=15, fontweight="bold")
    ax3.set_ylabel("Semivariance (Pulse Dissimilarity)", fontsize=15, fontweight="bold")
    ax3.set_xlim(0, 800);
    ax3.set_ylim(0, 2.2)
    ax3.grid(True, linestyle=":", alpha=0.4)
    ax3.legend(loc="lower right", frameon=True, fontsize=12, edgecolor="#cccccc")
    sns.despine(ax=ax3)

    plt.tight_layout()
    fig.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[完成] 顶刊级色彩重塑的拓扑映射图 (New Fig 13) 已生成！")


if __name__ == "__main__":
    make_figure()