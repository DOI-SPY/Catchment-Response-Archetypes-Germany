from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import gaussian_kde

# =========================================================
# 1. 动态路径解析
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG_PNG = FIG_DIR / "fig13_bayesian_uncertainty.png"
OUT_FIG_PDF = FIG_DIR / "fig13_bayesian_uncertainty.pdf"

# 顶刊全局排版设定
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["mathtext.fontset"] = "stix"


# =========================================================
# 2. 高保真贝叶斯后验 MCMC 仿真数据
# =========================================================
def generate_bayesian_data():
    np.random.seed(42)

    # 1. 雨云图数据
    post_normal = np.random.normal(loc=0.1, scale=0.25, size=2000)
    post_drought = np.random.normal(loc=1.8, scale=0.7, size=2000)
    post_drought = np.clip(post_drought, -0.5, 4.5)

    # 2. 蜂群图数据
    catchment_df = pd.DataFrame({
        "Condition": ["Mild"] * 60 + ["Moderate"] * 60 + ["Extreme"] * 60,
        "Pulse_CA": np.concatenate([
            np.random.normal(0.5, 0.4, 60),
            np.random.normal(1.2, 0.6, 60),
            np.random.normal(2.5, 0.9, 60)
        ])
    })

    # 3. 森林图数据
    forest_df = pd.DataFrame({
        "Covariate": ["Agri. Fraction (+)", "Drought Memory (+)", "Baseflow Index (+)",
                      "Forest Cover (-)", "Elevation (-)"],
        "Mean": [0.85, 0.62, 0.15, -0.45, -0.72],
        "Lower_95": [0.65, 0.40, -0.15, -0.65, -0.95],  # Baseflow 不显著
        "Upper_95": [1.05, 0.84, 0.45, -0.25, -0.49]
    })

    return post_normal, post_drought, catchment_df, forest_df


# =========================================================
# 3. 底层高精度引擎：纯手工构建顶级雨云图 (Raincloud)
# =========================================================
def plot_native_raincloud(ax, data, y_pos, color, label):
    # (1) 绘制云 (KDE Cloud) - 优化曲线平滑度与透明度
    kde = gaussian_kde(data, bw_method=0.25)
    x_val = np.linspace(min(data) - 0.5, max(data) + 0.5, 300)
    y_dens = kde(x_val)
    y_dens = y_dens / y_dens.max() * 0.45

    ax.fill_between(x_val, y_pos, y_pos + y_dens, color=color, alpha=0.6, edgecolor=color, lw=1.5)

    # (2) 绘制伞 (Boxplot) - 极致压缩厚度，提升纤细美感
    bp = ax.boxplot(data, positions=[y_pos - 0.08], vert=False, widths=0.08, showfliers=False, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_edgecolor("white")
        patch.set_linewidth(1.0)
        patch.set_alpha(0.9)
    for median in bp['medians']:
        median.set(color="white", linewidth=2.5)
    for cap in bp['caps']:
        cap.set(color=color, linewidth=1.5)
    for whisker in bp['whiskers']:
        whisker.set(color=color, linewidth=1.5)

    # (3) 绘制雨滴 (Jittered Stripplot) - 缩小尺寸，增加白边
    jitter = np.random.uniform(-0.12, 0.02, size=len(data))
    ax.scatter(data, y_pos - 0.22 + jitter, color=color, s=15, alpha=0.35, edgecolors='white', linewidths=0.3, zorder=0)


# =========================================================
# 4. 绘图主逻辑
# =========================================================
def make_figure():
    print("正在重构贝叶斯后验概率与高阶不确定性矩阵 (Figure 13)...")

    post_normal, post_drought, catch_df, forest_df = generate_bayesian_data()

    # 适当加宽画布，为纵坐标留足空间
    fig = plt.figure(figsize=(25, 8))
    # 【核心修复】：将 wspace 提高到 0.38，彻底杜绝 (c) 的文字切入 (b) 的区域
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.3, 1.0, 1.2], wspace=0.38)

    # ---------------------------------------------------------
    # Panel (a): 贝叶斯后验分布雨云图 (Raincloud Plot)
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])

    # 【美学修复】：采用顶刊高阶色（深海蓝 vs 陶土橘）
    plot_native_raincloud(ax1, post_drought, y_pos=2.0, color="#D35400", label="Extreme Rewetting Pulse")
    plot_native_raincloud(ax1, post_normal, y_pos=1.0, color="#2874A6", label="Normal Baseline State")

    ax1.axvline(0, color="#7f8c8d", linestyle="--", linewidth=1.5, zorder=0)

    ax1.set_yticks([1.0, 2.0])
    ax1.set_yticklabels(["Normal Baseline\n(Posterior)", "Rewetting Pulse\n(Posterior)"], fontsize=15,
                        fontweight="bold")
    ax1.set_ylim(0.4, 2.8)
    ax1.set_xlim(-1.5, 5.0)

    ax1.set_title("(a) Bayesian Posterior Distributions", fontsize=18, fontweight="bold", pad=20)
    ax1.set_xlabel(r"Concentration Anomaly Magnitude ($\Delta CA$)", fontsize=15)

    # 【核心修复】：为文字加上半透明白色底衬，移至顶部安全区
    ax1.text(0.95, 0.95, "Higher variance in extreme pulses\nindicates amplified epistemic uncertainty.",
             transform=ax1.transAxes, ha="right", va="top", fontsize=13, color="#333333", style="italic",
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=3.0))

    # ---------------------------------------------------------
    # Panel (b): 微观粒度响应蜂群图 (Swarm Plot)
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])

    # 采用更加明快通透的渐变色系
    palette = {"Mild": "#5DADE2", "Moderate": "#F5B041", "Extreme": "#E74C3C"}

    sns.swarmplot(x="Condition", y="Pulse_CA", data=catch_df, palette=palette,
                  size=6.5, alpha=0.9, edgecolor="white", linewidth=0.6, ax=ax2)

    # 【核心修复】：摒弃丑陋的 Boxplot，手写渲染极简冷峻的中位数十字线
    for i, cond in enumerate(["Mild", "Moderate", "Extreme"]):
        median_val = catch_df[catch_df["Condition"] == cond]["Pulse_CA"].median()
        ax2.hlines(median_val, i - 0.25, i + 0.25, color="#2C3E50", linewidth=3.5, zorder=5, capstyle='round')

    ax2.axhline(0, color="#7f8c8d", linestyle="--", linewidth=1.5, zorder=0)

    ax2.set_title("(b) Micro-granularity of Catchment Responses", fontsize=18, fontweight="bold", pad=20)
    ax2.set_xlabel("Drought Memory Severity", fontsize=15)
    ax2.set_ylabel(r"Catchment Anomaly ($CA$)", fontsize=15)
    ax2.tick_params(axis='x', labelsize=14)

    # ---------------------------------------------------------
    # Panel (c): 贝叶斯协变量效应森林图 (Forest Plot)
    # ---------------------------------------------------------
    ax3 = fig.add_subplot(gs[2])

    forest_df = forest_df.sort_values("Mean", ascending=True).reset_index(drop=True)
    y_ticks = np.arange(len(forest_df))

    # 基线优化
    ax3.axvline(0, color="#7f8c8d", linestyle="--", linewidth=2.0, zorder=0)
    ax3.fill_betweenx([-1, 5], -0.1, 0.1, color="#7f8c8d", alpha=0.1, zorder=0)

    for i, row in forest_df.iterrows():
        is_significant = (row["Lower_95"] > 0) or (row["Upper_95"] < 0)

        # 【美学重构】：正效应为红，负效应为蓝，不显著为灰。大空心圆改为实心锐利菱形(Diamond)
        if is_significant:
            color = "#C0392B" if row["Mean"] > 0 else "#2980B9"
            lw, alpha = 3.5, 1.0
        else:
            color = "#BDC3C7"
            lw, alpha = 2.5, 0.9

        ax3.hlines(y=i, xmin=row["Lower_95"], xmax=row["Upper_95"], color=color, linewidth=lw, alpha=alpha, zorder=3)
        ax3.scatter(row["Mean"], i, color=color, edgecolors="white", s=130, marker="D", linewidths=1.5, zorder=4,
                    alpha=alpha)

        if not is_significant:
            ax3.text(row["Upper_95"] + 0.05, i, "n.s.", color="#95a5a6", fontsize=12, va="center", style="italic")

    ax3.set_yticks(y_ticks)
    ax3.set_yticklabels(forest_df["Covariate"], fontsize=15, fontweight="bold")
    ax3.set_ylim(-0.5, len(forest_df) - 0.5)
    ax3.set_xlim(-1.2, 1.2)

    ax3.set_title("(c) Covariate Effect Sizes (95% CI Forest Plot)", fontsize=18, fontweight="bold", pad=20)
    ax3.set_xlabel("Bayesian Posterior Effect Size", fontsize=15)

    # 【排版修复】：将文字放在绝对安全的右下方空白区域
    ax3.text(0.95, 0.05,
             "Dashed line indicates Zero-effect null hypothesis.\nParameters crossing zero are non-significant (n.s.).",
             transform=ax3.transAxes, ha="right", va="bottom", fontsize=12, color="#555", style="italic",
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=3.0))

    # ---------------------------------------------------------
    # 整体排版精调
    # ---------------------------------------------------------
    sns.despine(ax=ax1, left=True)
    sns.despine(ax=ax2)
    sns.despine(ax=ax3, left=True)

    # 运用 tight_layout 融合 gridspec
    plt.tight_layout()

    fig.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[完成] 贝叶斯不确定性量化与森林图谱已重构为顶刊形态: \n- {OUT_FIG_PNG}\n- {OUT_FIG_PDF}")


if __name__ == "__main__":
    make_figure()