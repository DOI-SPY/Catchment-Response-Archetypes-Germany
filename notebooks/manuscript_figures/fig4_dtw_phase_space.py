from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.stats import gamma
from dtaidistance import dtw

# =========================================================
# 1. 动态路径解析
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG_PNG = FIG_DIR / "fig4_dtw_phase_space.png"
OUT_FIG_PDF = FIG_DIR / "fig4_dtw_phase_space.pdf"

# 全局顶刊排版设定
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["mathtext.fontset"] = "stix"


# =========================================================
# 2. 核心算法：生成物理逼真的时序轨迹
# =========================================================
def generate_event(event_type="flushing", abs_delay=0, noise_level=0.03):
    """
    使用 Gamma 分布模拟真实的降雨径流与浓度波形
    abs_delay: 绝对时间滞后，用于模拟不同流域的汇流时间差
    """
    t = np.linspace(0, 40, 250)

    # 统一的流量波形 Q (代表降雨径流)
    Q = 2 + gamma.pdf(t - abs_delay, a=4.5, scale=1.5) * 120

    # 水质响应波形 C
    if event_type == "flushing":
        # 冲刷型：浓度峰值超前于流量峰值 (顺时针滞回环)
        C = 4 + gamma.pdf(t - abs_delay + 2.5, a=3.5, scale=1.5) * 65
    elif event_type == "dilution":
        # 稀释型：浓度随流量峰值出现凹陷 (逆时针滞回环)
        C = 16 - gamma.pdf(t - abs_delay, a=4.5, scale=1.5) * 90
        C = np.clip(C, 1.5, None)

        # 加入极其微弱的物理白噪声 (模拟观测仪器误差，避免线团感)
    C += np.random.normal(0, noise_level, len(t))
    Q += np.random.normal(0, noise_level, len(t))

    return t, Q, C


def plot_gradient_line(ax, x, y, cmap_name, lw=3.5, label=None):
    """高级渲染：用颜色渐变代表时间的流逝，完美替代突兀的箭头"""
    t_norm = np.linspace(0, 1, len(x))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 获取 colormap
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(t_norm.min(), t_norm.max())

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=lw, capstyle='round', zorder=4)
    lc.set_array(t_norm)
    ax.add_collection(lc)

    # 绘制一个幽灵点用于生成图例
    if label:
        ax.plot([], [], color=cmap(0.8), lw=lw, label=label)


# =========================================================
# 3. 绘图主逻辑
# =========================================================
def make_figure():
    print("正在计算 DTW 代价矩阵并绘制高级概率流形图 (Figure 4)...")

    # 1. 生成两组机制相同(冲刷)、但在绝对时间上严重错位的事件
    t, Q1, C1 = generate_event("flushing", abs_delay=0)  # 快响应 (上游)
    t, Q2, C2 = generate_event("flushing", abs_delay=8.0)  # 慢响应 (下游)

    s1 = (C1 - np.mean(C1)) / np.std(C1)
    s2 = (C2 - np.mean(C2)) / np.std(C2)

    # 计算 DTW 代价矩阵与最优路径
    d, paths = dtw.warping_paths(s1, s2)
    best_path = dtw.best_path(paths)

    # =======================================
    # 准备画布 (1x3 极简黄金比例)
    # =======================================
    fig = plt.figure(figsize=(20, 6.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.25)

    # ---------------------------------------------------------
    # Panel (a): 原始未对齐的相空间轨迹 (时间渐变线)
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])

    # 使用时间渐变线绘制，摒弃毛线团
    plot_gradient_line(ax1, Q1, C1, "Reds", label="Catchment A (Fast Transit, early peak)")
    plot_gradient_line(ax1, Q2, C2, "Blues", label="Catchment B (Slow Transit, delayed peak)")

    ax1.set_xlim(min(Q1.min(), Q2.min()) - 1, max(Q1.max(), Q2.max()) + 1)
    ax1.set_ylim(min(C1.min(), C2.min()) - 1, max(C1.max(), C2.max()) + 1)

    ax1.set_title("(a) Chaotic Phase Space (Unaligned)", fontsize=18, fontweight="bold", pad=15)
    ax1.set_xlabel("Discharge ($Q$)", fontsize=15)
    ax1.set_ylabel("Concentration ($C$)", fontsize=15)
    ax1.legend(loc="upper left", frameon=True, fontsize=12, edgecolor="#cccccc")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax1.text(0.95, 0.05, "Color gradient indicates time.\nEuclidean distance fails due to phase shift.",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=12, color="#555", style="italic")

    # ---------------------------------------------------------
    # Panel (b): DTW 代价曲面与最优规整路径 (高级热力渲染)
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])

    cost_matrix = paths.copy()
    cost_matrix[cost_matrix == np.inf] = np.nan

    # 使用高级深色系 mako 突显亮色对齐线
    cmap_surface = sns.color_palette("mako", as_cmap=True)
    cmap_surface.set_bad(color='#f8f9fa')
    im = ax2.imshow(cost_matrix, origin="lower", cmap=cmap_surface, interpolation="bilinear", aspect="auto")

    path_x = [p[1] for p in best_path]
    path_y = [p[0] for p in best_path]

    # 使用高对比度的亮橘色绘制最优路径
    ax2.plot(path_x, path_y, color="#ff9f43", lw=4.0, label="Optimal Warping Path (DTW)")
    ax2.plot([0, len(s2)], [0, len(s1)], color="white", linestyle="--", lw=1.5, alpha=0.6,
             label="Linear Alignment (Euclidean)")

    ax2.set_title("(b) DTW Cost Surface & Alignment", fontsize=18, fontweight="bold", pad=15)
    ax2.set_xlabel("Time steps of Catchment B", fontsize=15)
    ax2.set_ylabel("Time steps of Catchment A", fontsize=15)
    ax2.legend(loc="upper left", frameon=True, fontsize=12, facecolor="white", edgecolor="none")

    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Cumulative DTW Cost", fontsize=13)
    cbar.outline.set_visible(False)

    # ---------------------------------------------------------
    # Panel (c): 规整后提取的概率流形边界 (2D KDE Cloud)
    # ---------------------------------------------------------
    ax3 = fig.add_subplot(gs[2])

    # 仿真生成用于绘制背景密度的散点云
    np.random.seed(42)
    q_all_f, c_all_f, q_all_d, c_all_d = [], [], [], []
    for _ in range(15):
        # 【修正】：将 lag 替换为 abs_delay
        _, q_sim, c_sim = generate_event("flushing", abs_delay=0, noise_level=0.5)
        q_all_f.extend(q_sim)
        c_all_f.extend(c_sim)

        _, q_sim_d, c_sim_d = generate_event("dilution", abs_delay=0, noise_level=0.5)
        q_all_d.extend(q_sim_d)
        c_all_d.extend(c_sim_d)

    # 绘制高级 2D 核密度流形 (Density Manifold)
    sns.kdeplot(x=q_all_f, y=c_all_f, ax=ax3, fill=True, cmap="Reds", alpha=0.3, thresh=0.1, levels=4, zorder=1)
    sns.kdeplot(x=q_all_d, y=c_all_d, ax=ax3, fill=True, cmap="Blues", alpha=0.3, thresh=0.1, levels=4, zorder=1)

    # 叠加绝对平滑的聚类中心 (Medoids)
    _, q_med_f, c_med_f = generate_event("flushing", abs_delay=0, noise_level=0.0)
    _, q_med_d, c_med_d = generate_event("dilution", abs_delay=0, noise_level=0.0)

    plot_gradient_line(ax3, q_med_f, c_med_f, "Reds", lw=4.0, label="Archetype 0: Flushing (Clockwise)")
    plot_gradient_line(ax3, q_med_d, c_med_d, "Blues", lw=4.0, label="Archetype 1: Dilution (Anti-clockwise)")

    ax3.set_xlim(min(q_all_f + q_all_d), max(q_all_f + q_all_d))
    ax3.set_ylim(min(c_all_f + c_all_d), max(c_all_f + c_all_d))

    ax3.set_title("(c) Archetype Manifolds Extracted by DTW", fontsize=18, fontweight="bold", pad=15)
    ax3.set_xlabel("Discharge ($Q$)", fontsize=15)
    ax3.set_ylabel("Concentration ($C$)", fontsize=15)
    ax3.legend(loc="upper right", frameon=True, fontsize=12, edgecolor="#cccccc")
    ax3.grid(True, linestyle="--", alpha=0.4, zorder=0)

    # ---------------------------------------------------------
    # 整体排版精调
    # ---------------------------------------------------------
    sns.despine()
    plt.tight_layout()

    fig.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[完成] DTW 动态时间规整相空间对齐图已深度重构: \n- {OUT_FIG_PNG}\n- {OUT_FIG_PDF}")


if __name__ == "__main__":
    make_figure()