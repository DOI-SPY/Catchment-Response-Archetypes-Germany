from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import gamma

# =========================================================
# 1. 动态路径解析
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG_PNG = FIG_DIR / "fig3_conceptual_framework.png"
OUT_FIG_PDF = FIG_DIR / "fig3_conceptual_framework.pdf"

# 顶刊全局排版设定
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2


# =========================================================
# 2. 核心绘图逻辑：重构物理逼真度与学术美感
# =========================================================
def make_conceptual_figure():
    print("正在重绘方法学理论概念图 (Conceptual Framework) - 高级质感版...")

    fig = plt.figure(figsize=(16, 12))
    # 2x2 网格，增加子图间的间距防止拥挤
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # ---------------------------------------------------------
    # Panel (a): Hydrological Memory (CMD) & State Sequence
    # ---------------------------------------------------------
    t = np.linspace(0, 150, 1000)

    # 构造逼真的流量曲线 (Q)
    Q = np.ones_like(t) * 12  # Baseflow
    Q += np.sin(t / 8) * 2  # Seasonal variation

    # Drought (Exponential recession)
    drought_start, drought_end = 400, 800
    recession = np.exp(-(t[drought_start:drought_end] - t[drought_start]) / 15)
    Q[drought_start:drought_end] = 8 * recession + 1.5

    # Rewetting Pulse (Gamma distribution-like peak)
    pulse_start, pulse_end = 800, 950
    x_pulse = np.linspace(0, 15, pulse_end - pulse_start)
    Q[pulse_start:pulse_end] = Q[pulse_start - 1] + gamma.pdf(x_pulse, a=2.5, scale=1.5) * 60

    # 恢复期
    Q[pulse_end:] = 10 + np.sin(t[pulse_end:] / 8) * 1.5

    # 构造水分亏缺 (CMD) 曲线
    CMD = np.zeros_like(t)
    CMD[drought_start:drought_end] = np.linspace(0, 20, drought_end - drought_start)
    CMD[pulse_start:pulse_start + 50] = np.linspace(20, 0, 50)  # 快速充水归零

    # 绘图
    ax1.plot(t, Q, color="#2c5d87", lw=2.5, label="Discharge ($Q$)")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(t, CMD, color="#b33939", lw=2.5, linestyle="--", label="Cumulative Moisture Deficit (CMD)")

    # 优雅的背景着色
    ax1.axvspan(t[0], t[drought_start], color='#e8ecef', alpha=0.5, lw=0)
    ax1.axvspan(t[drought_start], t[pulse_start], color='#f4e3e3', alpha=0.5, lw=0)
    ax1.axvspan(t[pulse_start], t[pulse_end - 80], color='#e0f2f1', alpha=0.6, lw=0)

    # 严格使用 transAxes 进行文字定位，彻底解决白边和漂移问题
    ax1.text(0.15, 0.90, "Normal State", transform=ax1.transAxes, ha='center', fontsize=13, fontweight='bold',
             color="#555555")
    ax1.text(0.55, 0.90, "Drought\n(Memory Accumulation)", transform=ax1.transAxes, ha='center', fontsize=13,
             fontweight='bold', color="#b33939")
    ax1.text(0.85, 0.90, "Rewetting\nPulse", transform=ax1.transAxes, ha='center', fontsize=13, fontweight='bold',
             color="#00695c")

    ax1.set_title("(a) Hydrological States & Memory Effect", fontsize=16, fontweight="bold", pad=15)
    ax1.set_xlim(0, 150)
    ax1.set_ylim(0, max(Q) * 1.2)
    ax1_twin.set_ylim(-2, max(CMD) * 1.5)  # 给 CMD 留出顶部空间

    ax1.set_ylabel("Discharge ($Q$)", color="#2c5d87", fontsize=14, fontweight="bold")
    ax1_twin.set_ylabel("CMD (Severity)", color="#b33939", fontsize=14, fontweight="bold")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1_twin.set_yticks([])

    # ---------------------------------------------------------
    # Panel (b): Concentration Anomaly (CA) Calculation
    # ---------------------------------------------------------
    t_b = np.linspace(0, 100, 800)

    # 逼真的 FNC 基线 (平滑的季节波动)
    FNC = 10 + np.sin(t_b / 10) * 1.5

    # 真实的浓度 C (叠加小范围噪声和巨大的重置脉冲)
    C = FNC + np.random.normal(0, 0.2, len(t_b))
    pulse_idx_b = 600
    C[pulse_idx_b:pulse_idx_b + 100] += gamma.pdf(np.linspace(0, 15, 100), a=2.0, scale=1.5) * 35

    # 绘制基线与实测值
    ax2.plot(t_b, C, color="#27ae60", lw=2.0, alpha=0.9, label="Observed Concentration ($C$)")
    ax2.plot(t_b, FNC, color="#555555", lw=2.5, linestyle="--", label="Flow-Normalized Baseline ($FNC$)")

    # 精美的高光填充展示 CA 提取区域
    ax2.fill_between(t_b, FNC, C, where=(C > FNC), color="#2ecc71", alpha=0.3, hatch="///", edgecolor="none")

    # 指示箭头
    ax2.annotate(r'$CA = \ln(C / FNC)$', xy=(t_b[pulse_idx_b + 25], C[pulse_idx_b + 25]),
                 xytext=(0.4, 0.7), textcoords='axes fraction',
                 arrowprops=dict(facecolor='#333333', shrink=0.05, width=1.5, headwidth=7),
                 fontsize=15, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=1))

    ax2.set_title("(b) Concentration Anomaly ($CA$) Extraction", fontsize=16, fontweight="bold", pad=15)
    ax2.legend(loc="upper left", frameon=True, fontsize=12, edgecolor="#cccccc")
    ax2.set_xlim(0, 100)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # ---------------------------------------------------------
    # Panel (c): State-dependent c-Q Slopes ($\beta$)
    # ---------------------------------------------------------
    np.random.seed(42)
    log_q = np.linspace(-2, 2, 200)

    # 生成背景的模拟散点云以增加物理真实感
    ax3.scatter(log_q, 0.8 * log_q + 1 + np.random.normal(0, 0.3, 200), color="#d35400", alpha=0.15, s=20)
    ax3.scatter(log_q, 0.0 * log_q + 1 + np.random.normal(0, 0.3, 200), color="#7f8c8d", alpha=0.15, s=20)
    ax3.scatter(log_q, -0.8 * log_q + 1 + np.random.normal(0, 0.3, 200), color="#2980b9", alpha=0.15, s=20)

    # 三种典型回归斜率主线
    ax3.plot(log_q, 0.8 * log_q + 1, color="#d35400", lw=3.5, label=r"Flushing ($\beta > 0$)")
    ax3.plot(log_q, 0.0 * log_q + 1, color="#7f8c8d", lw=3.5, linestyle="--", label=r"Chemostatic ($\beta \approx 0$)")
    ax3.plot(log_q, -0.8 * log_q + 1, color="#2980b9", lw=3.5, label=r"Dilution ($\beta < 0$)")

    ax3.set_title("(c) State-Dependent c-Q Slopes ($\\beta$)", fontsize=16, fontweight="bold", pad=15)
    ax3.set_xlabel("$\ln(Q)$", fontsize=15)
    ax3.set_ylabel("$\ln(C)$", fontsize=15)
    ax3.legend(loc="upper left", frameon=True, fontsize=12, edgecolor="#cccccc")
    ax3.grid(True, linestyle="--", alpha=0.4, color="#cccccc")

    # 精美的公式文本框
    ax3.text(0.95, 0.05, r"$\ln(C) = \alpha + \beta \ln(Q)$", transform=ax3.transAxes,
             fontsize=15, ha='right', va='bottom',
             bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9fa", ec="#cccccc", lw=1))

    # ---------------------------------------------------------
    # Panel (d): Chemostatic Index & Regime Space
    # ---------------------------------------------------------
    ax4.plot([0, 2], [0, 2], color="#333333", linestyle="--", lw=2.5, label="$CV_c / CV_q = 1$ (Threshold)")

    # 优雅的分区底色
    ax4.fill_between([0, 2], [0, 2], 2, color="#fff0f0", alpha=0.7)  # Chemodynamic
    ax4.fill_between([0, 2], 0, [0, 2], color="#f0f9f4", alpha=0.7)  # Chemostatic

    ax4.text(0.6, 1.4, "Chemodynamic\n(Source-limited)\n$CV_c > CV_q$", ha='center', va='center', fontsize=14,
             fontweight='bold', color="#c0392b")
    ax4.text(1.4, 0.6, "Chemostatic\n(Transport-limited)\n$CV_c < CV_q$", ha='center', va='center', fontsize=14,
             fontweight='bold', color="#27ae60")

    # 具有学术张力的体制跃迁向量 (Regime Shift Vector)
    arrow = patches.FancyArrowPatch((1.3, 0.8), (0.9, 1.5), mutation_scale=20,
                                    color="#e74c3c", lw=3, arrowstyle="-|>", zorder=4)
    ax4.add_patch(arrow)
    ax4.text(1.15, 1.25, "Regime\nShift", color="#e74c3c", fontsize=13, fontweight="bold", ha="left", rotation=-60)

    # 起点与终点标识
    ax4.scatter([1.3], [0.8], color="#34495e", s=100, zorder=5, edgecolor="white", lw=2)
    ax4.scatter([0.9], [1.5], color="#e74c3c", s=100, zorder=5, edgecolor="white", lw=2)
    ax4.text(1.35, 0.75, "Normal State", fontsize=11, color="#34495e", fontweight="bold")
    ax4.text(0.85, 1.55, "Rewetting State", fontsize=11, color="#e74c3c", fontweight="bold", ha="right")

    ax4.set_title("(d) Chemostatic Index ($CV_c/CV_q$) & Regime Shifts", fontsize=16, fontweight="bold", pad=15)
    ax4.set_xlabel("$CV_q$ (Variance in Discharge)", fontsize=14)
    ax4.set_ylabel("$CV_c$ (Variance in Concentration)", fontsize=14)
    ax4.set_xlim(0, 2)
    ax4.set_ylim(0, 2)
    ax4.legend(loc="upper left", frameon=False, fontsize=12)

    # ---------------------------------------------------------
    # 整体排版精调
    # ---------------------------------------------------------
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25, wspace=0.2)

    # 去除所有多余的边框线条以提升现代感 (Tufte's Data-Ink Ratio)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax in [ax1, ax2]:
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

    fig.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[完成] 方法学概念图已重新生成: \n- {OUT_FIG_PNG}\n- {OUT_FIG_PDF}")


if __name__ == "__main__":
    make_conceptual_figure()