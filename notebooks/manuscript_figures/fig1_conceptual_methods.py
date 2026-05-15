from __future__ import annotations
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gamma

# =========================================================
# 1. 动态路径解析
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG_PNG = FIG_DIR / "fig1_conceptual_methods.png"
OUT_FIG_PDF = FIG_DIR / "fig1_conceptual_methods.pdf"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 1.5


# =========================================================
# 2. 绘图主逻辑
# =========================================================
def make_conceptual_figure():
    print("正在生成 New Figure 1 (纯方法学概念图)...")

    fig = plt.figure(figsize=(22, 8))
    gs = GridSpec(1, 2, figure=fig, wspace=0.25)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # --- Panel (a): Hydrological States ---
    t = np.linspace(0, 150, 1000)
    Q = np.ones_like(t) * 12 + np.sin(t / 8) * 2
    drought_start, drought_end = 400, 800
    recession = np.exp(-(t[drought_start:drought_end] - t[drought_start]) / 15)
    Q[drought_start:drought_end] = 8 * recession + 1.5

    pulse_start, pulse_end = 800, 950
    x_pulse = np.linspace(0, 15, pulse_end - pulse_start)
    Q[pulse_start:pulse_end] = Q[pulse_start - 1] + gamma.pdf(x_pulse, a=2.5, scale=1.5) * 60
    Q[pulse_end:] = 10 + np.sin(t[pulse_end:] / 8) * 1.5

    CMD = np.zeros_like(t)
    CMD[drought_start:drought_end] = np.linspace(0, 20, drought_end - drought_start)
    CMD[pulse_start:pulse_start + 50] = np.linspace(20, 0, 50)

    ax1.plot(t, Q, color="#2c5d87", lw=3.0, label="Discharge ($Q$)")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(t, CMD, color="#b33939", lw=3.0, linestyle="--", label="CMD")

    ax1.axvspan(t[0], t[drought_start], color='#e8ecef', alpha=0.5, lw=0)
    ax1.axvspan(t[drought_start], t[pulse_start], color='#f4e3e3', alpha=0.5, lw=0)
    ax1.axvspan(t[pulse_start], t[pulse_end - 80], color='#e0f2f1', alpha=0.6, lw=0)

    ax1.set_title("(a) Hydrological States & Memory Effect", fontsize=18, fontweight="bold", pad=15)
    ax1.set_xlim(0, 150);
    ax1.set_ylim(0, max(Q) * 1.2);
    ax1_twin.set_ylim(-2, max(CMD) * 1.5)
    ax1.set_ylabel("Discharge ($Q$)", color="#2c5d87", fontsize=16, fontweight="bold")
    ax1_twin.set_ylabel("CMD (Severity)", color="#b33939", fontsize=16, fontweight="bold")
    ax1.set_xticks([]);
    ax1.set_yticks([]);
    ax1_twin.set_yticks([])

    # --- Panel (b): CA Extraction ---
    t_b = np.linspace(0, 100, 800)
    FNC = 10 + np.sin(t_b / 10) * 1.5
    C = FNC + np.random.normal(0, 0.2, len(t_b))
    pulse_idx_b = 600
    C[pulse_idx_b:pulse_idx_b + 100] += gamma.pdf(np.linspace(0, 15, 100), a=2.0, scale=1.5) * 35

    ax2.plot(t_b, C, color="#27ae60", lw=2.0, alpha=0.9, label="Observed Conc.")
    ax2.plot(t_b, FNC, color="#555555", lw=3.0, linestyle="--", label="FNC Baseline")
    ax2.fill_between(t_b, FNC, C, where=(C > FNC), color="#2ecc71", alpha=0.3, hatch="///", edgecolor="none")

    ax2.annotate(r'$CA = \ln(C / FNC)$', xy=(t_b[pulse_idx_b + 25], C[pulse_idx_b + 25]),
                 xytext=(0.4, 0.7), textcoords='axes fraction',
                 arrowprops=dict(facecolor='#333333', shrink=0.05, width=2.0, headwidth=8),
                 fontsize=17, fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", lw=1))

    ax2.set_title("(b) Concentration Anomaly ($CA$) Extraction", fontsize=18, fontweight="bold", pad=15)
    ax2.legend(loc="upper left", frameon=True, fontsize=14, edgecolor="#cccccc")
    ax2.set_xlim(0, 100);
    ax2.set_xticks([]);
    ax2.set_yticks([])

    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False);
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False);
        ax.spines['left'].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"[完成] New Figure 1 (概念图) 已保存。")


if __name__ == "__main__":
    make_conceptual_figure()