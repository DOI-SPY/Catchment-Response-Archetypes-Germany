from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch


# =========================================================
# 0. PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FIG3_TIFF = FIG_DIR / "Fig3_definition_of_hydrological_compound_states.tiff"
FIG3_PDF = FIG_DIR / "Fig3_definition_of_hydrological_compound_states.pdf"

FIG4_TIFF = FIG_DIR / "Fig4_analytical_framework_for_response_metrics.tiff"
FIG4_PDF = FIG_DIR / "Fig4_analytical_framework_for_response_metrics.pdf"


# =========================================================
# 1. GLOBAL STYLE
# =========================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 600


# =========================================================
# 2. HELPERS
# =========================================================
def add_panel_label(ax, label: str):
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12, fontweight="bold"
    )


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.7)
    ax.grid(False)


def draw_box(ax, xy, width, height, text, facecolor="#f7f7f7", edgecolor="#4d4d4d",
             fontsize=10, lw=1.0, rounded=True):
    x, y = xy
    if rounded:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=lw, edgecolor=edgecolor, facecolor=facecolor
        )
    else:
        patch = Rectangle(
            (x, y), width, height,
            linewidth=lw, edgecolor=edgecolor, facecolor=facecolor
        )
    ax.add_patch(patch)
    ax.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", fontsize=fontsize
    )
    return patch


def draw_arrow(ax, start, end, color="#4d4d4d", lw=1.2, mutation_scale=12):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color
    )
    ax.add_patch(arrow)
    return arrow


# =========================================================
# 3. FIG. 3
# =========================================================
def make_fig3():
    # Color palette
    c_normal = "#cfcfcf"
    c_drought = "#c77b7b"
    c_rewet = "#5fa6a6"
    c_high = "#7da6d8"
    c_line = "#4f4f4f"
    c_q20 = "#b55d5d"
    c_q50 = "#888888"
    c_q80 = "#5b89c7"

    fig = plt.figure(figsize=(13.5, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.1, 1.15], wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # -------------------------------------------------
    # (a) Monthly discharge series and thresholds
    # -------------------------------------------------
    months = np.arange(1, 25)
    q = np.array([34, 39, 36, 31, 28, 19, 14, 12, 15, 18, 22, 29,
                  33, 37, 35, 30, 27, 17, 11, 10, 14, 24, 32, 40], dtype=float)

    q20 = np.array([22] * len(months), dtype=float)
    q50 = np.array([28] * len(months), dtype=float)
    q80 = np.array([34] * len(months), dtype=float)

    ax1.plot(months, q, color=c_line, linewidth=1.8, marker="o", markersize=3.2)
    ax1.axhline(q20[0], color=c_q20, linestyle="--", linewidth=1.2, label="Q20")
    ax1.axhline(q50[0], color=c_q50, linestyle="--", linewidth=1.1, label="Q50")
    ax1.axhline(q80[0], color=c_q80, linestyle="--", linewidth=1.2, label="Q80")

    # highlight months
    drought_idx = [7, 8, 9, 19, 20]
    rewet_idx = [10, 21]
    high_idx = [2, 3, 24]

    ax1.scatter(drought_idx, q[np.array(drought_idx) - 1], color=c_drought, s=28, zorder=3, label="drought_like")
    ax1.scatter(rewet_idx, q[np.array(rewet_idx) - 1], color=c_rewet, s=32, zorder=3, label="post_drought_rewetting")
    ax1.scatter(high_idx, q[np.array(high_idx) - 1], color=c_high, s=30, zorder=3, label="non_drought_highflow")

    ax1.set_xlim(1, 24)
    ax1.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax1.set_xlabel("Month index")
    ax1.set_ylabel("Monthly discharge, Q")
    ax1.set_title("Monthly discharge series and thresholds", pad=8)
    add_panel_label(ax1, "(a)")
    clean_axes(ax1)

    leg = ax1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=False, ncol=2,
                     handletextpad=0.6, columnspacing=1.0)
    for txt in leg.get_texts():
        txt.set_fontfamily("Times New Roman")

    # -------------------------------------------------
    # (b) State partition logic
    # -------------------------------------------------
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 4.4)
    ax2.axis("off")

    ax2.set_title("State partition logic", pad=8)
    add_panel_label(ax2, "(b)")

    draw_box(ax2, (0.5, 3.1), 2.0, 0.65, "drought_like", facecolor="#f2dede", edgecolor=c_drought)
    draw_box(ax2, (3.2, 3.1), 2.6, 0.65, "post_drought_rewetting", facecolor="#d9efef", edgecolor=c_rewet)
    draw_box(ax2, (6.3, 3.1), 2.7, 0.65, "non_drought_highflow", facecolor="#dce8f8", edgecolor=c_high)

    draw_arrow(ax2, (2.5, 3.43), (3.2, 3.43))
    draw_arrow(ax2, (5.8, 3.43), (6.3, 3.43))

    ax2.text(1.5, 2.45, r"$Q \leq Q20$", ha="center", va="center", fontsize=10)
    ax2.text(4.5, 2.45, r"1st or 2nd month after drought" "\n" r"and $Q > Q50$", ha="center", va="center", fontsize=9.4)
    ax2.text(7.65, 2.45, r"$Q \geq Q80$" "\n" r"and not in rewetting window", ha="center", va="center", fontsize=9.4)

    draw_box(ax2, (3.2, 0.9), 2.5, 0.62, "normal", facecolor="#efefef", edgecolor="#8f8f8f")
    ax2.text(4.45, 1.95, "All remaining months", ha="center", va="center", fontsize=9.6)
    draw_arrow(ax2, (4.45, 1.9), (4.45, 1.52), color="#777777")

    # -------------------------------------------------
    # (c) Temporal mapping to monthly water-quality analysis
    # -------------------------------------------------
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 6)
    ax3.axis("off")

    ax3.set_title("Temporal mapping to monthly analysis", pad=8)
    add_panel_label(ax3, "(c)")

    draw_box(ax3, (0.6, 4.75), 3.1, 0.65, "Hydrological states", facecolor="#f6f6f6")
    draw_box(ax3, (0.9, 3.85), 2.5, 0.48, "drought_like", facecolor="#f2dede", edgecolor=c_drought, fontsize=9.3)
    draw_box(ax3, (0.9, 3.2), 2.5, 0.48, "post_drought_rewetting", facecolor="#d9efef", edgecolor=c_rewet, fontsize=9.0)
    draw_box(ax3, (0.9, 2.55), 2.5, 0.48, "non_drought_highflow", facecolor="#dce8f8", edgecolor=c_high, fontsize=9.0)
    draw_box(ax3, (0.9, 1.9), 2.5, 0.48, "normal", facecolor="#efefef", edgecolor="#8f8f8f", fontsize=9.3)

    draw_box(ax3, (4.3, 4.75), 1.5, 0.65, "Match by\nOBJECTID + Month", facecolor="#fff7e6",
             edgecolor="#c08c3e", fontsize=9.2)

    draw_box(ax3, (6.4, 4.75), 2.8, 0.65, "Monthly water-quality outputs", facecolor="#f6f6f6")
    draw_box(ax3, (6.7, 3.95), 2.2, 0.45, "mean_C / mean_FNC", facecolor="#ffffff", fontsize=9.1)
    draw_box(ax3, (6.7, 3.25), 2.2, 0.45, "mean_Flux / mean_FNFlux", facecolor="#ffffff", fontsize=9.1)
    draw_box(ax3, (6.7, 2.55), 2.2, 0.45, "mean_Q", facecolor="#ffffff", fontsize=9.1)
    draw_box(ax3, (6.7, 1.85), 2.2, 0.45, "State-grouped c–Q analysis", facecolor="#ffffff", fontsize=9.0)

    draw_arrow(ax3, (3.7, 5.07), (4.3, 5.07))
    draw_arrow(ax3, (5.8, 5.07), (6.4, 5.07))

    ax3.text(5.05, 4.0, "Direct monthly alignment", ha="center", va="center", fontsize=9.5, color="#7a5a1c")

    fig.savefig(FIG3_TIFF, dpi=600, bbox_inches="tight")
    fig.savefig(FIG3_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 4. FIG. 4
# =========================================================
def make_fig4():
    c_obs = "#4f7dbd"
    c_fn = "#d98c3f"
    c_shift1 = "#6a9e7a"
    c_shift2 = "#be6b6b"
    c_gray = "#7f7f7f"

    fig = plt.figure(figsize=(13.5, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # -------------------------------------------------
    # (a) concentration anomaly
    # -------------------------------------------------
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.axis("off")
    ax1.set_title("Concentration anomaly", pad=8)
    add_panel_label(ax1, "(a)")

    draw_box(ax1, (0.7, 4.4), 2.4, 0.75, "Observed monthly\nconcentration\n(mean_C)",
             facecolor="#e4edf8", edgecolor=c_obs, fontsize=10)
    draw_box(ax1, (0.7, 1.35), 2.4, 0.75, "Flow-normalized\nconcentration\n(mean_FNC)",
             facecolor="#fbe9d7", edgecolor=c_fn, fontsize=10)

    draw_box(ax1, (4.0, 2.65), 2.2, 1.2,
             r"$CA = \ln \left(\dfrac{mean\_C}{mean\_FNC}\right)$",
             facecolor="#f7f7f7", edgecolor="#4d4d4d", fontsize=10.5)

    draw_arrow(ax1, (3.1, 4.75), (4.0, 3.55), color=c_obs)
    draw_arrow(ax1, (3.1, 1.72), (4.0, 2.95), color=c_fn)

    ax1.text(7.15, 4.1, r"$CA > 0$", fontsize=10, ha="left", va="center")
    ax1.text(7.15, 3.65, "Observed concentration", fontsize=9.2, ha="left", va="center")
    ax1.text(7.15, 3.25, "higher than normalized baseline", fontsize=9.2, ha="left", va="center")

    ax1.text(7.15, 2.15, r"$CA < 0$", fontsize=10, ha="left", va="center")
    ax1.text(7.15, 1.7, "Observed concentration", fontsize=9.2, ha="left", va="center")
    ax1.text(7.15, 1.3, "lower than normalized baseline", fontsize=9.2, ha="left", va="center")

    # -------------------------------------------------
    # (b) flux anomaly
    # -------------------------------------------------
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis("off")
    ax2.set_title("Flux anomaly", pad=8)
    add_panel_label(ax2, "(b)")

    draw_box(ax2, (0.7, 4.4), 2.4, 0.75, "Observed monthly\nflux\n(mean_Flux)",
             facecolor="#e4edf8", edgecolor=c_obs, fontsize=10)
    draw_box(ax2, (0.7, 1.35), 2.4, 0.75, "Flow-normalized\nflux\n(mean_FNFlux)",
             facecolor="#fbe9d7", edgecolor=c_fn, fontsize=10)

    draw_box(ax2, (4.0, 2.65), 2.2, 1.2,
             r"$FA = \ln \left(\dfrac{mean\_Flux}{mean\_FNFlux}\right)$",
             facecolor="#f7f7f7", edgecolor="#4d4d4d", fontsize=10.3)

    draw_arrow(ax2, (3.1, 4.75), (4.0, 3.55), color=c_obs)
    draw_arrow(ax2, (3.1, 1.72), (4.0, 2.95), color=c_fn)

    ax2.text(7.1, 4.0, r"$FA > 0$", fontsize=10, ha="left", va="center")
    ax2.text(7.1, 3.55, "State-associated export", fontsize=9.2, ha="left", va="center")
    ax2.text(7.1, 3.15, "amplified relative to baseline", fontsize=9.2, ha="left", va="center")

    ax2.text(7.1, 2.15, r"$FA < 0$", fontsize=10, ha="left", va="center")
    ax2.text(7.1, 1.7, "State-associated export", fontsize=9.2, ha="left", va="center")
    ax2.text(7.1, 1.3, "weaker than normalized baseline", fontsize=9.2, ha="left", va="center")

    # -------------------------------------------------
    # (c) c–Q regime shift
    # -------------------------------------------------
    x = np.linspace(0.8, 9.0, 50)
    y_normal = 1.3 + 0.22 * x
    y_rewet = 0.9 + 0.40 * x
    y_high = 1.0 + 0.12 * x

    ax3.plot(x, y_normal, color=c_gray, linewidth=1.8, label=r"normal: $\beta_{normal}$")
    ax3.plot(x, y_rewet, color=c_shift2, linewidth=2.0, label=r"post-drought rewetting: $\beta_{rew}$")
    ax3.plot(x, y_high, color=c_shift1, linewidth=2.0, label=r"high-flow: $\beta_{high}$")

    ax3.set_xlim(0.5, 9.3)
    ax3.set_ylim(1.0, 5.3)
    ax3.set_xlabel(r"$\log(Q)$")
    ax3.set_ylabel(r"$\log(C)$")
    ax3.set_title("c–Q regime shift", pad=8)
    add_panel_label(ax3, "(c)")
    clean_axes(ax3)

    ax3.text(5.2, 4.75, r"$\Delta \beta_{state} = \beta_{state} - \beta_{normal}$",
             fontsize=10, ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.25", fc="#f7f7f7", ec="#7f7f7f", lw=0.9))

    leg = ax3.legend(loc="lower right", frameon=False)
    for txt in leg.get_texts():
        txt.set_fontfamily("Times New Roman")

    fig.savefig(FIG4_TIFF, dpi=600, bbox_inches="tight")
    fig.savefig(FIG4_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# 5. MAIN
# =========================================================
def main():
    make_fig3()
    make_fig4()

    print("\n=== Method figures generated successfully ===")
    print(f"Fig. 3 TIFF: {FIG3_TIFF}")
    print(f"Fig. 3 PDF : {FIG3_PDF}")
    print(f"Fig. 4 TIFF: {FIG4_TIFF}")
    print(f"Fig. 4 PDF : {FIG4_PDF}")


if __name__ == "__main__":
    main()