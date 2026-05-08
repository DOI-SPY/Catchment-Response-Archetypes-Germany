from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
import cmasher as cmr


# =========================================================
# 0. PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = PROJECT_ROOT / "data_final"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = FINAL_DIR / "analysis_panel_station_month_states.csv"

FIG3_TIFF = FIG_DIR / "Fig3_definition_of_monthly_hydrological_compound_states_v2.tiff"
FIG3_PDF = FIG_DIR / "Fig3_definition_of_monthly_hydrological_compound_states_v2.pdf"
FIG3_PNG = FIG_DIR / "Fig3_definition_of_monthly_hydrological_compound_states_v2.png"

FIG4_TIFF = FIG_DIR / "Fig4_analytical_framework_for_response_metrics_v2.tiff"
FIG4_PDF = FIG_DIR / "Fig4_analytical_framework_for_response_metrics_v2.pdf"
FIG4_PNG = FIG_DIR / "Fig4_analytical_framework_for_response_metrics_v2.png"

OUT_REPRESENTATIVE_REPORT = REPORT_DIR / "fig3_representative_station_report.txt"


# =========================================================
# 1. GLOBAL STYLE
# =========================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 600

plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 8.8


STATE_COLORS = {
    "drought_like": "#b65f5f",
    "post_drought_rewetting": "#4f9a9a",
    "non_drought_highflow": "#5b83bd",
    "normal": "#d2d2d2",
}

STATE_LABELS = {
    "drought_like": "drought-like",
    "post_drought_rewetting": "post-drought rewetting",
    "non_drought_highflow": "non-drought high flow",
    "normal": "normal",
}


# =========================================================
# 2. HELPERS
# =========================================================
def read_csv_fallback(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    raise RuntimeError(f"Could not read {path}")


def first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def add_panel_label(ax, label: str):
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold"
    )


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.7)
    ax.grid(False)


def draw_box(
    ax,
    xy,
    width,
    height,
    text,
    facecolor="#f7f7f7",
    edgecolor="#4d4d4d",
    fontsize=9.5,
    lw=1.0,
    rounded=True,
):
    x, y = xy
    if rounded:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=lw,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
    else:
        patch = Rectangle(
            (x, y),
            width,
            height,
            linewidth=lw,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def draw_arrow(ax, start, end, color="#555555", lw=1.2, mutation_scale=12):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
    )
    ax.add_patch(arr)
    return arr


# =========================================================
# 3. PREPARE REAL MONTHLY STATE SERIES FOR FIG. 3(a)
# =========================================================
def prepare_monthly_hydro_table() -> pd.DataFrame:
    if not STATE_FILE.exists():
        raise FileNotFoundError(f"Missing state file: {STATE_FILE}")

    df = read_csv_fallback(STATE_FILE)

    state_col = first_existing(df, ["hydro_state", "state", "monthly_hydro_state"])
    q_col = first_existing(df, ["mean_Q", "Q_mean", "q_mean"])
    obj_col = first_existing(df, ["OBJECTID", "objectid"])
    year_col = first_existing(df, ["Year", "year"])
    month_col = first_existing(df, ["Month", "month"])

    if state_col is None:
        raise KeyError("Could not find hydro_state column in analysis_panel_station_month_states.csv")
    if q_col is None:
        raise KeyError("Could not find mean_Q column in analysis_panel_station_month_states.csv")
    if obj_col is None or year_col is None:
        raise KeyError("Could not find OBJECTID or Year column in state table")

    # If Month is missing but Date exists, derive Month.
    if month_col is None:
        date_col = first_existing(df, ["Date", "date"])
        if date_col is None:
            raise KeyError("Could not find Month or Date column in state table")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["Year"] = df[date_col].dt.year
        df["Month"] = df[date_col].dt.month
        year_col = "Year"
        month_col = "Month"

    keep = [obj_col, year_col, month_col, q_col, state_col]
    if "solute" in df.columns:
        keep.append("solute")

    out = df[keep].copy()
    out = out.rename(
        columns={
            obj_col: "OBJECTID",
            year_col: "Year",
            month_col: "Month",
            q_col: "mean_Q",
            state_col: "hydro_state",
        }
    )

    out["mean_Q"] = pd.to_numeric(out["mean_Q"], errors="coerce")
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    out["Month"] = pd.to_numeric(out["Month"], errors="coerce")

    out = out.dropna(subset=["OBJECTID", "Year", "Month", "mean_Q", "hydro_state"]).copy()
    out["Year"] = out["Year"].astype(int)
    out["Month"] = out["Month"].astype(int)

    # Drop duplicate solute-level rows to station-month hydrology.
    out = out.drop_duplicates(subset=["OBJECTID", "Year", "Month", "mean_Q", "hydro_state"]).copy()

    out["Date"] = pd.to_datetime(
        dict(year=out["Year"], month=out["Month"], day=15),
        errors="coerce"
    )
    out = out.dropna(subset=["Date"]).copy()
    out = out.sort_values(["OBJECTID", "Date"]).reset_index(drop=True)

    return out


def add_seasonal_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    def q20(x):
        return np.nanquantile(x, 0.20)

    def q50(x):
        return np.nanquantile(x, 0.50)

    def q80(x):
        return np.nanquantile(x, 0.80)

    qs = (
        df.groupby(["OBJECTID", "Month"])["mean_Q"]
        .agg(Q20=q20, Q50=q50, Q80=q80)
        .reset_index()
    )

    out = df.merge(qs, on=["OBJECTID", "Month"], how="left")
    return out


def choose_representative_series(df: pd.DataFrame, window_months: int = 60) -> pd.DataFrame:
    """
    Choose one station and one consecutive window rich in non-normal states.
    """
    target_states = ["drought_like", "post_drought_rewetting", "non_drought_highflow", "normal"]

    station_scores = []
    for obj, g in df.groupby("OBJECTID"):
        states = set(g["hydro_state"].dropna().astype(str))
        n_states = sum(s in states for s in target_states)
        n_non_normal = (g["hydro_state"] != "normal").sum()
        n_total = len(g)
        if n_total >= 36:
            station_scores.append((obj, n_states, n_non_normal, n_total))

    if not station_scores:
        obj = df["OBJECTID"].iloc[0]
    else:
        station_scores = sorted(station_scores, key=lambda x: (x[1], x[2], x[3]), reverse=True)
        obj = station_scores[0][0]

    g = df[df["OBJECTID"] == obj].sort_values("Date").copy()

    # Find best rolling window.
    if len(g) <= window_months:
        window = g.copy()
    else:
        best_score = -1
        best_start = 0
        for start in range(0, len(g) - window_months + 1):
            w = g.iloc[start:start + window_months]
            non_normal = (w["hydro_state"] != "normal").sum()
            has_rewet = (w["hydro_state"] == "post_drought_rewetting").sum()
            has_dry = (w["hydro_state"] == "drought_like").sum()
            has_high = (w["hydro_state"] == "non_drought_highflow").sum()
            score = non_normal + 3 * has_rewet + 2 * has_dry + 1.5 * has_high
            if score > best_score:
                best_score = score
                best_start = start
        window = g.iloc[best_start:best_start + window_months].copy()

    with open(OUT_REPRESENTATIVE_REPORT, "w", encoding="utf-8") as f:
        f.write("Fig. 3 representative station report\n")
        f.write("===================================\n")
        f.write(f"Selected OBJECTID: {obj}\n")
        f.write(f"Window start: {window['Date'].min().date()}\n")
        f.write(f"Window end: {window['Date'].max().date()}\n")
        f.write("State counts in window:\n")
        f.write(window["hydro_state"].value_counts().to_string())
        f.write("\n")

    print(f"[OK] Fig. 3 representative OBJECTID: {obj}")
    print(f"[OK] Fig. 3 window: {window['Date'].min().date()} to {window['Date'].max().date()}")
    print(f"[OK] Representative report: {OUT_REPRESENTATIVE_REPORT}")

    return window


# =========================================================
# 4. FIG. 3
# =========================================================
def make_fig3():
    df = prepare_monthly_hydro_table()
    df = add_seasonal_thresholds(df)
    w = choose_representative_series(df, window_months=60)

    fig = plt.figure(figsize=(14.2, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.38, 1.02, 1.15], wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # -------------------------------------------------
    # (a) Real monthly Q series and seasonal thresholds
    # -------------------------------------------------
    dates = w["Date"]
    q = w["mean_Q"]

    # shade states
    ymin = max(0, np.nanmin(q) * 0.75)
    ymax = np.nanmax(q) * 1.25

    for _, row in w.iterrows():
        state = str(row["hydro_state"])
        color = STATE_COLORS.get(state, "#e5e5e5")
        ax1.axvspan(
            row["Date"] - pd.Timedelta(days=15),
            row["Date"] + pd.Timedelta(days=15),
            color=color,
            alpha=0.13 if state != "normal" else 0.06,
            linewidth=0,
            zorder=0,
        )

    ax1.plot(dates, q, color="#4a4a4a", linewidth=1.6, marker="o", markersize=2.8, zorder=4)
    ax1.plot(dates, w["Q20"], color=STATE_COLORS["drought_like"], linewidth=1.1, linestyle="--", zorder=2, label="Q20")
    ax1.plot(dates, w["Q50"], color="#777777", linewidth=1.0, linestyle="--", zorder=2, label="Q50")
    ax1.plot(dates, w["Q80"], color=STATE_COLORS["non_drought_highflow"], linewidth=1.1, linestyle="--", zorder=2, label="Q80")

    for state in ["drought_like", "post_drought_rewetting", "non_drought_highflow"]:
        tmp = w[w["hydro_state"] == state]
        if len(tmp) > 0:
            ax1.scatter(
                tmp["Date"],
                tmp["mean_Q"],
                s=28,
                color=STATE_COLORS[state],
                edgecolor="white",
                linewidth=0.35,
                zorder=5,
                label=STATE_LABELS[state],
            )

    ax1.set_ylim(ymin, ymax)
    ax1.set_title("Monthly discharge states and seasonal thresholds", pad=8)
    ax1.set_ylabel("Monthly discharge, Q")
    ax1.set_xlabel("Time")
    ax1.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for tick in ax1.get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha("right")

    clean_axes(ax1)
    add_panel_label(ax1, "(a)")

    handles, labels = ax1.get_legend_handles_labels()
    # de-duplicate
    seen = set()
    handles2, labels2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            handles2.append(h)
            labels2.append(l)
            seen.add(l)

    ax1.legend(handles2, labels2, loc="upper left", frameon=False, ncol=2, columnspacing=0.9, handletextpad=0.5)

    # -------------------------------------------------
    # (b) Operational state partition
    # -------------------------------------------------
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis("off")
    ax2.set_title("Operational state partition", pad=8)
    add_panel_label(ax2, "(b)")

    draw_box(ax2, (0.7, 4.55), 2.15, 0.68, "drought-like\nmonth", facecolor="#f1d8d8",
             edgecolor=STATE_COLORS["drought_like"], fontsize=9.6)
    draw_box(ax2, (3.7, 4.55), 2.45, 0.68, "post-drought\nrewetting month", facecolor="#dceeee",
             edgecolor=STATE_COLORS["post_drought_rewetting"], fontsize=9.4)
    draw_box(ax2, (6.95, 4.55), 2.35, 0.68, "non-drought\nhigh-flow month", facecolor="#dce6f5",
             edgecolor=STATE_COLORS["non_drought_highflow"], fontsize=9.4)

    draw_arrow(ax2, (2.85, 4.89), (3.7, 4.89))
    draw_arrow(ax2, (6.15, 4.89), (6.95, 4.89))

    ax2.text(1.78, 3.68, r"$Q_{i,t} \leq Q20_{i,m}$", ha="center", va="center", fontsize=10)
    ax2.text(
        4.93,
        3.60,
        r"$S_{i,t}=D,\ r\in\{1,2\}$" + "\n" + r"$Q_{i,t+r}>Q50_{i,m+r}$",
        ha="center",
        va="center",
        fontsize=9.6,
    )
    ax2.text(
        8.12,
        3.60,
        r"$Q_{i,t}\geq Q80_{i,m}$" + "\n" + r"and $S_{i,t}\neq R$",
        ha="center",
        va="center",
        fontsize=9.6,
    )

    draw_box(ax2, (3.8, 1.68), 2.35, 0.62, "normal month", facecolor="#eeeeee", edgecolor="#8a8a8a", fontsize=9.6)
    ax2.text(5.0, 2.77, r"$S_{i,t}\notin\{D,R,H\}$", ha="center", va="center", fontsize=10)
    draw_arrow(ax2, (5.0, 2.55), (5.0, 2.30), color="#777777")

    ax2.text(
        5.0,
        0.82,
        "Thresholds are defined for each\nstation and calendar month",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.32", fc="#f8f8f8", ec="#bbbbbb", lw=0.8),
    )

    # -------------------------------------------------
    # (c) Temporal alignment
    # -------------------------------------------------
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 6)
    ax3.axis("off")
    ax3.set_title("Temporal alignment for monthly analysis", pad=8)
    add_panel_label(ax3, "(c)")

    draw_box(ax3, (0.55, 4.68), 3.0, 0.65, "Hydrological state label", facecolor="#f6f6f6", fontsize=9.5)
    draw_box(ax3, (0.90, 3.75), 2.25, 0.46, "drought-like", facecolor="#f1d8d8",
             edgecolor=STATE_COLORS["drought_like"], fontsize=9.0)
    draw_box(ax3, (0.90, 3.12), 2.25, 0.46, "post-drought rewetting", facecolor="#dceeee",
             edgecolor=STATE_COLORS["post_drought_rewetting"], fontsize=8.7)
    draw_box(ax3, (0.90, 2.49), 2.25, 0.46, "non-drought high flow", facecolor="#dce6f5",
             edgecolor=STATE_COLORS["non_drought_highflow"], fontsize=8.7)
    draw_box(ax3, (0.90, 1.86), 2.25, 0.46, "normal", facecolor="#eeeeee", edgecolor="#8a8a8a", fontsize=9.0)

    draw_box(ax3, (4.05, 4.66), 1.88, 0.68, "Join by\nOBJECTID–Year–Month", facecolor="#fff6df",
             edgecolor="#c4913d", fontsize=9.1)

    draw_box(ax3, (6.55, 4.68), 2.85, 0.65, "Monthly WRTDS outputs", facecolor="#f6f6f6", fontsize=9.5)
    draw_box(ax3, (6.88, 3.86), 2.20, 0.42, "mean_C / mean_FNC", facecolor="#ffffff", fontsize=8.8)
    draw_box(ax3, (6.88, 3.25), 2.20, 0.42, "mean_Flux / mean_FNFlux", facecolor="#ffffff", fontsize=8.8)
    draw_box(ax3, (6.88, 2.64), 2.20, 0.42, "mean_Q", facecolor="#ffffff", fontsize=8.8)
    draw_box(ax3, (6.88, 2.03), 2.20, 0.42, "state-grouped c–Q", facecolor="#ffffff", fontsize=8.8)

    draw_arrow(ax3, (3.55, 5.0), (4.05, 5.0))
    draw_arrow(ax3, (5.93, 5.0), (6.55, 5.0))

    ax3.text(
        5.0,
        1.05,
        "No daily-to-monthly remapping is required:\nstate labels and water-quality outputs share the same monthly time step.",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f8f8f8", ec="#bbbbbb", lw=0.8),
    )

    fig.savefig(FIG3_TIFF, dpi=600, bbox_inches="tight")
    fig.savefig(FIG3_PDF, dpi=600, bbox_inches="tight")
    fig.savefig(FIG3_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Fig. 3 generated successfully ===")
    print(f"TIFF: {FIG3_TIFF}")
    print(f"PDF : {FIG3_PDF}")
    print(f"PNG : {FIG3_PNG}")


# =========================================================
# 5. FIG. 4
# =========================================================
def make_fig4():
    c_obs = "#4f78b5"
    c_norm = "#d28b3d"
    c_cq_normal = "#777777"
    c_cq_rewet = "#b65f5f"
    c_cq_high = "#4f9a9a"

    fig = plt.figure(figsize=(14.2, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.20], wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # -------------------------------------------------
    # (a) Concentration anomaly
    # -------------------------------------------------
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.axis("off")
    ax1.set_title("Concentration anomaly", pad=8)
    add_panel_label(ax1, "(a)")

    draw_box(ax1, (0.75, 4.50), 2.55, 0.75, "Observed monthly\nconcentration\nmean_C",
             facecolor="#e1e9f5", edgecolor=c_obs, fontsize=9.6)
    draw_box(ax1, (0.75, 1.20), 2.55, 0.75, "Flow-normalized\nconcentration\nmean_FNC",
             facecolor="#f8e7d2", edgecolor=c_norm, fontsize=9.6)

    draw_box(ax1, (4.05, 2.65), 2.55, 1.05,
             r"$CA_{i,t,s}=\ln\left(\frac{mean\_C}{mean\_FNC}\right)$",
             facecolor="#f7f7f7", edgecolor="#555555", fontsize=10.0)

    draw_arrow(ax1, (3.30, 4.85), (4.05, 3.45), color=c_obs)
    draw_arrow(ax1, (3.30, 1.55), (4.05, 2.90), color=c_norm)

    ax1.text(7.05, 4.15, r"$CA>0$", ha="left", va="center", fontsize=10)
    ax1.text(7.05, 3.72, "concentration above\nflow-normalized baseline", ha="left", va="center", fontsize=9.0)
    ax1.text(7.05, 2.15, r"$CA<0$", ha="left", va="center", fontsize=10)
    ax1.text(7.05, 1.72, "concentration below\nflow-normalized baseline", ha="left", va="center", fontsize=9.0)

    # -------------------------------------------------
    # (b) Flux anomaly
    # -------------------------------------------------
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis("off")
    ax2.set_title("Flux anomaly", pad=8)
    add_panel_label(ax2, "(b)")

    draw_box(ax2, (0.75, 4.50), 2.55, 0.75, "Observed monthly\nconstituent flux\nmean_Flux",
             facecolor="#e1e9f5", edgecolor=c_obs, fontsize=9.6)
    draw_box(ax2, (0.75, 1.20), 2.55, 0.75, "Flow-normalized\nconstituent flux\nmean_FNFlux",
             facecolor="#f8e7d2", edgecolor=c_norm, fontsize=9.6)

    draw_box(ax2, (4.05, 2.65), 2.55, 1.05,
             r"$FA_{i,t,s}=\ln\left(\frac{mean\_Flux}{mean\_FNFlux}\right)$",
             facecolor="#f7f7f7", edgecolor="#555555", fontsize=9.8)

    draw_arrow(ax2, (3.30, 4.85), (4.05, 3.45), color=c_obs)
    draw_arrow(ax2, (3.30, 1.55), (4.05, 2.90), color=c_norm)

    ax2.text(7.05, 4.15, r"$FA>0$", ha="left", va="center", fontsize=10)
    ax2.text(7.05, 3.72, "export above\nflow-normalized baseline", ha="left", va="center", fontsize=9.0)
    ax2.text(7.05, 2.15, r"$FA<0$", ha="left", va="center", fontsize=10)
    ax2.text(7.05, 1.72, "export below\nflow-normalized baseline", ha="left", va="center", fontsize=9.0)

    # -------------------------------------------------
    # (c) c-Q relationship change
    # -------------------------------------------------
    x = np.linspace(0.8, 9.2, 80)
    y_normal = 1.15 + 0.22 * x
    y_rewet = 0.82 + 0.42 * x
    y_high = 1.35 + 0.10 * x

    ax3.plot(x, y_normal, color=c_cq_normal, linewidth=1.9, label=r"normal: $\beta_N$")
    ax3.plot(x, y_rewet, color=c_cq_rewet, linewidth=2.0, label=r"post-drought rewetting: $\beta_R$")
    ax3.plot(x, y_high, color=c_cq_high, linewidth=2.0, label=r"non-drought high flow: $\beta_H$")

    # schematic points
    rng = np.random.default_rng(42)
    xs = np.linspace(1.2, 8.5, 18)
    ax3.scatter(xs, 1.15 + 0.22 * xs + rng.normal(0, 0.10, len(xs)), s=16, color=c_cq_normal, alpha=0.45, edgecolor="none")
    ax3.scatter(xs, 0.82 + 0.42 * xs + rng.normal(0, 0.12, len(xs)), s=16, color=c_cq_rewet, alpha=0.45, edgecolor="none")
    ax3.scatter(xs, 1.35 + 0.10 * xs + rng.normal(0, 0.10, len(xs)), s=16, color=c_cq_high, alpha=0.45, edgecolor="none")

    ax3.set_xlim(0.5, 9.5)
    ax3.set_ylim(0.85, 5.35)
    ax3.set_xlabel(r"$\ln(Q)$")
    ax3.set_ylabel(r"$\ln(C)$")
    ax3.set_title("State-dependent c–Q relationship", pad=8)
    add_panel_label(ax3, "(c)")
    clean_axes(ax3)

    ax3.text(
        0.06,
        0.92,
        r"$\Delta\beta_{i,s,k}=\beta_{i,s,k}-\beta_{i,s,N}$",
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.30", fc="#f7f7f7", ec="#9a9a9a", lw=0.8),
    )

    ax3.legend(loc="lower right", frameon=False, handlelength=2.5)

    fig.savefig(FIG4_TIFF, dpi=600, bbox_inches="tight")
    fig.savefig(FIG4_PDF, dpi=600, bbox_inches="tight")
    fig.savefig(FIG4_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Fig. 4 generated successfully ===")
    print(f"TIFF: {FIG4_TIFF}")
    print(f"PDF : {FIG4_PDF}")
    print(f"PNG : {FIG4_PNG}")


# =========================================================
# 6. MAIN
# =========================================================
def main():
    make_fig3()
    make_fig4()


if __name__ == "__main__":
    main()