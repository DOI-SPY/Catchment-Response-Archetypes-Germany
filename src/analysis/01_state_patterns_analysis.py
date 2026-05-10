from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = PROJECT_ROOT / "data_final"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = FINAL_DIR / "analysis_panel_station_month_states.csv"
TARGET_SOLUTES = ["NO3N", "PO4P", "DOC"]
STATE_ORDER = ["drought_like", "post_drought_rewetting", "non_drought_highflow", "normal"]


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"未找到输入文件: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    return df


def build_summary_tables(df: pd.DataFrame):
    # 1) overall：按唯一 station-month 去重
    overall = (
        df[["OBJECTID", "Year", "Month", "hydro_state"]]
        .drop_duplicates()
        .groupby("hydro_state")
        .size()
        .reset_index(name="n_station_months")
    )
    total_overall = overall["n_station_months"].sum()
    overall["share"] = overall["n_station_months"] / total_overall

    # 2) by solute
    by_solute = (
        df[["OBJECTID", "Year", "Month", "solute", "hydro_state"]]
        .drop_duplicates()
        .groupby(["solute", "hydro_state"])
        .size()
        .reset_index(name="n_station_months")
    )
    total_by_solute = by_solute.groupby("solute")["n_station_months"].transform("sum")
    by_solute["share"] = by_solute["n_station_months"] / total_by_solute

    # 3) seasonality：按月份和状态统计
    monthly_state = (
        df[["OBJECTID", "Year", "Month", "hydro_state"]]
        .drop_duplicates()
        .groupby(["Month", "hydro_state"])
        .size()
        .reset_index(name="n_station_months")
    )
    total_by_month = monthly_state.groupby("Month")["n_station_months"].transform("sum")
    monthly_state["share_within_month"] = monthly_state["n_station_months"] / total_by_month

    # 4) seasonality by solute
    monthly_state_solute = (
        df[["OBJECTID", "Year", "Month", "solute", "hydro_state"]]
        .drop_duplicates()
        .groupby(["solute", "Month", "hydro_state"])
        .size()
        .reset_index(name="n_station_months")
    )
    total_solute_month = monthly_state_solute.groupby(["solute", "Month"])["n_station_months"].transform("sum")
    monthly_state_solute["share_within_solute_month"] = (
        monthly_state_solute["n_station_months"] / total_solute_month
    )

    # 5) station-level state frequency
    station_state = (
        df[["OBJECTID", "Year", "Month", "hydro_state"]]
        .drop_duplicates()
        .groupby(["OBJECTID", "hydro_state"])
        .size()
        .reset_index(name="n_station_months")
    )

    station_total = station_state.groupby("OBJECTID")["n_station_months"].transform("sum")
    station_state["share_within_station"] = station_state["n_station_months"] / station_total

    return overall, by_solute, monthly_state, monthly_state_solute, station_state


def save_tables(overall, by_solute, monthly_state, monthly_state_solute, station_state):
    overall.to_csv(TABLE_DIR / "table6_overall_state_distribution.csv", index=False, encoding="utf-8-sig")
    by_solute.to_csv(TABLE_DIR / "table6_state_distribution_by_solute.csv", index=False, encoding="utf-8-sig")
    monthly_state.to_csv(TABLE_DIR / "state_monthly_distribution_overall.csv", index=False, encoding="utf-8-sig")
    monthly_state_solute.to_csv(TABLE_DIR / "state_monthly_distribution_by_solute.csv", index=False, encoding="utf-8-sig")
    station_state.to_csv(TABLE_DIR / "state_distribution_by_station.csv", index=False, encoding="utf-8-sig")


def plot_fig5(overall, by_solute):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # panel a: overall
    overall_plot = overall.copy()
    overall_plot["hydro_state"] = pd.Categorical(overall_plot["hydro_state"], categories=STATE_ORDER, ordered=True)
    overall_plot = overall_plot.sort_values("hydro_state")

    axes[0].bar(overall_plot["hydro_state"], overall_plot["share"])
    axes[0].set_title("(a) Overall state distribution")
    axes[0].set_ylabel("Share of station-months")
    axes[0].tick_params(axis="x", rotation=25)

    # panel b: by solute
    by_solute_plot = by_solute.copy()
    by_solute_plot["hydro_state"] = pd.Categorical(by_solute_plot["hydro_state"], categories=STATE_ORDER, ordered=True)
    by_solute_plot = by_solute_plot.sort_values(["solute", "hydro_state"])

    x = np.arange(len(STATE_ORDER))
    width = 0.25

    for i, sol in enumerate(TARGET_SOLUTES):
        sub = by_solute_plot[by_solute_plot["solute"] == sol].set_index("hydro_state").reindex(STATE_ORDER)
        axes[1].bar(x + (i - 1) * width, sub["share"], width=width, label=sol)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(STATE_ORDER, rotation=25)
    axes[1].set_ylabel("Share within solute")
    axes[1].set_title("(b) State distribution by solute")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_state_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig6(monthly_state_solute):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for ax, sol in zip(axes, TARGET_SOLUTES):
        sub = monthly_state_solute[monthly_state_solute["solute"] == sol].copy()
        pivot = sub.pivot(index="hydro_state", columns="Month", values="share_within_solute_month")
        pivot = pivot.reindex(STATE_ORDER)

        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_title(sol)
        ax.set_xlabel("Month")
        ax.set_xticks(range(12))
        ax.set_xticklabels(range(1, 13))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Share within solute-month")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig6_state_seasonality_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_data()
    print(f"[OK] input loaded: {df.shape}")

    overall, by_solute, monthly_state, monthly_state_solute, station_state = build_summary_tables(df)
    save_tables(overall, by_solute, monthly_state, monthly_state_solute, station_state)

    plot_fig5(overall, by_solute)
    plot_fig6(monthly_state_solute)

    print("\n=== 4.1 第一批结果已生成 ===")
    print("输出表格：")
    print(f"- {TABLE_DIR / 'table6_overall_state_distribution.csv'}")
    print(f"- {TABLE_DIR / 'table6_state_distribution_by_solute.csv'}")
    print(f"- {TABLE_DIR / 'state_monthly_distribution_overall.csv'}")
    print(f"- {TABLE_DIR / 'state_monthly_distribution_by_solute.csv'}")
    print(f"- {TABLE_DIR / 'state_distribution_by_station.csv'}")

    print("\n输出图件：")
    print(f"- {FIG_DIR / 'fig5_state_distribution.png'}")
    print(f"- {FIG_DIR / 'fig6_state_seasonality_heatmap.png'}")

    print("\n=== 总体状态分布 ===")
    print(overall.to_string(index=False))

    print("\n=== 分变量状态分布 ===")
    print(by_solute.to_string(index=False))


if __name__ == "__main__":
    main()