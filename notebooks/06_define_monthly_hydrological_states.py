from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = PROJECT_ROOT / "data_final"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = FINAL_DIR / "analysis_panel_station_month.csv"
OUTPUT_FILE = FINAL_DIR / "analysis_panel_station_month_states.csv"
SUMMARY_FILE = REPORT_DIR / "monthly_state_summary_report.csv"


TARGET_SOLUTES = ["NO3N", "PO4P", "DOC"]


def safe_qcut_threshold(x: pd.Series, q: float) -> float:
    """
    返回分位数阈值；如果全为空则返回 NaN
    """
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return x.quantile(q)


def assign_monthly_states(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 统一类型
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out["Month"] = pd.to_numeric(out["Month"], errors="coerce").astype("Int64")
    out["mean_Q"] = pd.to_numeric(out["mean_Q"], errors="coerce")
    out["OBJECTID"] = pd.to_numeric(out["OBJECTID"], errors="coerce").astype("Int64")

    # 只保留目标变量
    out = out[out["solute"].isin(TARGET_SOLUTES)].copy()

    # 同一站点-月份对不同变量会重复出现，所以“状态识别”先按站点-月份唯一化
    month_hydro = (
        out[["OBJECTID", "Year", "Month", "YearMonth", "mean_Q"]]
        .drop_duplicates(subset=["OBJECTID", "Year", "Month"])
        .copy()
    )

    # 为每个站点、每个月构建月尺度低流量/高流量阈值
    # 这里采用“同站点、同月份”的长期分布：
    # Q20 -> drought-like threshold
    # Q80 -> non-drought high-flow threshold
    thresholds = (
        month_hydro.groupby(["OBJECTID", "Month"])["mean_Q"]
        .agg(
            q20=lambda x: safe_qcut_threshold(x, 0.20),
            q80=lambda x: safe_qcut_threshold(x, 0.80),
            q50=lambda x: safe_qcut_threshold(x, 0.50),
            n_obs="count"
        )
        .reset_index()
    )

    month_hydro = month_hydro.merge(
        thresholds,
        on=["OBJECTID", "Month"],
        how="left"
    )

    # 基础状态：drought-like / high-flow candidate
    month_hydro["is_drought_like"] = (
        month_hydro["mean_Q"].notna()
        & month_hydro["q20"].notna()
        & (month_hydro["mean_Q"] <= month_hydro["q20"])
    )

    month_hydro["is_highflow_candidate"] = (
        month_hydro["mean_Q"].notna()
        & month_hydro["q80"].notna()
        & (month_hydro["mean_Q"] >= month_hydro["q80"])
    )

    # 按站点排序，识别连续序列
    month_hydro = month_hydro.sort_values(["OBJECTID", "Year", "Month"]).reset_index(drop=True)

    month_hydro["drought_like_month"] = False
    month_hydro["post_drought_rewetting_month"] = False
    month_hydro["non_drought_highflow_month"] = False
    month_hydro["normal_month"] = False

    # 定义 drought-like month：这里直接用低流量月份
    month_hydro.loc[month_hydro["is_drought_like"], "drought_like_month"] = True

    # 对每个站点识别 post-drought rewetting
    # 规则：
    # 1) 某月是 drought-like
    # 2) 其后第 1 个月如果 mean_Q > q50，则定义为 post-drought rewetting month
    # 3) 如果第 1 个月不满足，第 2 个月满足，也可定义为 post-drought rewetting month
    for obj_id, g in month_hydro.groupby("OBJECTID", sort=False):
        idx = g.index.to_list()
        for i in range(len(idx) - 1):
            cur = idx[i]
            nxt = idx[i + 1]

            if bool(month_hydro.loc[cur, "drought_like_month"]):
                # 先看下一个月
                if (
                    pd.notna(month_hydro.loc[nxt, "mean_Q"])
                    and pd.notna(month_hydro.loc[nxt, "q50"])
                    and month_hydro.loc[nxt, "mean_Q"] > month_hydro.loc[nxt, "q50"]
                ):
                    month_hydro.loc[nxt, "post_drought_rewetting_month"] = True
                elif i + 2 < len(idx):
                    nxt2 = idx[i + 2]
                    if (
                        pd.notna(month_hydro.loc[nxt2, "mean_Q"])
                        and pd.notna(month_hydro.loc[nxt2, "q50"])
                        and month_hydro.loc[nxt2, "mean_Q"] > month_hydro.loc[nxt2, "q50"]
                    ):
                        month_hydro.loc[nxt2, "post_drought_rewetting_month"] = True

    # 非干旱高流量月：高流量候选，但不属于 post-drought rewetting
    month_hydro.loc[
        month_hydro["is_highflow_candidate"] & (~month_hydro["post_drought_rewetting_month"]),
        "non_drought_highflow_month"
    ] = True

    # normal month：既不是 drought-like，也不是 rewetting，也不是 non-drought high flow
    month_hydro.loc[
        (~month_hydro["drought_like_month"])
        & (~month_hydro["post_drought_rewetting_month"])
        & (~month_hydro["non_drought_highflow_month"]),
        "normal_month"
    ] = True

    # 生成单一标签，后面画图和统计都更方便
    month_hydro["hydro_state"] = np.select(
        [
            month_hydro["drought_like_month"],
            month_hydro["post_drought_rewetting_month"],
            month_hydro["non_drought_highflow_month"],
            month_hydro["normal_month"],
        ],
        [
            "drought_like",
            "post_drought_rewetting",
            "non_drought_highflow",
            "normal",
        ],
        default="unclassified"
    )

    # 合并回完整 analysis panel
    out = out.merge(
        month_hydro[
            [
                "OBJECTID", "Year", "Month",
                "q20", "q50", "q80",
                "drought_like_month",
                "post_drought_rewetting_month",
                "non_drought_highflow_month",
                "normal_month",
                "hydro_state"
            ]
        ],
        on=["OBJECTID", "Year", "Month"],
        how="left"
    )

    return out


def make_summary_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # 总体状态统计
    state_counts = df[["OBJECTID", "Year", "Month", "hydro_state"]].drop_duplicates()
    total_unique_station_months = len(state_counts)

    for state, g in state_counts.groupby("hydro_state"):
        rows.append({
            "section": "overall_state_distribution",
            "solute": "ALL",
            "hydro_state": state,
            "n_station_months": len(g),
            "share_of_station_months": len(g) / total_unique_station_months if total_unique_station_months > 0 else np.nan,
            "n_unique_stations": g["OBJECTID"].nunique(),
        })

    # 分变量统计
    for sol in TARGET_SOLUTES:
        sub = df[df["solute"] == sol].copy()
        sub_unique = sub[["OBJECTID", "Year", "Month", "hydro_state"]].drop_duplicates()
        total = len(sub_unique)

        for state, g in sub_unique.groupby("hydro_state"):
            rows.append({
                "section": "solute_specific_state_distribution",
                "solute": sol,
                "hydro_state": state,
                "n_station_months": len(g),
                "share_of_station_months": len(g) / total if total > 0 else np.nan,
                "n_unique_stations": g["OBJECTID"].nunique(),
            })

    return pd.DataFrame(rows)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"未找到输入文件: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"[OK] input loaded: {df.shape}")

    df_states = assign_monthly_states(df)
    summary = make_summary_report(df_states)

    df_states.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

    print("\n=== 月尺度复合水文状态识别完成 ===")
    print(f"analysis_panel_station_month_states: {df_states.shape}")
    print("\n=== 状态摘要（前20行） ===")
    print(summary.head(20).to_string(index=False))
    print(f"\n输出文件：\n- {OUTPUT_FILE}\n- {SUMMARY_FILE}")


if __name__ == "__main__":
    main()