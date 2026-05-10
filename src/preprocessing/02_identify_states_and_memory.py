from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = PROJECT_ROOT / "data_final"
INPUT_FILE = FINAL_DIR / "analysis_panel_station_month_main.csv"
OUTPUT_FILE = FINAL_DIR / "analysis_panel_station_month_states.csv"


def assign_states_and_memory(df: pd.DataFrame) -> pd.DataFrame:
    print("开始计算水文记忆效应 (Hydrological Memory) 并识别复合状态...")
    out = df.copy()
    out = out.sort_values(["OBJECTID", "Year", "Month"]).reset_index(drop=True)

    # ---------------------------------------------------------
    # 第一层跃升：水文记忆效应 (Catchment Memory) 量化
    # ---------------------------------------------------------
    # 计算当月水分亏缺 (Moisture Deficit = PET - PRE). 若大于0则为亏缺。
    out["moisture_deficit"] = out["pet"] - out["pre"]

    # 计算过去 3 个月和 6 个月的累积水分亏缺 (CMD_3m, CMD_6m)
    # 这是 JoH 级别文章中表征 "干旱严重度 (Drought Severity)" 的核心变量
    def calc_rolling_deficit(series, window):
        return series.rolling(window, min_periods=1).sum()

    out["CMD_3m"] = out.groupby("OBJECTID")["moisture_deficit"].transform(lambda x: calc_rolling_deficit(x, 3))
    out["CMD_6m"] = out.groupby("OBJECTID")["moisture_deficit"].transform(lambda x: calc_rolling_deficit(x, 6))

    # ---------------------------------------------------------
    # 第二层跃升：基于月度分布分位数的极速向量化状态识别
    # ---------------------------------------------------------
    # 按站点和日历月计算流量的长期背景分布
    thresholds = out.groupby(["OBJECTID", "Month"])["mean_Q"].agg(
        q20=lambda x: x.quantile(0.20),
        q50=lambda x: x.quantile(0.50),
        q80=lambda x: x.quantile(0.80)
    ).reset_index()

    out = out.merge(thresholds, on=["OBJECTID", "Month"], how="left")

    # 1. 基础状态标识
    out["is_drought_like"] = (out["mean_Q"] <= out["q20"])
    out["is_highflow_candidate"] = (out["mean_Q"] >= out["q80"])

    # 2. 利用 shift() 进行极速再润湿识别 (替代慢速 for 循环)
    # 判断前一个月 (t-1) 或前两个月 (t-2) 是否为干旱
    out["drought_lag1"] = out.groupby("OBJECTID")["is_drought_like"].shift(1).fillna(False)
    out["drought_lag2"] = out.groupby("OBJECTID")["is_drought_like"].shift(2).fillna(False)

    # 再润湿条件：前期存在干旱，且当月流量反弹恢复到历史中位数以上
    out["is_rewetting"] = (out["drought_lag1"] | out["drought_lag2"]) & (out["mean_Q"] > out["q50"])

    # 3. 严格分配唯一水文状态标签
    # 优先级：drought -> rewetting -> non_drought_highflow -> normal
    conditions = [
        out["is_drought_like"],
        out["is_rewetting"] & ~out["is_drought_like"],
        out["is_highflow_candidate"] & ~out["is_rewetting"] & ~out["is_drought_like"]
    ]
    choices = [
        "drought_like",
        "post_drought_rewetting",
        "non_drought_highflow"
    ]
    out["hydro_state"] = np.select(conditions, choices, default="normal")

    # 清理辅助列，保持数据框干净
    drop_cols = ["moisture_deficit", "is_drought_like", "is_highflow_candidate", "drought_lag1", "drought_lag2",
                 "is_rewetting"]
    out = out.drop(columns=drop_cols)

    return out


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到面板数据：{INPUT_FILE}，请先运行 01_build_monthly_panel.py")

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df_states = assign_states_and_memory(df)

    df_states.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n[状态分布统计摘要]")
    print(df_states["hydro_state"].value_counts(normalize=True).map(lambda x: f"{x:.1%}"))
    print(f"\n[完成] 状态识别与水文记忆量化已完成！保存在: {OUTPUT_FILE}")
    print("现在您的模型不仅知道何时干旱，还知道了流域‘渴了多久 (CMD_3m)’！")


if __name__ == "__main__":
    main()