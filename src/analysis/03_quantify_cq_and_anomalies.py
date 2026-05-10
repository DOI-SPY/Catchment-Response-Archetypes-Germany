from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# =========================================================
# 1. 动态路径解析
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = PROJECT_ROOT / "data_final"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = FINAL_DIR / "analysis_panel_station_month_states.csv"
OUT_PANEL = FINAL_DIR / "analysis_panel_with_anomalies.csv"
OUT_SLOPES = FINAL_DIR / "catchment_cq_slopes_matrix.csv"


# =========================================================
# 2. 核心算法: 计算 CA 和 FA
# =========================================================
def calculate_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算浓度异常 (CA) 和通量异常 (FA)
    公式: CA = ln(mean_C / mean_FNC)
    """
    print("正在计算浓度异常 (CA) 与通量异常 (FA)...")
    out = df.copy()

    mask_c = (out["mean_C"] > 0) & (out["mean_FNC"] > 0)
    out.loc[mask_c, "CA"] = np.log(out.loc[mask_c, "mean_C"] / out.loc[mask_c, "mean_FNC"])

    mask_f = (out["mean_Flux"] > 0) & (out["mean_FNFlux"] > 0)
    out.loc[mask_f, "FA"] = np.log(out.loc[mask_f, "mean_Flux"] / out.loc[mask_f, "mean_FNFlux"])

    return out


# =========================================================
# 3. 核心算法: 计算状态依附 c-Q 斜率 (Beta) 与 化变性指数 (CVc/CVq)
# =========================================================
def calculate_state_dependent_cq_slopes(df: pd.DataFrame) -> pd.DataFrame:
    print("正在量化状态依附的 c-Q 耦合斜率 (Beta) 与化变性指数 (CVc/CVq)...")

    valid_data = df.dropna(subset=["log_mean_C", "log_mean_Q", "mean_C", "mean_Q"]).copy()

    def ols_slope_and_cv(g):
        # 样本量阈值防线
        if len(g) < 5:
            return pd.Series({
                "beta": np.nan, "alpha": np.nan, "n_obs": len(g),
                "cv_c": np.nan, "cv_q": np.nan, "cv_ratio": np.nan
            })

        # 1. 计算对数域斜率 Beta
        x_log = g["log_mean_Q"].values
        y_log = g["log_mean_C"].values
        cov_matrix = np.cov(x_log, y_log)
        var_x = cov_matrix[0, 0]
        cov_xy = cov_matrix[0, 1]

        beta = cov_xy / var_x if var_x > 0 else np.nan
        alpha = np.mean(y_log) - beta * np.mean(x_log) if not np.isnan(beta) else np.nan

        # 2. 计算自然域的化变性指数 (CV = std / mean)
        mean_c_val = g["mean_C"].mean()
        std_c_val = g["mean_C"].std(ddof=1)
        cv_c = std_c_val / mean_c_val if mean_c_val > 0 else np.nan

        mean_q_val = g["mean_Q"].mean()
        std_q_val = g["mean_Q"].std(ddof=1)
        cv_q = std_q_val / mean_q_val if mean_q_val > 0 else np.nan

        cv_ratio = cv_c / cv_q if (pd.notna(cv_c) and pd.notna(cv_q) and cv_q > 0) else np.nan

        return pd.Series({
            "beta": beta, "alpha": alpha, "n_obs": len(g),
            "cv_c": cv_c, "cv_q": cv_q, "cv_ratio": cv_ratio
        })

    # 按站点、溶质、状态进行分组计算
    slopes = valid_data.groupby(["OBJECTID", "solute", "hydro_state"]).apply(ols_slope_and_cv).reset_index()

    # 宽表转换
    slopes_pivot = slopes.pivot(
        index=["OBJECTID", "solute"],
        columns="hydro_state",
        values=["beta", "alpha", "n_obs", "cv_c", "cv_q", "cv_ratio"]
    )
    slopes_pivot.columns = [f"{col[0]}_{col[1]}" for col in slopes_pivot.columns]
    slopes_pivot = slopes_pivot.reset_index()

    # ---------------------------------------------------------
    # 计算复合状态下的体制重塑量 (Delta Beta 与 Delta CV_ratio)
    # ---------------------------------------------------------
    for state in ["drought_like", "post_drought_rewetting", "non_drought_highflow"]:
        # Beta 偏移
        target_beta = f"beta_{state}"
        if target_beta in slopes_pivot.columns and "beta_normal" in slopes_pivot.columns:
            slopes_pivot[f"delta_beta_{state}"] = slopes_pivot[target_beta] - slopes_pivot["beta_normal"]

        # 化变性 (Chemostatic) 演化偏移
        target_cv = f"cv_ratio_{state}"
        if target_cv in slopes_pivot.columns and "cv_ratio_normal" in slopes_pivot.columns:
            slopes_pivot[f"delta_cv_ratio_{state}"] = slopes_pivot[target_cv] - slopes_pivot["cv_ratio_normal"]

    return slopes_pivot


# =========================================================
# 4. 主控程序
# =========================================================
def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到面板数据：{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, low_memory=False, encoding="utf-8-sig")
    df_with_anomalies = calculate_anomalies(df)
    slopes_matrix = calculate_state_dependent_cq_slopes(df_with_anomalies)

    df_with_anomalies.to_csv(OUT_PANEL, index=False, encoding="utf-8-sig")
    slopes_matrix.to_csv(OUT_SLOPES, index=False, encoding="utf-8-sig")

    print("\n[量化完成] 数据已成功保存。")
    print(f"- 面板数据: {OUT_PANEL}")
    print(f"- c-Q 特征矩阵: {OUT_SLOPES}")

    print("\n[关键指标预览: 再润湿期的化变性跃迁]")
    preview = slopes_matrix.dropna(subset=["delta_cv_ratio_post_drought_rewetting"])
    if not preview.empty:
        print(preview[["OBJECTID", "solute", "cv_ratio_normal", "cv_ratio_post_drought_rewetting",
                       "delta_cv_ratio_post_drought_rewetting"]].head())


if __name__ == "__main__":
    main()