from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================================================
# 1. 动态路径解析
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = PROJECT_ROOT / "data_final"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PANEL = FINAL_DIR / "analysis_panel_with_anomalies.csv"
INPUT_SLOPES = FINAL_DIR / "catchment_cq_slopes_matrix.csv"
OUT_ARCHETYPES = FINAL_DIR / "catchment_response_archetypes.csv"
OUT_CLUSTER_CENTERS = OUTPUT_DIR / "archetype_cluster_centers.csv"

TARGET_SOLUTES = ["NO3N", "PO4P", "DOC"]


# =========================================================
# 2. 特征矩阵提取 (Feature Extraction)
# =========================================================
def build_feature_matrix(panel: pd.DataFrame, slopes: pd.DataFrame) -> pd.DataFrame:
    print("正在构建多维流域响应特征矩阵...")

    # 提取1：再润湿期的浓度与通量异常脉冲 (反映脆弱性与瞬间冲刷强度)
    rewet_panel = panel[panel["hydro_state"] == "post_drought_rewetting"].copy()
    rewet_anomalies = rewet_panel.groupby(["OBJECTID", "solute"])[["CA", "FA"]].median().reset_index()
    rewet_anomalies.rename(columns={"CA": "CA_rewet", "FA": "FA_rewet"}, inplace=True)

    # 提取2：再润湿期的体制跃迁量 (反映源-汇限制的转换)
    # 取 delta_beta_post_drought_rewetting 和 delta_cv_ratio_post_drought_rewetting
    shift_features = slopes[[
        "OBJECTID", "solute",
        "delta_beta_post_drought_rewetting",
        "delta_cv_ratio_post_drought_rewetting"
    ]].copy()

    # 合并特征
    feature_matrix = rewet_anomalies.merge(shift_features, on=["OBJECTID", "solute"], how="inner")

    # 重命名以确保学术规范
    feature_matrix.rename(columns={
        "delta_beta_post_drought_rewetting": "Delta_Beta",
        "delta_cv_ratio_post_drought_rewetting": "Delta_CV_Ratio"
    }, inplace=True)

    # 剔除存在缺失值的行，确保聚类算法正常运行
    feature_matrix = feature_matrix.dropna().copy()
    return feature_matrix


# =========================================================
# 3. 聚类算法寻优与执行 (K-Means with Silhouette Analysis)
# =========================================================
def perform_clustering(df_features: pd.DataFrame, max_k: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    features_cols = ["CA_rewet", "FA_rewet", "Delta_Beta", "Delta_CV_Ratio"]
    X = df_features[features_cols].values

    # 数据标准化 (Z-score 归一化，极其关键，消除量纲影响)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 轮廓系数寻优 (寻找最佳聚类数 K)
    best_k = 3
    best_score = -1

    print(f"正在进行 K-Means 聚类寻优 (测试 K=2 到 {max_k})...")
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"  K={k}, 轮廓系数 Silhouette Score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    # 【注】水文研究中通常锁定 K=3 或 4 具有最佳的物理可解释性。
    # 这里我们强制选择具有最高轮廓系数的 K
    print(f"-> 选定最优聚类数: K={best_k}")

    # 最终模型拟合
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    df_features["Archetype"] = final_kmeans.fit_predict(X_scaled)

    # 逆标准化获取聚类中心 (Cluster Centers) 用于物理解释
    centers = scaler.inverse_transform(final_kmeans.cluster_centers_)
    df_centers = pd.DataFrame(centers, columns=features_cols)
    df_centers.index.name = "Archetype"
    df_centers.reset_index(inplace=True)

    return df_features, df_centers


# =========================================================
# 4. 主控程序
# =========================================================
def main():
    if not INPUT_PANEL.exists() or not INPUT_SLOPES.exists():
        raise FileNotFoundError("未找到输入文件，请确保 01, 02, 03 脚本已成功运行。")

    panel = pd.read_csv(INPUT_PANEL, low_memory=False, encoding="utf-8-sig")
    slopes = pd.read_csv(INPUT_SLOPES, low_memory=False, encoding="utf-8-sig")

    # 针对每种溶质独立聚类 (因为 NO3 和 DOC 的响应机制截然不同)
    final_archetypes = []
    final_centers = []

    for solute in TARGET_SOLUTES:
        print(f"\n========== 开始处理溶质: {solute} ==========")
        solute_panel = panel[panel["solute"] == solute]
        solute_slopes = slopes[slopes["solute"] == solute]

        feature_matrix = build_feature_matrix(solute_panel, solute_slopes)

        if len(feature_matrix) < 20:
            print(f"[警告] {solute} 的有效特征样本数过少 ({len(feature_matrix)})，跳过聚类。")
            continue

        print(f"提取出 {len(feature_matrix)} 个流域站点的有效多维响应特征。")
        clustered_df, centers_df = perform_clustering(feature_matrix)

        # 【修正】：clustered_df 在 merge 时已保留 solute 列，无需重复插入
        # 仅为纯特征矩阵 centers_df 插入 solute 标签
        centers_df.insert(0, "solute", solute)

        final_archetypes.append(clustered_df)
        final_centers.append(centers_df)

    # 合并并保存结果
    if final_archetypes:
        all_archetypes = pd.concat(final_archetypes, ignore_index=True)
        all_centers = pd.concat(final_centers, ignore_index=True)

        all_archetypes.to_csv(OUT_ARCHETYPES, index=False, encoding="utf-8-sig")
        all_centers.to_csv(OUT_CLUSTER_CENTERS, index=False, encoding="utf-8-sig")

        print("\n[聚类分析完成]")
        print(f"- 流域原型分类结果: {OUT_ARCHETYPES}")
        print(f"- 聚类中心 (物理解释用): {OUT_CLUSTER_CENTERS}")

        print("\n[各类原型中心坐标预览 (以决定原型的物理命名)]")
        print(all_centers.to_string(index=False))


if __name__ == "__main__":
    main()