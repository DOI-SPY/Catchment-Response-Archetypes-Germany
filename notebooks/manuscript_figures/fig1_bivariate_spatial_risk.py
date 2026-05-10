from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# =========================================================
# 1. 动态路径解析
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path("D:/quadica_project/data_raw/QUADICA_v2/data/contents/data")
GIS_DIR = RAW_DIR / "gis"
ATTR_FILE = RAW_DIR / "attributes.csv"

FINAL_DIR = PROJECT_ROOT / "data_final"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

IN_PANEL = FINAL_DIR / "analysis_panel_with_anomalies.csv"
CATCH_FILE = GIS_DIR / "catchments.shp"
STATION_FILE = GIS_DIR / "stations.shp"
STATION_MOD_FILE = GIS_DIR / "stations_mod.shp"

OUT_FIG_PNG = FIG_DIR / "fig1_bivariate_spatial_risk.png"
OUT_FIG_PDF = FIG_DIR / "fig1_bivariate_spatial_risk.pdf"

# =========================================================
# 2. 核心配色与辅助函数
# =========================================================
BIVARIATE_COLORS = {
    (0, 0): "#e8e8e8", (1, 0): "#ace4e4", (2, 0): "#5ac8c8",
    (0, 1): "#dfb0d6", (1, 1): "#a5add3", (2, 1): "#5698b9",
    (0, 2): "#be64ac", (1, 2): "#8c62aa", (2, 2): "#3b4994"
}


def get_actual_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_cols = {str(c).lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols: return lower_cols[cand.lower()]
    raise KeyError(f"未找到候选列名: {candidates}")


def read_csv_fallback(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except:
            pass
    raise RuntimeError(f"无法读取文件 {path}")


# =========================================================
# 3. 数据计算与组装
# =========================================================
def build_bivariate_data():
    print("正在聚合双变量脆弱性数据...")
    attr = read_csv_fallback(ATTR_FILE)
    col_agric = get_actual_col(attr, ["f_agric", "f_agric_18", "agri_frac"])
    attr_sub = attr[["OBJECTID", col_agric]].copy()
    attr_sub.rename(columns={col_agric: "Agri_Frac"}, inplace=True)

    panel = read_csv_fallback(IN_PANEL)
    sub_panel = panel[(panel["solute"] == "NO3N") & (panel["hydro_state"] == "post_drought_rewetting")]
    memory = sub_panel.groupby("OBJECTID")["CMD_3m"].mean().reset_index()

    df = attr_sub.merge(memory, on="OBJECTID", how="inner").dropna()
    df["CMD_Class"] = pd.qcut(df["CMD_3m"], q=3, labels=[0, 1, 2]).astype(int)
    df["Agri_Class"] = pd.qcut(df["Agri_Frac"], q=3, labels=[0, 1, 2]).astype(int)
    df["Color"] = df.apply(lambda row: BIVARIATE_COLORS[(row["CMD_Class"], row["Agri_Class"])], axis=1)
    return df


def get_gis_data(biv_df: pd.DataFrame):
    catch, st_matched = None, None
    if CATCH_FILE.exists():
        catch = gpd.read_file(CATCH_FILE)
        if catch.crs is None: catch = catch.set_crs("EPSG:3035")

    for f in [STATION_FILE, STATION_MOD_FILE]:
        if f.exists():
            st = gpd.read_file(f)
            if st.crs is None: st = st.set_crs("EPSG:3035")

            drop_cols = [c for c in st.columns if str(c).upper() == "OBJECTID"]
            if drop_cols: st = st.drop(columns=drop_cols)

            st_col = [c for c in st.columns if str(c).lower() == "station"][0]
            st["station_key"] = st[st_col].astype(str).str.strip().str.upper()

            attr = read_csv_fallback(ATTR_FILE)
            attr["station_key"] = attr["Station"].astype(str).str.strip().str.upper()
            st_matched = st.merge(attr[["OBJECTID", "station_key"]], on="station_key", how="inner")
            st_matched = st_matched.merge(biv_df[["OBJECTID", "Color", "CMD_Class", "Agri_Class"]], on="OBJECTID",
                                          how="inner")
            break

    return catch, st_matched


# =========================================================
# 4. 绘图子模块 (提升顶刊美感)
# =========================================================
def draw_bivariate_legend(ax):
    """绘制 3x3 双变量颜色图例 (增加纯白底衬防重叠)"""
    ax.set_aspect('equal')
    ax.axis('off')

    # 绘制一层白色不透明背景遮挡底层地图点
    bg_rect = patches.Rectangle((-1.2, -1.2), 5.5, 5.5, linewidth=0, facecolor='white', alpha=0.85, zorder=0)
    ax.add_patch(bg_rect)

    sz = 1.0
    for i in range(3):
        for j in range(3):
            color = BIVARIATE_COLORS[(i, j)]
            # 加入细致的白色边框提升精致感
            rect = patches.Rectangle((i * sz, j * sz), sz, sz, linewidth=1.5, edgecolor='white', facecolor=color,
                                     zorder=1)
            ax.add_patch(rect)

    # 文字排版优化
    ax.text(1.5 * sz, -0.3 * sz, "Drought Memory (CMD)\n$\longrightarrow$", ha='center', va='top', fontsize=12,
            fontweight='bold', color="#1d6487", zorder=2)
    ax.text(-0.3 * sz, 1.5 * sz, "Agricultural Frac.\n$\longrightarrow$", ha='right', va='center', rotation=90,
            fontsize=12, fontweight='bold', color="#9e116b", zorder=2)
    ax.set_xlim(-1.5, 3.5)
    ax.set_ylim(-1.5, 3.5)


def plot_waffle_chart(ax, df):
    """(c) 华夫饼图：呈现各脆弱性等级的绝对流域占比 (学术镂空版)"""
    ax.axis('off')
    counts = df.groupby(["CMD_Class", "Agri_Class"]).size()
    total = len(df)

    colors_list = []
    order = [(2, 2), (2, 1), (1, 2), (2, 0), (0, 2), (1, 1), (1, 0), (0, 1), (0, 0)]
    for key in order:
        if key in counts:
            num_squares = int(round((counts[key] / total) * 100))
            colors_list.extend([BIVARIATE_COLORS[key]] * num_squares)

    while len(colors_list) < 100: colors_list.append("#e8e8e8")
    while len(colors_list) > 100: colors_list.pop()

    # 绘制华夫饼：核心改动在于加入白边和微缩方块，创造网格感
    idx = 0
    for y in range(10):
        for x in range(10):
            color = colors_list[idx]
            # edgecolor="white", linewidth=2.5 创造高级网格留白
            rect = patches.Rectangle((x, 9 - y), 1.0, 1.0, facecolor=color, edgecolor="white", linewidth=2.0)
            ax.add_patch(rect)
            idx += 1

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("(c) Catchment Vulnerability Proportion", fontsize=16, fontweight="bold", pad=15)

    high_risk_pct = (counts.get((2, 2), 0) / total) * 100
    ax.text(5, -1.0, f"High-Agri & High-Drought Basin: {high_risk_pct:.1f}%", ha='center', va='top', fontsize=14,
            fontweight='bold', color="#3b4994")


def plot_marginal_scatter(ax_scatter, ax_hist_x, ax_hist_y, df):
    """(b) 边缘直方图散点：极致精简墨水"""
    sns.scatterplot(data=df, x="CMD_3m", y="Agri_Frac", c=df["Color"], s=70, edgecolor="white", linewidth=0.8,
                    ax=ax_scatter, zorder=3, alpha=0.9)

    # 精简网格线
    ax_scatter.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax_scatter.set_xlabel("Average Drought Memory (CMD)", fontsize=14, fontweight='bold')
    ax_scatter.set_ylabel("Agricultural Fraction", fontsize=14, fontweight='bold')

    # X 轴直方图：去掉边框 (edgecolor="none")
    sns.histplot(data=df, x="CMD_3m", ax=ax_hist_x, color="#5ac8c8", bins=22, fill=True, alpha=0.85, edgecolor="white",
                 linewidth=0.5)
    ax_hist_x.axis('off')

    # Y 轴直方图
    sns.histplot(data=df, y="Agri_Frac", ax=ax_hist_y, color="#be64ac", bins=22, fill=True, alpha=0.85,
                 edgecolor="white", linewidth=0.5)
    ax_hist_y.axis('off')

    sns.despine(ax=ax_scatter)


# =========================================================
# 5. 主排版整合 (顶级拓扑控制)
# =========================================================
def make_figure():
    print("正在生成 Figure 1 (双变量空间基线与脆弱性雷达) - 顶刊美学进化版...")
    plt.rcParams["font.family"] = "Times New Roman"

    biv_df = build_bivariate_data()
    catch_shp, st_shp = get_gis_data(biv_df)

    # 使用黄金分割比例
    fig = plt.figure(figsize=(17, 9))

    # 彻底重构的排版网格：极简 2x2 结构
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.15, 1], height_ratios=[1, 0.75], wspace=0.15, hspace=0.3)

    # -----------------------------
    # (a) 双变量地图 (左侧纵贯两行)
    # -----------------------------
    ax_map = fig.add_subplot(gs[:, 0])
    if catch_shp is not None:
        # 加深底图轮廓线，增加体量感
        catch_shp.plot(ax=ax_map, facecolor="#f8f9fa", edgecolor="#b0b0b0", linewidth=0.6, zorder=1)
    if st_shp is not None:
        st_shp.plot(ax=ax_map, color=st_shp["Color"], markersize=65, edgecolor="white", linewidth=0.8, zorder=3,
                    alpha=0.95)

    ax_map.set_title("(a) Spatial Baseline of Catchment Vulnerability", fontsize=18, fontweight="bold", pad=15)
    ax_map.axis("off")

    # 图例悬浮放置于左下，利用 Bbox 阻隔干扰
    ax_legend = ax_map.inset_axes([0.02, 0.05, 0.35, 0.35])
    draw_bivariate_legend(ax_legend)

    # -----------------------------
    # (b) 边缘直方图散点 (右上角)
    # -----------------------------
    # 内部精细网格，压低边缘直方图的厚度比例 (从 1:3 压到 1:4)
    gs_b = gs[0, 1].subgridspec(4, 4, wspace=0.03, hspace=0.03)
    ax_scatter = fig.add_subplot(gs_b[1:4, 0:3])
    ax_hist_x = fig.add_subplot(gs_b[0, 0:3], sharex=ax_scatter)
    ax_hist_y = fig.add_subplot(gs_b[1:4, 3], sharey=ax_scatter)

    plot_marginal_scatter(ax_scatter, ax_hist_x, ax_hist_y, biv_df)
    fig.text(0.53, 0.90, "(b) Bivariate Distribution & Marginal Density", fontsize=18, fontweight="bold")

    # -----------------------------
    # (c) 华夫饼图 (右下角)
    # -----------------------------
    # 在右下角的区域居中渲染华夫饼
    gs_c = gs[1, 1].subgridspec(1, 1)
    ax_waffle = fig.add_subplot(gs_c[0])
    plot_waffle_chart(ax_waffle, biv_df)

    # 全局保存
    plt.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight")
    plt.savefig(OUT_FIG_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[完成] 空间双变量图已升维重构: \n- {OUT_FIG_PNG}\n- {OUT_FIG_PDF}")


if __name__ == "__main__":
    make_figure()