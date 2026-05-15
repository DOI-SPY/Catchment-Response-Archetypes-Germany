from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patheffects as PathEffects
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)

IN_ARCHETYPES = FINAL_DIR / "catchment_response_archetypes.csv"
IN_CENTERS = OUTPUT_DIR / "archetype_cluster_centers.csv"

CATCH_FILE = GIS_DIR / "catchments.shp"
STATION_FILE = GIS_DIR / "stations.shp"
STATION_MOD_FILE = GIS_DIR / "stations_mod.shp"

# 【关键修改】：将保存路径与文件名更新为 fig7
OUT_FIG_PNG = FIG_DIR / "fig7_archetype_spatial_divergence.png"
OUT_FIG_PDF = FIG_DIR / "fig7_archetype_spatial_divergence.pdf"

TARGET_SOLUTES = ["NO3N", "PO4P", "DOC"]

ARCHETYPE_NAMES = {
    "NO3N": {0: "Buffered Chemostatic", 1: "Source-Depleted Dilution"},
    "PO4P": {0: "Surface-Pathway Activation", 1: "Point-Source Dilution"},
    "DOC": {0: "Non-responsive", 1: "Accumulation-Flush Pulse"}
}
COLORS = {0: "#DF9B4B", 1: "#448C82"}  # 0: 橘橙, 1: 森绿

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["mathtext.fontset"] = "stix"


# =========================================================
# 2. 辅助函数与 GIS 处理
# =========================================================
def read_csv_fallback(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except:
            pass
    raise RuntimeError(f"无法读取文件 {path}")


def get_gis_data():
    print("正在加载 GIS 空间图层...")
    catch, st_matched = None, None
    try:
        if CATCH_FILE.exists():
            catch = gpd.read_file(CATCH_FILE)
            if catch.crs is None: catch = catch.set_crs("EPSG:3035")
    except Exception as e:
        print(f"[警告] Catchments 失败: {e}")

    try:
        attr = read_csv_fallback(ATTR_FILE)
        attr["station_key"] = attr["Station"].astype(str).str.strip().str.upper()

        for f in [STATION_FILE, STATION_MOD_FILE]:
            if f.exists():
                st = gpd.read_file(f)
                if st.crs is None: st = st.set_crs("EPSG:3035")
                drop_cols = [c for c in st.columns if str(c).upper() == "OBJECTID"]
                if drop_cols: st = st.drop(columns=drop_cols)

                st_col_candidates = [c for c in st.columns if str(c).lower() == "station"]
                if not st_col_candidates: continue

                st["station_key"] = st[st_col_candidates[0]].astype(str).str.strip().str.upper()
                st_matched = st.merge(attr[["OBJECTID", "station_key"]], on="station_key", how="inner")
                break
    except Exception as e:
        print(f"[警告] Stations 失败: {e}")

    return catch, st_matched


# =========================================================
# ✨ 新增：Text-Data Integration 报告生成器
# =========================================================
def print_manuscript_report(centers_df):
    """自动计算硝酸盐(NO3N)各个特征的最大绝对偏差值(Delta)，用于替换原文的定性描述"""
    try:
        sub = centers_df[centers_df["solute"] == "NO3N"].sort_values("Archetype")
        if len(sub) == 2:
            raw_vals = sub[["CA_rewet", "Delta_Beta", "Delta_CV_Ratio"]].values
            delta_ca = abs(raw_vals[0, 0] - raw_vals[1, 0])
            delta_beta = abs(raw_vals[0, 1] - raw_vals[1, 1])
            delta_cv = abs(raw_vals[0, 2] - raw_vals[1, 2])

            print("\n" + "=" * 70)
            print(" 📄 [Text-Data Integration] 专属正文数据填空报告 (New Fig 7) ")
            print("=" * 70)
            print("请将以下真实数据填入本文 3.3 节对应的 [括号] 内：\n")
            print(f"[DATA_DELTA_CA]   = {delta_ca:.2f} (NO3N 的浓度异常脉冲振幅绝对偏差)")
            print(f"[DATA_DELTA_BETA] = {delta_beta:.2f} (NO3N 的 c-Q 斜率特征绝对偏差)")
            print(f"[DATA_DELTA_CV]   = {delta_cv:.2f} (NO3N 的 CV比值特征绝对偏差)")
            print("=" * 70 + "\n")
    except Exception as e:
        print(f"数据报告生成失败 (请检查 {IN_CENTERS.name} 数据格式): {e}")


# =========================================================
# 3. 核心绘图逻辑：特征偏离哑铃图 (Dumbbell Plot)
# =========================================================
def plot_dumbbell(ax, solute, centers_df):
    features = ["CA_rewet", "FA_rewet", "Delta_Beta", "Delta_CV_Ratio"]
    labels = ["CA\n(Pulse)", "FA\n(Export)", r"$\Delta\beta$" + "\n(c-Q Shift)", r"$\Delta CV$" + "\n(Source Shift)"]

    sub = centers_df[centers_df["solute"] == solute].sort_values("Archetype")
    raw_vals = sub[features].values  # Shape: (2, 4)

    log_vals = np.sign(raw_vals) * np.log1p(np.abs(raw_vals) * 10)
    max_abs = np.max(np.abs(log_vals), axis=0)
    max_abs[max_abs == 0] = 1.0
    scaled_vals = log_vals / max_abs

    ax.axvline(0, color="#BDC3C7", linestyle="--", linewidth=1.5, zorder=1)

    for j in range(4):
        if j % 2 == 0:
            ax.axhspan(j - 0.5, j + 0.5, color="#F8F9F9", zorder=0)

    for j in range(4):
        y_pos = 3 - j
        x0, x1 = scaled_vals[0, j], scaled_vals[1, j]
        r0, r1 = raw_vals[0, j], raw_vals[1, j]

        ax.plot([x0, x1], [y_pos, y_pos], color="#95A5A6", linewidth=4.0, zorder=2, alpha=0.6)
        ax.scatter(x0, y_pos, color=COLORS[0], s=350, zorder=3, edgecolors="white", linewidths=2.5)
        ax.scatter(x1, y_pos, color=COLORS[1], s=350, zorder=3, edgecolors="white", linewidths=2.5)

        pe = [PathEffects.withStroke(linewidth=3, foreground='w')]
        if abs(x0 - x1) < 0.15:
            t1 = ax.text(x0, y_pos + 0.25, f"{r0:.2f}", color=COLORS[0], ha="center", va="center", fontsize=13,
                         fontweight="bold")
            t2 = ax.text(x1, y_pos - 0.25, f"{r1:.2f}", color=COLORS[1], ha="center", va="center", fontsize=13,
                         fontweight="bold")
        elif x0 < x1:
            t1 = ax.text(x0 - 0.15, y_pos, f"{r0:.2f}", color=COLORS[0], ha="right", va="center", fontsize=13,
                         fontweight="bold")
            t2 = ax.text(x1 + 0.15, y_pos, f"{r1:.2f}", color=COLORS[1], ha="left", va="center", fontsize=13,
                         fontweight="bold")
        else:
            t1 = ax.text(x0 + 0.15, y_pos, f"{r0:.2f}", color=COLORS[0], ha="left", va="center", fontsize=13,
                         fontweight="bold")
            t2 = ax.text(x1 - 0.15, y_pos, f"{r1:.2f}", color=COLORS[1], ha="right", va="center", fontsize=13,
                         fontweight="bold")
        t1.set_path_effects(pe);
        t2.set_path_effects(pe)

        diff = abs(r0 - r1)
        mid_x = (x0 + x1) / 2
        ax.text(mid_x, y_pos + 0.28, f"$\Delta$ {diff:.2f}", color="#34495E", ha="center", va="center",
                fontsize=11, fontweight="bold", style="italic",
                bbox=dict(facecolor="white", edgecolor="#D5D8DC", boxstyle="round,pad=0.3", alpha=0.9), zorder=4)

    ax.set_yticks([3, 2, 1, 0])
    ax.set_yticklabels(labels, fontsize=14, fontweight="bold")
    ax.set_xlim(-1.6, 1.6);
    ax.set_ylim(-0.8, 3.8);
    ax.set_xticks([])
    ax.set_xlabel("Relative Feature Divergence (Log-Modulus Axis)", fontsize=13, fontweight="bold", color="#7F8C8D")
    sns.despine(ax=ax, left=True, bottom=True)
    ax.tick_params(axis='y', length=0)


# =========================================================
# 4. 整体聚合构建
# =========================================================
def make_figure():
    print("正在聚合空间数据与多维特征 (New Figure 7 哑铃图)...")
    df_arch = read_csv_fallback(IN_ARCHETYPES)
    df_centers = read_csv_fallback(IN_CENTERS)

    # 💡 打印用于文本替换的数据
    print_manuscript_report(df_centers)

    catch_shp, st_shp = get_gis_data()

    fig = plt.figure(figsize=(22, 12.5))

    for col_idx, solute in enumerate(TARGET_SOLUTES):
        ax_map = fig.add_subplot(2, 3, col_idx + 1)
        sub_arch = df_arch[df_arch["solute"] == solute]

        if catch_shp is not None:
            catch_shp.plot(ax=ax_map, facecolor="#F4F6F6", edgecolor="#D5D8DC", linewidth=0.4, zorder=1)

        if st_shp is not None:
            map_data = st_shp.merge(sub_arch[["OBJECTID", "Archetype"]], on="OBJECTID", how="inner")
            if not map_data.empty:
                for arch_id in [0, 1]:
                    arch_points = map_data[map_data["Archetype"] == arch_id]
                    if not arch_points.empty:
                        arch_points.plot(ax=ax_map, color=COLORS[arch_id], markersize=65,
                                         alpha=0.9, edgecolor="white", linewidth=0.8, zorder=3)

        ax_map.set_title(f"({chr(97 + col_idx)}) {solute} Archetypes Map", fontsize=18, fontweight="bold", pad=20)
        ax_map.axis("off")

        legend_elements = [
            mlines.Line2D([0], [0], marker='o', color='w', label=f"Type 0: {ARCHETYPE_NAMES[solute][0]}",
                          markerfacecolor=COLORS[0], markersize=14),
            mlines.Line2D([0], [0], marker='o', color='w', label=f"Type 1: {ARCHETYPE_NAMES[solute][1]}",
                          markerfacecolor=COLORS[1], markersize=14)
        ]
        ax_map.legend(handles=legend_elements, loc="lower left", frameon=True, fontsize=12, facecolor="white",
                      edgecolor="#cccccc", bbox_to_anchor=(-0.05, -0.05))

        ax_dumbbell = fig.add_subplot(2, 3, col_idx + 4)
        plot_dumbbell(ax_dumbbell, solute, df_centers)
        ax_dumbbell.set_title(f"({chr(100 + col_idx)}) {solute} Feature Divergence", fontsize=18, fontweight="bold",
                              pad=25)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    fig.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[完成] 空间特征偏离哑铃图(New Fig 7)已重构完成。")


if __name__ == "__main__":
    make_figure()