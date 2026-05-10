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

# =========================================================
# 1. 动态路径解析 (已修正为本地绝对路径)
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 强制使用您本地真实的 QUADICA_v2 数据仓库绝对路径
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

OUT_FIG_PNG = FIG_DIR / "fig8_archetype_spatial_radar.png"
OUT_FIG_PDF = FIG_DIR / "fig8_archetype_spatial_radar.pdf"

TARGET_SOLUTES = ["NO3N", "PO4P", "DOC"]

ARCHETYPE_NAMES = {
    "NO3N": {0: "Buffered Chemostatic", 1: "Source-Depleted Dilution"},
    "PO4P": {0: "Surface-Pathway Activation", 1: "Point-Source Dilution"},
    "DOC": {0: "Non-responsive", 1: "Accumulation-Flush Pulse"}
}
COLORS = {0: "#df9b4b", 1: "#448c82"}


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

                # 【防冲突核心修复】：彻底剔除 Shapefile 自身可能带有的 OBJECTID 变体
                drop_cols = [c for c in st.columns if str(c).upper() == "OBJECTID"]
                if drop_cols:
                    st = st.drop(columns=drop_cols)

                # 智能识别 station 字段 (规避大小写问题)
                st_col_candidates = [c for c in st.columns if str(c).lower() == "station"]
                if not st_col_candidates:
                    print(f"  -> 找不到站名列，跳过该图层: {f.name}")
                    continue
                st_col = st_col_candidates[0]

                st["station_key"] = st[st_col].astype(str).str.strip().str.upper()

                # 此时合并，得到的 OBJECTID 绝对纯粹无后缀
                st_matched = st.merge(attr[["OBJECTID", "station_key"]], on="station_key", how="inner")
                break
    except Exception as e:
        print(f"[警告] Stations 失败: {e}")

    return catch, st_matched


# =========================================================
# 3. 核心绘图逻辑 (解决奇点塌陷，提升学术美感)
# =========================================================
def plot_radar(ax, solute, centers_df):
    features = ["CA_rewet", "FA_rewet", "Delta_Beta", "Delta_CV_Ratio"]
    labels = ["CA\n(Pulse)", "FA\n(Export)", r"$\Delta\beta$" + "\n(c-Q Shift)", r"$\Delta CV$" + "\n(Source Shift)"]

    sub = centers_df[centers_df["solute"] == solute].sort_values("Archetype")
    num_vars = len(features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    raw_vals = sub[features].values

    # 【核心升级】：对数模变换 (Log-Modulus) 配合防极点压缩
    # 这能极其优雅地处理如 FA=-6.0 导致的中心塌陷问题
    log_vals = np.sign(raw_vals) * np.log1p(np.abs(raw_vals) * 10)
    max_abs = np.max(np.abs(log_vals), axis=0)
    max_abs[max_abs == 0] = 1.0
    scaled_vals = log_vals / max_abs

    # 映射到 [0.1, 0.9]，彻底规避原点塌陷，0.5 依然为绝对基线
    radar_vals = 0.5 + 0.4 * scaled_vals

    # 绘制学术分区背景底色 (Inner = Depletion, Outer = Flush)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.fill_between(theta, 0, 0.5, color="#ffeaea", alpha=0.3, zorder=0)
    ax.fill_between(theta, 0.5, 1.0, color="#eaf7f2", alpha=0.3, zorder=0)
    ax.plot(theta, [0.5] * 100, color="black", linestyle="--", linewidth=1.2, zorder=1)

    # 绘制原型的多边形
    for i, row in enumerate(radar_vals):
        arch_id = sub.iloc[i]["Archetype"]
        values = row.tolist()
        values += values[:1]

        ax.plot(angles, values, color=COLORS[arch_id], linewidth=3.0, linestyle="-", marker="o", markersize=6, zorder=3)
        ax.fill(angles, values, color=COLORS[arch_id], alpha=0.25, zorder=2)

        # 标签渲染：添加白色描边防重叠
        for j, val in enumerate(row):
            actual_val = raw_vals[i][j]
            offset = 0.08 if actual_val >= 0 else -0.08
            txt = ax.text(angles[j], values[j] + offset, f"{actual_val:.2f}",
                          color=COLORS[arch_id], fontsize=11, ha="center", va="center", fontweight="bold")
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])

    # 极坐标样式打磨
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=13, fontweight="bold")
    ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)
    ax.grid(color="#cccccc", linestyle=":", linewidth=1.0)
    ax.spines['polar'].set_visible(False)


def make_figure():
    print("正在聚合空间数据与多维特征...")
    df_arch = read_csv_fallback(IN_ARCHETYPES)
    df_centers = read_csv_fallback(IN_CENTERS)
    catch_shp, st_shp = get_gis_data()

    plt.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=(20, 13))

    for col_idx, solute in enumerate(TARGET_SOLUTES):
        # -----------------------------
        # 上排：空间分布地图 (a, b, c)
        # -----------------------------
        ax_map = fig.add_subplot(2, 3, col_idx + 1)
        sub_arch = df_arch[df_arch["solute"] == solute]

        if catch_shp is not None:
            catch_shp.plot(ax=ax_map, facecolor="#f0f2f5", edgecolor="#cccccc", linewidth=0.3, zorder=1)

        if st_shp is not None:
            map_data = st_shp.merge(sub_arch[["OBJECTID", "Archetype"]], on="OBJECTID", how="inner")
            if not map_data.empty:
                for arch_id in [0, 1]:
                    arch_points = map_data[map_data["Archetype"] == arch_id]
                    if not arch_points.empty:
                        arch_points.plot(ax=ax_map, color=COLORS[arch_id], markersize=60,
                                         alpha=0.9, edgecolor="white", linewidth=0.8, zorder=3)

        # 添加严格的 (a) (b) (c) 编号体系
        ax_map.set_title(f"({chr(97 + col_idx)}) {solute} Archetypes Map", fontsize=16, fontweight="bold", pad=20)
        ax_map.axis("off")

        legend_elements = [
            mlines.Line2D([0], [0], marker='o', color='w', label=f"Type 0: {ARCHETYPE_NAMES[solute][0]}",
                          markerfacecolor=COLORS[0], markersize=14),
            mlines.Line2D([0], [0], marker='o', color='w', label=f"Type 1: {ARCHETYPE_NAMES[solute][1]}",
                          markerfacecolor=COLORS[1], markersize=14)
        ]
        ax_map.legend(handles=legend_elements, loc="lower left", frameon=True, fontsize=12,
                      facecolor="white", edgecolor="#cccccc", bbox_to_anchor=(0.0, 0.0))

        # -----------------------------
        # 下排：特征雷达图 (d, e, f)
        # -----------------------------
        ax_radar = fig.add_subplot(2, 3, col_idx + 4, polar=True)
        plot_radar(ax_radar, solute, df_centers)
        ax_radar.set_title(f"({chr(100 + col_idx)}) {solute} Feature Profile", fontsize=16, fontweight="bold", pad=25)

        if col_idx == 0:
            ax_radar.text(-0.25, -0.15,
                          "--- Baseline (0)\nRed Inner: Depletion / Dilution\nGreen Outer: Accumulation / Flushing",
                          transform=ax_radar.transAxes, fontsize=11, color="#555555", style="italic")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    fig.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[完成] 空间特征雷达图已生成: \n- {OUT_FIG_PNG}\n- {OUT_FIG_PDF}")


if __name__ == "__main__":
    make_figure()