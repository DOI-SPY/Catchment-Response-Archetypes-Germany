from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmasher as cmr


# =========================================================
# 0. PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data_raw" / "QUADICA_v2"
DATA_DIR = RAW_DIR / "data" / "contents" / "data"
GIS_DIR = DATA_DIR / "gis"

ATTR_FILE = DATA_DIR / "attributes.csv"
META_Q_FILE = DATA_DIR / "metadata_q.csv"

CATCH_FILE = GIS_DIR / "catchments.shp"
RIVER_FILE = GIS_DIR / "riversegments_mod.shp"
STATION_FILE = GIS_DIR / "stations.shp"
STATION_MOD_FILE = GIS_DIR / "stations_mod.shp"

FINAL_DIR = PROJECT_ROOT / "data_final"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"

MONTHLY_PANEL_FILE = FINAL_DIR / "analysis_panel_station_month_main.csv"
MONTHLY_SUMMARY_FILE = REPORT_DIR / "monthly_panel_summary_report.csv"

FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TIFF = FIG_DIR / "Fig2_study_domain_sample_and_environment_v3.tiff"
OUT_PDF = FIG_DIR / "Fig2_study_domain_sample_and_environment_v3.pdf"
OUT_PNG = FIG_DIR / "Fig2_study_domain_sample_and_environment_v3.png"
OUT_JOIN_REPORT = REPORT_DIR / "fig2_gis_join_report.txt"


# =========================================================
# 1. STYLE
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
plt.rcParams["legend.fontsize"] = 9


# =========================================================
# 2. HELPERS
# =========================================================
def read_csv_fallback(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Failed to read {path.name}: {last_error}")


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s)
    return s.strip().upper()


def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def add_panel_label(ax, label: str):
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12,
        fontweight="bold",
    )


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.7)
    ax.grid(False)


def guess_and_set_crs(gdf: gpd.GeoDataFrame, fallback_crs: str = "EPSG:3035") -> gpd.GeoDataFrame:
    """
    If CRS is missing, guess whether coordinates are geographic or projected.
    QUADICA GIS should normally include .prj, so this is only a safety net.
    """
    out = gdf.copy()
    if out.crs is not None:
        return out

    xmin, ymin, xmax, ymax = out.total_bounds
    if abs(xmin) <= 180 and abs(xmax) <= 180 and abs(ymin) <= 90 and abs(ymax) <= 90:
        out = out.set_crs("EPSG:4326")
    else:
        out = out.set_crs(fallback_crs)

    return out


def to_plot_crs(gdf: gpd.GeoDataFrame, plot_crs: str = "EPSG:3035") -> gpd.GeoDataFrame:
    gdf = guess_and_set_crs(gdf)
    try:
        return gdf.to_crs(plot_crs)
    except Exception:
        return gdf


def read_vector(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing vector file: {path}")
    gdf = gpd.read_file(path)
    gdf = guess_and_set_crs(gdf)
    return gdf


# =========================================================
# 3. DATA LOADING
# =========================================================
def load_station_attributes() -> pd.DataFrame:
    attr = read_csv_fallback(ATTR_FILE)
    meta_q = read_csv_fallback(META_Q_FILE)

    keep_cols = [c for c in ["OBJECTID", "Station", "Station_Q", "River", "Q_flag", "CMLS_match", "caravan"] if c in meta_q.columns]
    meta_q_sub = meta_q[keep_cols].copy()

    df = attr.merge(
        meta_q_sub,
        on="OBJECTID",
        how="left",
        suffixes=("", "_q")
    )

    # keep a single station name field
    if "Station" not in df.columns and "Station_q" in df.columns:
        df["Station"] = df["Station_q"]

    df["station_key"] = df["Station"].map(norm_text)

    if "Q_flag" in df.columns:
        df["Q_flag"] = pd.to_numeric(df["Q_flag"], errors="coerce").fillna(0).astype(int)
    else:
        df["Q_flag"] = 0

    return df


def load_station_gis(attr_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Use stations.shp as primary point layer.
    If merge quality is poor, try stations_mod.shp.
    """
    candidate_files = [STATION_FILE, STATION_MOD_FILE]
    best_gdf = None
    best_match = -1
    best_name = None

    for f in candidate_files:
        if not f.exists():
            continue

        st = read_vector(f)
        station_col = first_existing(st, ["Station", "station", "site", "name"])

        if station_col is None:
            continue

        st = st.copy()
        st["station_key"] = st[station_col].map(norm_text)

        # Avoid duplicate non-key Station columns from attr_df
        attr_join = attr_df.copy()
        attr_join = attr_join.drop(columns=[c for c in ["geometry"] if c in attr_join.columns], errors="ignore")

        gdf = st.merge(
            attr_join,
            on="station_key",
            how="left",
            suffixes=("_gis", "")
        )

        match_n = gdf["OBJECTID"].notna().sum() if "OBJECTID" in gdf.columns else 0

        if match_n > best_match:
            best_match = match_n
            best_gdf = gdf
            best_name = f.name

    if best_gdf is None:
        raise RuntimeError("Could not merge stations.shp / stations_mod.shp with attributes by Station.")

    if "Q_flag" in best_gdf.columns:
        best_gdf["Q_flag"] = pd.to_numeric(best_gdf["Q_flag"], errors="coerce").fillna(0).astype(int)
    else:
        best_gdf["Q_flag"] = 0

    with open(OUT_JOIN_REPORT, "w", encoding="utf-8") as f:
        f.write("Fig. 2 GIS join report\n")
        f.write("======================\n")
        f.write(f"Selected station layer: {best_name}\n")
        f.write(f"Total station geometries: {len(best_gdf)}\n")
        f.write(f"Matched to attributes by station name: {best_match}\n")
        f.write(f"Paired discharge stations after join: {int((best_gdf['Q_flag'] == 1).sum())}\n")
        f.write(f"CRS: {best_gdf.crs}\n")

    print(f"[OK] selected station layer: {best_name}")
    print(f"[OK] station geometries: {len(best_gdf)}")
    print(f"[OK] matched stations: {best_match}")
    print(f"[OK] paired discharge stations: {int((best_gdf['Q_flag'] == 1).sum())}")
    print(f"[OK] join report: {OUT_JOIN_REPORT}")

    return best_gdf


def load_gis_layers():
    catchments = read_vector(CATCH_FILE)
    stations_attr = load_station_attributes()
    stations = load_station_gis(stations_attr)

    rivers = None
    if RIVER_FILE.exists():
        try:
            rivers = read_vector(RIVER_FILE)
        except Exception:
            rivers = None

    return catchments, rivers, stations, stations_attr


def get_sample_summary() -> pd.DataFrame:
    if MONTHLY_PANEL_FILE.exists():
        panel = read_csv_fallback(MONTHLY_PANEL_FILE)

        out = (
            panel.groupby("solute")
            .agg(
                n_rows=("solute", "size"),
                n_unique_stations=("OBJECTID", pd.Series.nunique),
                year_min=("Year", "min"),
                year_max=("Year", "max"),
            )
            .reset_index()
        )
    elif MONTHLY_SUMMARY_FILE.exists():
        out = read_csv_fallback(MONTHLY_SUMMARY_FILE)
    else:
        raise FileNotFoundError("No monthly panel or monthly summary report found.")

    # Normalize field names
    required = ["solute", "n_rows", "n_unique_stations", "year_min", "year_max"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"Sample summary missing fields: {missing}")

    order = ["NO3N", "PO4P", "DOC"]
    out = out[required].copy()
    out["solute"] = pd.Categorical(out["solute"], categories=order, ordered=True)
    out = out.sort_values("solute").reset_index(drop=True)
    return out


def prepare_environment_data(attr_df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "agri_frac": first_existing(attr_df, ["f_agric", "f_agric_18"]),
        "aridity_index": first_existing(attr_df, ["AI"]),
        "elevation_mean": first_existing(attr_df, ["dem.mean", "dem_mean"]),
        "area_km2": first_existing(attr_df, ["Area_km2"]),
        "qflag": first_existing(attr_df, ["Q_flag"]),
    }

    required = ["agri_frac", "aridity_index", "elevation_mean", "area_km2", "qflag"]
    missing = [k for k in required if cols[k] is None]
    if missing:
        raise KeyError(f"Missing environmental columns for panel (c): {missing}")

    env = attr_df[
        ["OBJECTID", cols["agri_frac"], cols["aridity_index"], cols["elevation_mean"], cols["area_km2"], cols["qflag"]]
    ].copy()

    env.columns = ["OBJECTID", "agri_frac", "aridity_index", "elevation_mean", "area_km2", "Q_flag"]

    for c in ["agri_frac", "aridity_index", "elevation_mean", "area_km2", "Q_flag"]:
        env[c] = pd.to_numeric(env[c], errors="coerce")

    env = env[env["Q_flag"] == 1].copy()
    env = env.dropna(subset=["agri_frac", "aridity_index", "elevation_mean", "area_km2"]).copy()

    env["point_size"] = np.sqrt(env["area_km2"].clip(lower=1)) * 1.8
    return env


# =========================================================
# 4. PANELS
# =========================================================
def plot_panel_a(ax, catchments, rivers, stations):
    c_catch = "#eef1f4"
    c_catch_edge = "#c6ccd3"
    c_river = "#9db9d7"
    c_all = "#b8bec7"
    c_pair = "#2f5d8c"

    catch = to_plot_crs(catchments)
    st = to_plot_crs(stations)

    if rivers is not None:
        rv = to_plot_crs(rivers)
    else:
        rv = None

    # Catchment background
    catch.plot(
        ax=ax,
        facecolor=c_catch,
        edgecolor=c_catch_edge,
        linewidth=0.20,
        alpha=0.95,
        zorder=1,
    )

    # Plot rivers lightly, clipped by the catchment extent
    if rv is not None:
        try:
            xmin, ymin, xmax, ymax = catch.total_bounds
            rv_clip = rv.cx[xmin:xmax, ymin:ymax]
            # Avoid extremely heavy rendering if needed
            rv_clip.plot(
                ax=ax,
                color=c_river,
                linewidth=0.12,
                alpha=0.45,
                zorder=2,
            )
        except Exception:
            pass

    all_st = st.copy()
    paired_st = st[st["Q_flag"] == 1].copy()

    all_st.plot(
        ax=ax,
        color=c_all,
        markersize=5,
        alpha=0.55,
        zorder=3,
    )
    paired_st.plot(
        ax=ax,
        color=c_pair,
        markersize=10,
        alpha=0.90,
        zorder=4,
    )

    xmin, ymin, xmax, ymax = catch.total_bounds
    dx = (xmax - xmin) * 0.03
    dy = (ymax - ymin) * 0.03
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)

    ax.set_title("Study domain and monitoring network", pad=8)
    add_panel_label(ax, "(a)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_edgecolor("#777777")

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=c_all,
            markeredgecolor="none",
            markersize=5.5,
            label=f"All water-quality stations (n = {len(all_st)})",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=c_pair,
            markeredgecolor="none",
            markersize=6.5,
            label=f"Paired discharge stations (n = {len(paired_st)})",
        ),
        Line2D(
            [0], [0],
            color=c_river,
            linewidth=1.1,
            label="River network",
        ),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, handletextpad=0.8)

    try:
        scalebar = ScaleBar(
            1,
            units="m",
            location="lower right",
            box_alpha=0.0,
            color="#555555",
            font_properties={"family": "Times New Roman", "size": 8},
        )
        ax.add_artist(scalebar)
    except Exception:
        pass

    # Locator inset: bounding rectangle of study domain
    try:
        axins = inset_axes(ax, width="25%", height="25%", loc="upper right", borderpad=0.8)
        outline = catch.dissolve()
        outline.plot(ax=axins, facecolor="#d9e2ec", edgecolor="#6f7d8a", linewidth=0.5)
        axins.set_xticks([])
        axins.set_yticks([])
        for sp in axins.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)
            sp.set_edgecolor("#777777")
    except Exception:
        pass


def plot_panel_b(ax, sample):
    add_panel_label(ax, "(b)")
    ax.set_title("Constituent-specific analytical sample", pad=8)

    solutes = sample["solute"].astype(str).tolist()
    y = np.arange(len(solutes))[::-1]

    color_map = {
        "NO3N": "#2f7f73",
        "PO4P": "#c48a3a",
        "DOC": "#7b5ea7",
    }

    counts = sample["n_unique_stations"].astype(int).values
    rows = sample["n_rows"].astype(int).values
    ymins = sample["year_min"].astype(int).values
    ymaxs = sample["year_max"].astype(int).values

    max_count = max(counts) * 1.28
    ax.set_xlim(0, max_count)
    ax.set_ylim(-0.8, len(solutes) - 0.2)

    for yi, s, c, r, y0, y1 in zip(y, solutes, counts, rows, ymins, ymaxs):
        color = color_map.get(s, "#4d4d4d")

        ax.hlines(yi, 0, c, color="#d7d7d7", linewidth=5, zorder=1)
        ax.scatter(c, yi, s=130, color=color, edgecolor="white", linewidth=0.8, zorder=3)

        ax.text(
            -max_count * 0.03,
            yi,
            s,
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=color,
        )

        ax.text(
            c + max_count * 0.025,
            yi + 0.11,
            f"{c} stations",
            ha="left",
            va="center",
            fontsize=10,
        )

        ax.text(
            c + max_count * 0.025,
            yi - 0.17,
            f"{y0}–{y1} | {r:,} station-months",
            ha="left",
            va="center",
            fontsize=9.0,
            color="#5f5f5f",
        )

    ax.set_yticks([])
    ax.set_xlabel("Number of stations")
    clean_axes(ax)

    ax.text(
        0.98,
        0.06,
        "Monthly WRTDS outputs\naligned by OBJECTID–Year–Month",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f7", ec="#b0b0b0", lw=0.8),
    )


def plot_panel_c(ax, env):
    add_panel_label(ax, "(c)")
    ax.set_title("Environmental gradient space", pad=8)

    cmap = cmr.get_sub_cmap("cmr.guppy", 0.08, 0.92)

    sc = ax.scatter(
        env["agri_frac"],
        env["aridity_index"],
        c=env["elevation_mean"],
        s=env["point_size"],
        cmap=cmap,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.35,
    )

    ax.set_xlabel("Agricultural fraction")
    ax.set_ylabel("Aridity index")
    clean_axes(ax)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Mean elevation (m)")
    cbar.ax.tick_params(labelsize=8)

    candidate_areas = [50, 500, 5000]
    handles = [
        plt.scatter(
            [],
            [],
            s=np.sqrt(a) * 1.8,
            color="#7d8ca0",
            alpha=0.5,
            edgecolors="white",
            linewidths=0.35,
            label=f"{a} km$^2$",
        )
        for a in candidate_areas
    ]

    leg = ax.legend(
        handles=handles,
        title="Catchment area",
        loc="lower right",
        frameon=False,
        borderpad=0.2,
        labelspacing=0.7,
    )
    if leg.get_title():
        leg.get_title().set_fontfamily("Times New Roman")

    ax.text(
        0.02,
        0.02,
        "Paired discharge stations only",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        color="#666666",
    )


# =========================================================
# 5. MAIN
# =========================================================
def make_figure():
    catchments, rivers, stations, attr_df = load_gis_layers()
    sample = get_sample_summary()
    env = prepare_environment_data(attr_df)

    fig = plt.figure(figsize=(14.3, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.95, 1.10], wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    plot_panel_a(ax1, catchments, rivers, stations)
    plot_panel_b(ax2, sample)
    plot_panel_c(ax3, env)

    fig.savefig(OUT_TIFF, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Fig. 2 generated successfully ===")
    print(f"TIFF: {OUT_TIFF}")
    print(f"PDF : {OUT_PDF}")
    print(f"PNG : {OUT_PNG}")
    print(f"Join report: {OUT_JOIN_REPORT}")


if __name__ == "__main__":
    make_figure()