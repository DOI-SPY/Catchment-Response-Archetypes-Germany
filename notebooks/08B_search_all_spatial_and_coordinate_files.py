from pathlib import Path
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data_raw" / "QUADICA_v2"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_REPORT = REPORT_DIR / "spatial_coordinate_file_search_report.csv"

COORD_KEYWORDS = [
    "lon", "long", "longitude", "lng",
    "lat", "latitude",
    "x", "y",
    "coord", "coordinate",
    "easting", "northing",
    "utm", "epsg",
    "geometry", "geom",
    "station", "site", "gauge"
]

VECTOR_EXTS = [".shp", ".gpkg", ".geojson", ".json"]
TABLE_EXTS = [".csv", ".txt", ".xlsx", ".xls"]


def read_table_preview(path: Path, nrows=5):
    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, nrows=nrows), "excel"

    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            if suffix == ".csv":
                return pd.read_csv(path, encoding=enc, low_memory=False, nrows=nrows), enc
            elif suffix == ".txt":
                return pd.read_csv(path, sep=None, engine="python", encoding=enc, nrows=nrows), enc
        except Exception:
            pass

    raise RuntimeError("Could not read table preview")


def score_columns(columns):
    cols = [str(c) for c in columns]
    lower_cols = [c.lower() for c in cols]
    hits = []
    for c, lc in zip(cols, lower_cols):
        for kw in COORD_KEYWORDS:
            if kw in lc:
                hits.append(c)
                break
    return hits


def main():
    rows = []

    print(f"Searching RAW_DIR = {RAW_DIR}")

    files = [p for p in RAW_DIR.rglob("*") if p.is_file()]
    print(f"Total files found: {len(files)}")

    for p in files:
        suffix = p.suffix.lower()
        rel = str(p.relative_to(PROJECT_ROOT))

        if suffix in VECTOR_EXTS:
            try:
                gdf = gpd.read_file(p)
                geom_types = "; ".join(gdf.geometry.geom_type.dropna().unique().tolist())
                rows.append({
                    "relative_path": rel,
                    "file_type": "vector",
                    "suffix": suffix,
                    "read_status": "OK",
                    "shape": str(gdf.shape),
                    "crs": str(gdf.crs),
                    "geometry_types": geom_types,
                    "candidate_columns": " | ".join(score_columns(gdf.columns)),
                    "all_columns_preview": " | ".join([str(c) for c in gdf.columns[:30]])
                })
                print(f"[VECTOR OK] {rel} | {gdf.shape} | {geom_types}")
            except Exception as e:
                rows.append({
                    "relative_path": rel,
                    "file_type": "vector",
                    "suffix": suffix,
                    "read_status": f"ERROR: {e}",
                    "shape": "",
                    "crs": "",
                    "geometry_types": "",
                    "candidate_columns": "",
                    "all_columns_preview": ""
                })
                print(f"[VECTOR ERROR] {rel} | {e}")

        elif suffix in TABLE_EXTS:
            try:
                df, enc = read_table_preview(p, nrows=5)
                hits = score_columns(df.columns)
                # Only keep files with possible coordinate/station fields or known small metadata files
                keep = bool(hits) or ("meta" in p.name.lower()) or ("station" in p.name.lower()) or ("site" in p.name.lower())
                if keep:
                    rows.append({
                        "relative_path": rel,
                        "file_type": "table",
                        "suffix": suffix,
                        "read_status": f"OK ({enc})",
                        "shape": f"preview_rows={df.shape[0]}, n_cols={df.shape[1]}",
                        "crs": "",
                        "geometry_types": "",
                        "candidate_columns": " | ".join(hits),
                        "all_columns_preview": " | ".join([str(c) for c in df.columns[:40]])
                    })
                    print(f"[TABLE CANDIDATE] {rel} | hits={hits}")
            except Exception:
                pass

    report = pd.DataFrame(rows)
    report.to_csv(OUT_REPORT, index=False, encoding="utf-8-sig")

    print("\n=== Search finished ===")
    print(f"Report saved to: {OUT_REPORT}")
    if len(report) > 0:
        print(report[["relative_path", "file_type", "read_status", "candidate_columns"]].to_string(index=False))
    else:
        print("No spatial or coordinate-like files found.")


if __name__ == "__main__":
    main()