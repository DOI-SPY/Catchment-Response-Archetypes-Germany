from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = PROJECT_ROOT / "data_final"

INPUT_FILE = FINAL_DIR / "analysis_panel_station_month_main.csv"
OUTPUT_FILE = FINAL_DIR / "analysis_panel_station_month_states.csv"


KEY_COLS = ["OBJECTID", "Year", "Month"]
HYDROMET_COLS = ["mean_Q", "pre", "pet"]


def read_csv_fallback(path: Path, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file using several possible encodings.

    Some QUADICA metadata files contain German characters and may not be
    consistently encoded as UTF-8.
    """
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_error = e

    raise RuntimeError(
        f"Could not read file {path}. Tried encodings: {encodings}. "
        f"Last error: {last_error}"
    )


def _check_required_columns(df: pd.DataFrame) -> None:
    required = set(KEY_COLS + HYDROMET_COLS)
    missing = sorted(required.difference(df.columns))

    if missing:
        raise ValueError(
            "Input panel is missing required columns for state and CMD "
            f"classification: {missing}"
        )


def _build_unique_hydromet_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a unique OBJECTID-Year-Month hydrometeorological panel.

    The input analysis panel is expected to be solute-expanded, meaning that
    the same OBJECTID-Year-Month may appear multiple times for different
    solutes. Hydrological states and antecedent CMD must NOT be computed on
    that expanded panel. They must be computed once per catchment-month and
    then merged back to the solute-expanded panel.

    If duplicate hydrometeorological values exist for the same
    OBJECTID-Year-Month, the median is used. A warning is printed if duplicate
    rows disagree numerically.
    """
    work = df[KEY_COLS + HYDROMET_COLS].copy()

    for col in ["Year", "Month"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    for col in HYDROMET_COLS:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["OBJECTID", "Year", "Month"]).copy()
    work["Year"] = work["Year"].astype(int)
    work["Month"] = work["Month"].astype(int)

    # Diagnostic: do duplicate solute rows have identical hydromet values?
    ranges = (
        work.groupby(KEY_COLS)[HYDROMET_COLS]
        .agg(lambda s: s.max() - s.min())
    )

    bad_duplicates = ranges.gt(1e-9).any(axis=1)

    if bad_duplicates.any():
        n_bad = int(bad_duplicates.sum())
        print(
            "\n[WARNING] Some duplicate OBJECTID-Year-Month rows have "
            "non-identical hydrometeorological values. The median will be "
            f"used for state classification. Number of affected months: {n_bad}"
        )

        print("\n[Example duplicate hydromet disagreements]")
        print(ranges.loc[bad_duplicates].head())

    # Collapse solute-expanded rows to one hydromet row per OBJECTID-Year-Month.
    hyd = (
        work.groupby(KEY_COLS, as_index=False)
        .agg(
            mean_Q=("mean_Q", "median"),
            pre=("pre", "median"),
            pet=("pet", "median"),
        )
    )

    hyd["_date"] = pd.to_datetime(
        {
            "year": hyd["Year"],
            "month": hyd["Month"],
            "day": 1,
        }
    )

    hyd["_has_original_panel_month"] = True

    # Reindex each OBJECTID to a complete monthly calendar.
    # This makes shift(1) and shift(2) true calendar-month lags rather than
    # previous available-row lags.
    completed_groups: list[pd.DataFrame] = []

    for object_id, g in hyd.groupby("OBJECTID", sort=False):
        g = g.sort_values("_date").drop_duplicates("_date").copy()

        if g["_date"].isna().all():
            continue

        full_index = pd.date_range(
            start=g["_date"].min(),
            end=g["_date"].max(),
            freq="MS",
        )

        g_full = (
            g.set_index("_date")
            .reindex(full_index)
            .rename_axis("_date")
            .reset_index()
        )

        g_full["OBJECTID"] = object_id
        g_full["Year"] = g_full["_date"].dt.year.astype(int)
        g_full["Month"] = g_full["_date"].dt.month.astype(int)

        g_full["_has_original_panel_month"] = (
            g_full["_has_original_panel_month"]
            .eq(True)
            .astype(bool)
        )

        completed_groups.append(g_full)

    if not completed_groups:
        raise RuntimeError("No valid hydrometeorological monthly panel could be built.")

    hyd_full = pd.concat(completed_groups, ignore_index=True)

    return hyd_full


def _antecedent_rolling_sum(
    hyd: pd.DataFrame,
    value_col: str,
    window: int,
) -> pd.Series:
    """
    Compute antecedent rolling sums on a complete monthly calendar.

    For month t, the returned value is:

        sum_{l=1}^{window} x_{t-l}

    The current month is intentionally excluded.
    """

    return (
        hyd.groupby("OBJECTID", group_keys=False)[value_col]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=window).sum())
    )


def _assign_states_on_unique_hydromet_panel(hyd_full: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CMD and hydrological states on the unique monthly hydromet panel.
    """
    hyd = hyd_full.copy()
    hyd = hyd.sort_values(["OBJECTID", "_date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1. Monthly climatic water deficit
    # ------------------------------------------------------------------
    hyd["water_deficit_mm"] = np.where(
        hyd["pre"].notna() & hyd["pet"].notna(),
        np.maximum(0.0, hyd["pet"] - hyd["pre"]),
        np.nan,
    )

    # ------------------------------------------------------------------
    # 2. Antecedent cumulative moisture deficit
    # ------------------------------------------------------------------
    for window in [3, 6]:
        cmd_abs_col = f"CMD_{window}m_mm"
        pet_ant_col = f"PET_ant_{window}m_mm"
        cmd_norm_col = f"CMD_{window}m_norm"

        hyd[cmd_abs_col] = _antecedent_rolling_sum(
            hyd,
            value_col="water_deficit_mm",
            window=window,
        )

        hyd[pet_ant_col] = _antecedent_rolling_sum(
            hyd,
            value_col="pet",
            window=window,
        )

        hyd[cmd_norm_col] = np.where(
            hyd[pet_ant_col] > 0,
            hyd[cmd_abs_col] / hyd[pet_ant_col],
            np.nan,
        )

        # Backward-compatible compact name.
        hyd[f"CMD_{window}m"] = hyd[cmd_norm_col]

    # ------------------------------------------------------------------
    # 3. Catchment- and calendar-month-specific discharge thresholds
    # ------------------------------------------------------------------
    thresholds = (
        hyd.dropna(subset=["mean_Q"])
        .groupby(["OBJECTID", "Month"])["mean_Q"]
        .agg(
            q20=lambda x: x.quantile(0.20),
            q50=lambda x: x.quantile(0.50),
            q80=lambda x: x.quantile(0.80),
        )
        .reset_index()
    )

    hyd = hyd.merge(thresholds, on=["OBJECTID", "Month"], how="left")

    # ------------------------------------------------------------------
    # 4. Hydrological-state classification
    # ------------------------------------------------------------------
    hyd["is_drought_like"] = (
        hyd["mean_Q"].notna()
        & hyd["q20"].notna()
        & (hyd["mean_Q"] <= hyd["q20"])
    )

    hyd["is_highflow_candidate"] = (
        hyd["mean_Q"].notna()
        & hyd["q80"].notna()
        & (hyd["mean_Q"] >= hyd["q80"])
    )

    hyd["drought_lag1"] = (
        hyd.groupby("OBJECTID")["is_drought_like"]
        .shift(1, fill_value=False)
        .astype(bool)
    )

    hyd["drought_lag2"] = (
        hyd.groupby("OBJECTID")["is_drought_like"]
        .shift(2, fill_value=False)
        .astype(bool)
    )

    hyd["is_rewetting"] = (
        (hyd["drought_lag1"] | hyd["drought_lag2"])
        & hyd["mean_Q"].notna()
        & hyd["q50"].notna()
        & (hyd["mean_Q"] > hyd["q50"])
    )

    conditions = [
        hyd["is_drought_like"],
        hyd["is_rewetting"] & ~hyd["is_drought_like"],
        (
            hyd["is_highflow_candidate"]
            & ~hyd["is_rewetting"]
            & ~hyd["is_drought_like"]
        ),
    ]

    choices = [
        "drought_like",
        "post_drought_rewetting",
        "non_drought_highflow",
    ]

    hyd["hydro_state"] = np.select(
        conditions,
        choices,
        default="normal",
    )

    hyd.loc[hyd["mean_Q"].isna(), "hydro_state"] = pd.NA

    return hyd


def assign_states_and_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign monthly hydrological states and compute antecedent CMD metrics.

    Critical design choice:
        The input analysis panel may be solute-expanded. Hydrological states
        and CMD are therefore computed on a unique OBJECTID-Year-Month
        hydrometeorological panel and then merged back to the solute-expanded
        panel.

    CMD definitions:
        water_deficit_mm = max(0, PET - precipitation)

        CMD_3m_mm:
            sum of water_deficit_mm over the previous 3 calendar months

        CMD_3m_norm:
            CMD_3m_mm divided by the sum of PET over the previous 3 calendar
            months

        CMD_3m:
            alias for CMD_3m_norm, retained for backward compatibility.

    Hydrological states:
        drought_like:
            monthly mean discharge <= catchment- and calendar-month-specific Q20

        post_drought_rewetting:
            monthly mean discharge > seasonal median and at least one of the
            previous two calendar months was drought_like

        non_drought_highflow:
            monthly mean discharge >= seasonal Q80, provided the month is not
            drought_like or post_drought_rewetting

        normal:
            all remaining months
    """
    _check_required_columns(df)

    print(
        "Computing hydrological states and dimensionally consistent "
        "antecedent CMD metrics on unique OBJECTID-Year-Month hydromet panel..."
    )

    out = df.copy()

    for col in ["Year", "Month"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["OBJECTID", "Year", "Month"]).copy()
    out["Year"] = out["Year"].astype(int)
    out["Month"] = out["Month"].astype(int)

    hyd_full = _build_unique_hydromet_panel(out)
    hyd_states = _assign_states_on_unique_hydromet_panel(hyd_full)

    state_cols = [
        "water_deficit_mm",
        "CMD_3m_mm",
        "PET_ant_3m_mm",
        "CMD_3m_norm",
        "CMD_3m",
        "CMD_6m_mm",
        "PET_ant_6m_mm",
        "CMD_6m_norm",
        "CMD_6m",
        "q20",
        "q50",
        "q80",
        "is_drought_like",
        "is_highflow_candidate",
        "drought_lag1",
        "drought_lag2",
        "is_rewetting",
        "hydro_state",
    ]

    merge_panel = hyd_states[KEY_COLS + state_cols].drop_duplicates(KEY_COLS)

    # Remove old state/CMD columns if rerunning on a previously processed file.
    old_cols = [c for c in state_cols if c in out.columns]
    if old_cols:
        out = out.drop(columns=old_cols)

    out = out.merge(
        merge_panel,
        on=KEY_COLS,
        how="left",
        validate="many_to_one",
    )

    # Validation: every OBJECTID-Year-Month must have only one state after merge.
    conflict_check = (
        out.groupby(KEY_COLS)["hydro_state"]
        .nunique(dropna=False)
    )

    max_states_per_month = int(conflict_check.max()) if not conflict_check.empty else 0

    if max_states_per_month > 1:
        raise RuntimeError(
            "State assignment conflict detected: at least one "
            "OBJECTID-Year-Month has multiple hydro_state values after merge."
        )

    return out


def _print_state_summary(df_states: pd.DataFrame) -> None:
    """
    Print both unique-month and solute-expanded state summaries.
    """
    unique_months = df_states.drop_duplicates(KEY_COLS).copy()

    print("\n[Hydrological-state distribution: unique OBJECTID-Year-Month]")
    unique_counts = unique_months["hydro_state"].value_counts(dropna=False)
    unique_freq = unique_months["hydro_state"].value_counts(
        normalize=True,
        dropna=False,
    )

    unique_summary = pd.DataFrame(
        {
            "n_unique_months": unique_counts,
            "fraction": unique_freq,
        }
    )

    print(unique_summary)

    print("\n[Hydrological-state distribution: solute-station-month long panel]")
    long_counts = df_states["hydro_state"].value_counts(dropna=False)
    long_freq = df_states["hydro_state"].value_counts(
        normalize=True,
        dropna=False,
    )

    long_summary = pd.DataFrame(
        {
            "n_solute_station_months": long_counts,
            "fraction": long_freq,
        }
    )

    print(long_summary)


def _print_cmd_summary(df_states: pd.DataFrame) -> None:
    """
    Print CMD diagnostics on unique months to avoid solute-duplicate weighting.
    """
    unique_months = df_states.drop_duplicates(KEY_COLS).copy()

    cmd_cols = [
        "CMD_3m_mm",
        "CMD_3m_norm",
        "CMD_6m_mm",
        "CMD_6m_norm",
    ]

    available_cmd_cols = [c for c in cmd_cols if c in unique_months.columns]

    print("\n[CMD diagnostic summary: unique OBJECTID-Year-Month]")
    print(
        unique_months[available_cmd_cols]
        .describe(percentiles=[0.25, 0.5, 0.75])
    )

    print("\n[CMD missing values: unique OBJECTID-Year-Month]")
    print(unique_months[available_cmd_cols].isna().sum())


def _print_duplicate_state_validation(df_states: pd.DataFrame) -> None:
    """
    Confirm that duplicate solute rows for the same catchment-month share the
    same hydrological state and CMD values.
    """
    check_cols = [
        "hydro_state",
        "CMD_3m_mm",
        "CMD_3m_norm",
        "CMD_6m_mm",
        "CMD_6m_norm",
    ]

    available_cols = [c for c in check_cols if c in df_states.columns]

    print("\n[Duplicate OBJECTID-Year-Month validation]")
    for col in available_cols:
        max_nunique = (
            df_states.groupby(KEY_COLS)[col]
            .nunique(dropna=False)
            .max()
        )

        print(f"{col}: max unique values within OBJECTID-Year-Month = {max_nunique}")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find input panel: {INPUT_FILE}. "
            "Please run 01_build_monthly_panel.py first."
        )

    df = read_csv_fallback(INPUT_FILE, low_memory=False)
    df_states = assign_states_and_memory(df)

    df_states.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    _print_state_summary(df_states)
    _print_cmd_summary(df_states)
    _print_duplicate_state_validation(df_states)

    print(f"\n[Completed] Output saved to: {OUTPUT_FILE}")
    print(
        "Note: CMD_3m and CMD_6m are dimensionless normalized antecedent "
        "moisture-deficit metrics in the revised workflow."
    )
    print(
        "Hydrological states and CMD were computed on unique "
        "OBJECTID-Year-Month rows and then merged back to the solute-expanded "
        "analysis panel."
    )


if __name__ == "__main__":
    main()