"""Import vessel GPS track data into SQLite.

This script reads exactly two CSV files containing UTC GPS observations,
combines them, assigns survey IDs with continuity across incremental imports,
and appends the resulting rows to a SQLite database.

Example run:

    python3 metadatabase/import_tracks.py \
        --platform SAILOR2 \
        --interpolation-seconds 60 \
        /home/jonas/Documents/vscode/echoedge/temp/temp_pos/SB2530A.txt \
        /home/jonas/Documents/vscode/echoedge/temp/temp_pos/SB2530D.txt \
        /home/jonas/Documents/vscode/echoedge/metadatabase/sailbuoy_metadatabase.db

To skip interpolation entirely, pass ``--interpolation-seconds 0``.
To keep a coarse track, the default is one row per minute.
Do not leave any spaces after a line-continuation backslash.

The CSV files must contain the columns: time, lat, long.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SURVEY_GAP_DAYS = 3
DEFAULT_INTERPOLATION_SECONDS = 60

TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS track_points (
    id INTEGER PRIMARY KEY,
    platform TEXT,
    survey_id TEXT,
    timestamp_utc TEXT,
    latitude REAL,
    longitude REAL,
    distance_m REAL,
    speed_ms REAL,
    is_interpolated INTEGER,
    source_file TEXT,
    created_at TEXT,
    UNIQUE(platform, timestamp_utc)
)
"""


def create_database(db_path: Path) -> sqlite3.Connection:
    """Open the SQLite database and create the target table if needed."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute(TABLE_SCHEMA)
    connection.commit()
    return connection


def utc_now_string() -> str:
    """Return the current UTC time as an ISO-8601 string with a Z suffix."""

    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    """Parse timestamps as timezone-aware UTC values."""

    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.dt.tz_convert("UTC")


def format_timestamp_utc(series: pd.Series) -> pd.Series:
    """Format timezone-aware timestamps for SQLite storage."""

    return series.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_utc_timestamp(value: object) -> pd.Timestamp:
    """Return a timezone-aware UTC timestamp for any timestamp-like value."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def load_track_csv(csv_path: Path) -> pd.DataFrame:
    """Load a GPS track file with automatic delimiter detection.

    The Sailor/Sailbuoy exports are often tab-delimited with capitalized
    column names, so we normalize headers before validating the required
    fields.
    """

    frame = pd.read_csv(csv_path, sep=None, engine="python")
    frame = frame.rename(columns={column: column.strip().lower() for column in frame.columns})
    frame = frame.loc[:, ~frame.columns.duplicated()]
    return frame


def read_csvs(csv_paths: Sequence[Path], platform: str) -> pd.DataFrame:
    """Read the two CSV inputs and combine them into a single dataframe."""

    frames: List[pd.DataFrame] = []
    for csv_path in csv_paths:
        frame = load_track_csv(csv_path)
        missing_columns = {"time", "lat", "long"} - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing_columns)}")

        frame = frame.rename(columns={"time": "timestamp_utc", "lat": "latitude", "long": "longitude"})
        frame["source_file"] = csv_path.name
        frame["platform"] = platform
        frame["timestamp_utc"] = parse_timestamp_series(frame["timestamp_utc"])
        frame = frame.dropna(subset=["timestamp_utc", "latitude", "longitude"])
        frames.append(frame[["platform", "timestamp_utc", "latitude", "longitude", "source_file"]])

    if not frames:
        return pd.DataFrame(columns=["platform", "timestamp_utc", "latitude", "longitude", "source_file"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("timestamp_utc").reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["timestamp_utc"], keep="first").reset_index(drop=True)
    return combined


def load_existing_state(connection: sqlite3.Connection, platform: str) -> Dict[str, object]:
    """Load existing timestamps, the most recent row, and survey sequence state."""

    existing_df = pd.read_sql_query(
        "SELECT timestamp_utc, survey_id, latitude, longitude FROM track_points WHERE platform = ?",
        connection,
        params=(platform,),
    )

    if existing_df.empty:
        return {
            "timestamps": set(),
            "latest_row": None,
            "max_sequence_by_year": {},
        }

    existing_df["timestamp_utc"] = pd.to_datetime(existing_df["timestamp_utc"], utc=True, errors="coerce")
    existing_df = existing_df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")

    latest_row = existing_df.iloc[-1].to_dict()
    latest_row["timestamp_utc"] = ensure_utc_timestamp(latest_row["timestamp_utc"])

    max_sequence_by_year: Dict[int, int] = {}
    surveys = pd.read_sql_query(
        "SELECT DISTINCT survey_id FROM track_points WHERE platform = ?",
        connection,
        params=(platform,),
    )
    for survey_id in surveys["survey_id"].dropna():
        try:
            _, year_str, sequence_str = survey_id.rsplit("_", 2)
            year = int(year_str)
            sequence = int(sequence_str)
        except ValueError:
            continue
        max_sequence_by_year[year] = max(max_sequence_by_year.get(year, 0), sequence)

    return {
        "timestamps": set(format_timestamp_utc(existing_df["timestamp_utc"])),
        "latest_row": latest_row,
        "max_sequence_by_year": max_sequence_by_year,
    }


def remove_existing_duplicates(df: pd.DataFrame, existing_timestamps: set[str]) -> pd.DataFrame:
    """Drop rows already present in the database using the platform/timestamp key."""

    if df.empty or not existing_timestamps:
        return df.copy()

    timestamp_strings = format_timestamp_utc(df["timestamp_utc"])
    mask = ~timestamp_strings.isin(existing_timestamps)
    return df.loc[mask].copy().reset_index(drop=True)


def next_survey_id(platform: str, year: int, next_sequence_by_year: Dict[int, int]) -> str:
    """Generate the next survey ID for a given platform and survey year."""

    sequence = next_sequence_by_year.get(year, 0) + 1
    next_sequence_by_year[year] = sequence
    return f"{platform}_{year}_{sequence:03d}"


def assign_surveys(
    df: pd.DataFrame,
    platform: str,
    existing_state: Dict[str, object],
    survey_gap_days: int = SURVEY_GAP_DAYS,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Assign survey IDs to new rows and return the anchor row, if continuity applies."""

    if df.empty:
        empty = df.copy()
        empty["survey_id"] = pd.Series(dtype="object")
        empty["is_interpolated"] = pd.Series(dtype="int64")
        return empty, None

    assigned = df.sort_values("timestamp_utc").reset_index(drop=True).copy()
    assigned["survey_id"] = None
    assigned["is_interpolated"] = 0

    next_sequence_by_year = dict(existing_state["max_sequence_by_year"])
    gap_threshold = pd.Timedelta(days=survey_gap_days)
    latest_row = existing_state["latest_row"]
    anchor_row: Optional[pd.Series] = None

    for index, row in assigned.iterrows():
        timestamp = row["timestamp_utc"]
        year = int(timestamp.year)

        if index == 0:
            continue_survey = False
            if latest_row is not None:
                latest_timestamp = ensure_utc_timestamp(latest_row["timestamp_utc"])
                latest_survey_id = latest_row["survey_id"]
                if pd.notna(latest_timestamp) and timestamp > latest_timestamp and timestamp - latest_timestamp <= gap_threshold:
                    continue_survey = True
                    assigned.at[index, "survey_id"] = latest_survey_id
                    anchor_row = pd.Series(
                        {
                            "platform": platform,
                            "survey_id": latest_survey_id,
                            "timestamp_utc": latest_timestamp,
                            "latitude": float(latest_row["latitude"]),
                            "longitude": float(latest_row["longitude"]),
                            "source_file": "database_anchor",
                            "is_interpolated": 0,
                        }
                    )

            if not continue_survey:
                assigned.at[index, "survey_id"] = next_survey_id(platform, year, next_sequence_by_year)
            continue

        previous_timestamp = assigned.at[index - 1, "timestamp_utc"]
        if timestamp - previous_timestamp > gap_threshold:
            assigned.at[index, "survey_id"] = next_survey_id(platform, year, next_sequence_by_year)
        else:
            assigned.at[index, "survey_id"] = assigned.at[index - 1, "survey_id"]

    return assigned, anchor_row


def interpolate_group_at_interval(
    group: pd.DataFrame,
    interpolation_seconds: int,
    anchor_row: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Interpolate a single survey at a configurable interval.

    Pass ``interpolation_seconds=0`` to skip interpolation and keep the original
    observations only.
    """

    working = group.sort_values("timestamp_utc").reset_index(drop=True).copy()
    if anchor_row is not None:
        anchor_df = pd.DataFrame([anchor_row])
        working = pd.concat([anchor_df, working], ignore_index=True)

    working = working.drop_duplicates(subset=["timestamp_utc"], keep="first").sort_values("timestamp_utc")
    working = working.set_index("timestamp_utc")

    if interpolation_seconds <= 0:
        interpolated = working.copy()
    else:
        full_range = pd.date_range(
            working.index.min(),
            working.index.max(),
            freq=pd.to_timedelta(interpolation_seconds, unit="s"),
            tz="UTC",
        )
        interpolated = working.reindex(full_range)

    original_timestamps = set(group["timestamp_utc"])
    if anchor_row is not None:
        original_timestamps.add(pd.Timestamp(anchor_row["timestamp_utc"]).tz_convert("UTC"))

    interpolated["latitude"] = interpolated["latitude"].astype(float).interpolate(method="linear")
    interpolated["longitude"] = interpolated["longitude"].astype(float).interpolate(method="linear")

    interpolated["platform"] = group["platform"].iloc[0]
    interpolated["survey_id"] = group["survey_id"].iloc[0]

    if "source_file" in interpolated.columns:
        interpolated["source_file"] = interpolated["source_file"].ffill().bfill().fillna("interpolated")
    else:
        interpolated["source_file"] = "interpolated"

    interpolated["is_interpolated"] = (~interpolated.index.isin(original_timestamps)).astype(int)
    interpolated = interpolated.reset_index().rename(columns={"index": "timestamp_utc"})
    interpolated["timestamp_utc"] = pd.to_datetime(interpolated["timestamp_utc"], utc=True)
    interpolated.loc[interpolated["is_interpolated"] == 1, "source_file"] = "interpolated"

    return interpolated.reset_index(drop=True)


def calculate_distance_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate great-circle distance and speed within each survey."""

    result = df.sort_values(["survey_id", "timestamp_utc"]).reset_index(drop=True).copy()
    result["distance_m"] = np.nan
    result["speed_ms"] = np.nan

    for survey_id, group_index in result.groupby("survey_id").groups.items():
        indices = list(group_index)
        previous_index = None
        for current_index in indices:
            if previous_index is None:
                previous_index = current_index
                continue

            lat1 = float(result.at[previous_index, "latitude"])
            lon1 = float(result.at[previous_index, "longitude"])
            lat2 = float(result.at[current_index, "latitude"])
            lon2 = float(result.at[current_index, "longitude"])

            distance_m = haversine_distance_m(lat1, lon1, lat2, lon2)
            elapsed_seconds = (
                pd.Timestamp(result.at[current_index, "timestamp_utc"])
                - pd.Timestamp(result.at[previous_index, "timestamp_utc"])
            ).total_seconds()
            speed_ms = distance_m / elapsed_seconds if elapsed_seconds > 0 else np.nan

            result.at[current_index, "distance_m"] = distance_m
            result.at[current_index, "speed_ms"] = speed_ms
            previous_index = current_index

    return result


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two WGS84 coordinates in meters."""

    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_m * c


def write_to_database(connection: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Append rows to SQLite using bulk insertion and ignore duplicates safely."""

    if df.empty:
        return 0

    insert_df = df.copy()
    insert_df["timestamp_utc"] = format_timestamp_utc(insert_df["timestamp_utc"])
    insert_df["created_at"] = utc_now_string()

    rows = list(
        insert_df[
            [
                "platform",
                "survey_id",
                "timestamp_utc",
                "latitude",
                "longitude",
                "distance_m",
                "speed_ms",
                "is_interpolated",
                "source_file",
                "created_at",
            ]
        ].itertuples(index=False, name=None)
    )

    sql = """
    INSERT OR IGNORE INTO track_points (
        platform,
        survey_id,
        timestamp_utc,
        latitude,
        longitude,
        distance_m,
        speed_ms,
        is_interpolated,
        source_file,
        created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    with connection:
        connection.executemany(sql, rows)

    return len(rows)


def build_output_frame(
    new_rows: pd.DataFrame,
    anchor_row: Optional[pd.Series],
    interpolation_seconds: int,
) -> pd.DataFrame:
    """Interpolate each survey and produce the final insert dataframe."""

    if new_rows.empty:
        return new_rows.copy()

    outputs: List[pd.DataFrame] = []
    anchor_timestamp: Optional[pd.Timestamp] = None
    for survey_id, group in new_rows.groupby("survey_id", sort=False):
        survey_anchor = anchor_row if anchor_row is not None and group.index.min() == 0 else None
        if survey_anchor is not None:
            anchor_timestamp = pd.Timestamp(survey_anchor["timestamp_utc"]).tz_convert("UTC")
        interpolated_group = interpolate_group_at_interval(
            group,
            interpolation_seconds=interpolation_seconds,
            anchor_row=survey_anchor,
        )
        outputs.append(interpolated_group)

    if not outputs:
        return pd.DataFrame(columns=list(new_rows.columns) + ["distance_m", "speed_ms"])

    final_df = pd.concat(outputs, ignore_index=True)
    final_df = calculate_distance_speed(final_df)
    if anchor_timestamp is not None:
        final_df = final_df.loc[final_df["timestamp_utc"] != anchor_timestamp].copy()
    final_df = final_df.sort_values("timestamp_utc").reset_index(drop=True)
    return final_df


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Import vessel GPS tracks into SQLite.")
    parser.add_argument("--platform", required=True, help="Platform name for all imported rows.")
    parser.add_argument(
        "csv_files",
        nargs=2,
        type=Path,
        help="Exactly two CSV files containing time, lat, long columns.",
    )
    parser.add_argument("database", type=Path, help="Target SQLite database file.")
    parser.add_argument(
        "--survey-gap-days",
        type=int,
        default=SURVEY_GAP_DAYS,
        help="Gap in days that starts a new survey.",
    )
    parser.add_argument(
        "--interpolation-seconds",
        type=int,
        default=DEFAULT_INTERPOLATION_SECONDS,
        help="Interpolation interval in seconds; use 0 to skip interpolation.",
    )

    args = parser.parse_args(argv)

    csv_paths = args.csv_files
    platform = args.platform
    database_path = args.database

    print(f"Reading CSV files for platform {platform!r}...")
    incoming = read_csvs(csv_paths, platform)
    print(f"Loaded {len(incoming)} combined rows before duplicate filtering.")

    connection = create_database(database_path)
    try:
        existing_state = load_existing_state(connection, platform)

        filtered = remove_existing_duplicates(incoming, existing_state["timestamps"])
        removed = len(incoming) - len(filtered)
        print(f"Skipped {removed} rows already present in the database.")

        if filtered.empty:
            print("No new track points to import.")
            return 0

        assigned, anchor_row = assign_surveys(filtered, platform, existing_state, args.survey_gap_days)
        if anchor_row is not None:
            print("Continuing the latest existing survey from the database.")

        prepared = build_output_frame(assigned, anchor_row, args.interpolation_seconds)
        if prepared.empty:
            print("No rows remained after interpolation.")
            return 0

        inserted_rows = write_to_database(connection, prepared)
        print(f"Prepared {len(prepared)} rows and attempted to insert {inserted_rows} rows into {database_path}.")
        print("Import completed successfully.")
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())