# Copyright (c) 2026 Rajeev Atla
"""Download weight data from Google Sheets and create a line chart."""

from __future__ import annotations

import io
import sys
from datetime import date
from pathlib import Path
from urllib.request import urlopen

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl

# Use a non-interactive backend so chart generation also works in CI.
mpl.use("Agg")

SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1qon9wSJhz9pmLyybgV68wkzmHTtT4hyc1cWcb56KOfs/export?format=csv&gid=0"
)
OUTPUT_PATH = Path(__file__).with_name("weight_over_time.png")


def parse_dates(values: list[object]) -> list[date | None]:
    """Parse full and yearless sheet dates in their source order.

    Args:
        values: Raw date cell values from the sheet.

    Returns:
        Parsed dates. Invalid values and yearless values before the first full
        date are represented by ``None``.
    """
    dates: list[date | None] = []
    previous_date: date | None = None
    anchor_year: int | None = None

    # Google Sheets exports some cells in their display format, which omits
    # the year. The first full date anchors the year inference below.
    for value in values:
        text = "" if value is None else str(value).strip()
        if not text:
            dates.append(None)
            continue

        try:
            month, day, year = text.split("/")
            parsed_date = date(int(year), int(month), int(day))
            anchor_year = anchor_year or parsed_date.year
        except ValueError:
            try:
                # 2000 validates month/day values, including February 29.
                month, day = text.split("/")
                month_day = date(2000, int(month), int(day))
            except ValueError:
                dates.append(None)
                continue

            if anchor_year is None or previous_date is None:
                dates.append(None)
                continue

            year = previous_date.year
            parsed_date = date(year, month_day.month, month_day.day)
            if parsed_date < previous_date:
                parsed_date = date(year + 1, month_day.month, month_day.day)

        dates.append(parsed_date)
        previous_date = parsed_date

    return dates


def load_weight_data() -> pl.DataFrame:
    """Download and clean weight measurements from the configured sheet.

    Returns:
        A DataFrame containing sorted ``date`` and ``weight_lbs`` columns.

    Raises:
        RuntimeError: If the sheet cannot be downloaded or parsed as CSV.
        ValueError: If required columns or valid measurements are missing.
    """
    try:
        # The sheet is intentionally read through its public CSV export, so no
        # credentials or extra API dependency are needed.
        with urlopen(SHEET_CSV_URL, timeout=30) as response:
            csv_bytes = response.read()
    except Exception as error:
        message = f"Could not download the Google Sheet: {error}"
        raise RuntimeError(message) from error

    try:
        raw = pl.read_csv(io.BytesIO(csv_bytes), infer_schema_length=0)
    except Exception as error:
        message = f"Could not read the sheet CSV: {error}"
        raise RuntimeError(message) from error

    required_columns = {"Date", "Weight (lbs)"}
    missing_columns = required_columns - set(raw.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        message = f"The sheet is missing required column(s): {missing}"
        raise ValueError(message)

    selected = raw.select(
        pl.col("Date").cast(pl.String).str.strip_chars().alias("date_text"),
        pl.col("Weight (lbs)").cast(pl.Float64, strict=False).alias("weight_lbs"),
    )
    cleaned = (
        selected.with_columns(
            pl.Series(
                "date", parse_dates(selected["date_text"].to_list()), dtype=pl.Date
            )
        )
        .select("date", "weight_lbs")
        .drop_nulls()
        .sort("date")
    )

    if cleaned.is_empty():
        message = "No valid Date/Weight (lbs) rows were found in the sheet."
        raise ValueError(message)

    return cleaned


def save_chart(data: pl.DataFrame) -> None:
    """Save a line chart for the supplied weight measurements.

    Args:
        data: A sorted DataFrame with ``date`` and ``weight_lbs`` columns.

    Returns:
        None.
    """
    start_date = data["date"].min()
    end_date = data["date"].max()

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        data["date"].to_list(),
        data["weight_lbs"].to_list(),
        marker="o",
        markersize=2.5,
        linewidth=1.25,
    )
    axis.set_title(f"Weight Over Time ({start_date:%b %d, %Y} - {end_date:%b %d, %Y})")
    axis.set_xlabel("Date")
    axis.set_ylabel("Weight (lbs)")
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axis.xaxis.get_major_locator())
    )
    axis.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    """Download the current measurements and save their chart."""
    data = load_weight_data()
    save_chart(data)
    sys.stdout.write(f"Saved {len(data)} measurements to {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
