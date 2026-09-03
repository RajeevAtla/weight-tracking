# Copyright (c) 2026 Rajeev Atla
"""Download weight data from Google Sheets and create a line chart."""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast
from urllib.parse import urlparse
from urllib.request import urlopen

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl

from weight_forecasting import (
    Forecast,
    ModelSelection,
    model_history,
    select_forecast,
)

# Use a non-interactive backend so chart generation also works in CI.
mpl.use("Agg")

SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1qon9wSJhz9pmLyybgV68wkzmHTtT4hyc1cWcb56KOfs/export?format=csv&gid=0"
)
OUTPUT_PATH = Path(__file__).with_name("weight_over_time.png")
FORECAST_OUTPUT_PATH = Path(__file__).with_name("weight_forecast.png")


@dataclass(frozen=True, slots=True)
class ChartSpec:
    """Immutable values and labels required to render the chart."""

    dates: tuple[date, ...]
    weights: tuple[float, ...]
    title: str


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
            try:
                parsed_date = date(year, month_day.month, month_day.day)
            except ValueError:
                dates.append(None)
                continue
            if parsed_date < previous_date:
                try:
                    parsed_date = date(year + 1, month_day.month, month_day.day)
                except ValueError:
                    dates.append(None)
                    continue

        dates.append(parsed_date)
        previous_date = parsed_date

    return dates


def fetch_csv(url: str) -> bytes:
    """Fetch CSV bytes from a public Google Sheets export.

    Args:
        url: Public CSV export URL.

    Returns:
        Raw bytes returned by the CSV endpoint.

    Raises:
        RuntimeError: If the URL cannot be downloaded.
        ValueError: If the URL is not an HTTPS Google Sheets URL.
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or parsed_url.hostname != "docs.google.com":
        message = "Only HTTPS Google Sheets URLs are supported."
        raise ValueError(message)

    try:
        # Network access is kept at the edge so the data pipeline stays pure.
        with urlopen(url, timeout=30) as response:  # noqa: S310
            return response.read()
    except (OSError, ValueError) as error:
        message = f"Could not download the Google Sheet: {error}"
        raise RuntimeError(message) from error


def parse_weight_data(csv_bytes: bytes) -> pl.DataFrame:
    """Parse and clean weight measurements from CSV bytes.

    Args:
        csv_bytes: CSV content returned by the sheet export.

    Returns:
        A DataFrame containing sorted ``date`` and ``weight_lbs`` columns.

    Raises:
        ValueError: If the CSV is invalid, required columns are missing, or no
            valid measurements remain.
    """
    try:
        raw = pl.read_csv(io.BytesIO(csv_bytes), infer_schema_length=0)
    except pl.exceptions.PolarsError as error:
        message = f"Could not read the sheet CSV: {error}"
        raise ValueError(message) from error

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


def build_chart_spec(data: pl.DataFrame) -> ChartSpec:
    """Build an immutable chart specification from sorted measurements.

    Args:
        data: A sorted DataFrame with ``date`` and ``weight_lbs`` columns.

    Returns:
        Immutable dates, weights, and title for the chart renderer.

    Raises:
        ValueError: If the measurement DataFrame is empty.
    """
    if data.is_empty():
        message = "Cannot build a chart specification from empty data."
        raise ValueError(message)

    dates = tuple(cast("list[date]", data["date"].to_list()))
    weights = tuple(cast("list[float]", data["weight_lbs"].to_list()))
    start_date = min(dates)
    end_date = max(dates)

    return ChartSpec(
        dates=dates,
        weights=weights,
        title=f"Weight Over Time ({start_date:%b %d, %Y} - {end_date:%b %d, %Y})",
    )


def render_chart(spec: ChartSpec, output_path: Path) -> None:
    """Render and save an immutable chart specification.

    Args:
        spec: Immutable chart data and labels.
        output_path: Destination path for the PNG file.

    Returns:
        None.
    """
    # Matplotlib mutates a figure and writes the file; this is the deliberate
    # imperative edge around the pure chart specification.
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        mdates.date2num(spec.dates),
        spec.weights,
        marker="o",
        markersize=2.5,
        linewidth=1.25,
    )
    axis.set_title(spec.title)
    axis.set_xlabel("Date")
    axis.set_ylabel("Weight (lbs)")
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axis.xaxis.get_major_locator())
    )
    axis.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_forecast_chart(
    spec: ChartSpec,
    history: Forecast,
    forecast: Forecast,
    output_path: Path,
) -> None:
    """Render observations, historical estimates, and a forecast interval.

    Args:
        spec: Immutable observed chart data and labels.
        history: Historical model estimates and intervals.
        forecast: Future point predictions and intervals.
        output_path: Destination path for the PNG file.

    Returns:
        None.
    """
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        mdates.date2num(spec.dates),
        spec.weights,
        marker="o",
        markersize=2.5,
        linewidth=1.25,
        label="Observed",
    )
    history_dates = mdates.date2num(history.dates)
    axis.plot(
        history_dates,
        history.values,
        linewidth=1.25,
        color="tab:orange",
        label="Historical model estimate",
    )
    axis.fill_between(
        history_dates,
        history.lower,
        history.upper,
        color="tab:orange",
        alpha=0.12,
        label="Historical 95% interval",
    )
    forecast_dates = mdates.date2num(forecast.dates)
    axis.plot(
        forecast_dates,
        forecast.values,
        linestyle="--",
        linewidth=1.5,
        color="tab:green",
        label=f"Forecast ({forecast.model_name.replace('_', ' ')})",
    )
    axis.fill_between(
        forecast_dates,
        forecast.lower,
        forecast.upper,
        color="tab:green",
        alpha=0.2,
        label="Forecast 95% interval",
    )
    axis.axvline(float(mdates.date2num(spec.dates[-1])), color="0.35", linestyle=":")
    axis.set_title(f"{spec.title} and Forecast")
    axis.set_xlabel("Date")
    axis.set_ylabel("Weight (lbs)")
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axis.xaxis.get_major_locator())
    )
    axis.grid(visible=True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Download the current measurements and save their chart."""
    data = parse_weight_data(fetch_csv(SHEET_CSV_URL))
    spec = build_chart_spec(data)
    render_chart(spec, OUTPUT_PATH)
    selection: ModelSelection = select_forecast(data)
    history = model_history(data, selection.score.model_name)
    render_forecast_chart(spec, history, selection.forecast, FORECAST_OUTPUT_PATH)
    sys.stdout.write(f"Saved {len(spec.dates)} measurements to {OUTPUT_PATH}\n")
    sys.stdout.write(
        f"Selected {selection.score.model_name} with "
        f"rolling CV MAE {selection.score.mae:.2f} lbs; "
        f"saved forecast to {FORECAST_OUTPUT_PATH}\n"
    )


if __name__ == "__main__":
    main()
