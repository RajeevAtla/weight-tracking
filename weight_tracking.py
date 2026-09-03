from __future__ import annotations

import io
from datetime import date, datetime
from pathlib import Path
from urllib.request import urlopen

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl

SHEET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1qon9wSJhz9pmLyybgV68wkzmHTtT4hyc1cWcb56KOfs/export?format=csv&gid=0"
)
OUTPUT_PATH = Path(__file__).with_name("weight_over_time.png")


def parse_dates(values: list[object]) -> list[date | None]:
    dates: list[date | None] = []
    previous_date: date | None = None
    anchor_year: int | None = None

    for value in values:
        text = "" if value is None else str(value).strip()
        if not text:
            dates.append(None)
            continue

        try:
            parsed_date = datetime.strptime(text, "%m/%d/%Y").date()
            anchor_year = anchor_year or parsed_date.year
        except ValueError:
            try:
                month_day = datetime.strptime(f"{text}/2000", "%m/%d/%Y").date()
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
    try:
        with urlopen(SHEET_CSV_URL, timeout=30) as response:
            csv_bytes = response.read()
    except Exception as error:
        raise RuntimeError(f"Could not download the Google Sheet: {error}") from error

    try:
        raw = pl.read_csv(io.BytesIO(csv_bytes), infer_schema_length=0)
    except Exception as error:
        raise RuntimeError(f"Could not read the sheet CSV: {error}") from error

    required_columns = {"Date", "Weight (lbs)"}
    missing_columns = required_columns - set(raw.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"The sheet is missing required column(s): {missing}")

    selected = raw.select(
        pl.col("Date").cast(pl.String).str.strip_chars().alias("date_text"),
        pl.col("Weight (lbs)")
        .cast(pl.Float64, strict=False)
        .alias("weight_lbs"),
    )
    cleaned = (
        selected.with_columns(
            pl.Series("date", parse_dates(selected["date_text"].to_list()), dtype=pl.Date)
        )
        .select("date", "weight_lbs")
        .drop_nulls()
        .sort("date")
    )

    if cleaned.is_empty():
        raise ValueError("No valid Date/Weight (lbs) rows were found in the sheet.")

    return cleaned


def save_chart(data: pl.DataFrame) -> None:
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
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    data = load_weight_data()
    save_chart(data)
    print(f"Saved {len(data)} measurements to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
