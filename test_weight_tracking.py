# Copyright (c) 2026 Rajeev Atla
"""Tests for the pure weight-data transformation functions."""

import unittest
from datetime import date

import polars as pl

from weight_tracking import ChartSpec, build_chart_spec, parse_dates, parse_weight_data


class ParseDatesTests(unittest.TestCase):
    """Test normalization of full and yearless sheet dates."""

    def test_infers_years_in_source_order(self) -> None:
        """Infer a new year when a month/day value rolls backward."""
        values = parse_dates(["6/6/2024", "12/31", "1/1", "1/2"])

        self.assertEqual(
            values,
            [date(2024, 6, 6), date(2024, 12, 31), date(2025, 1, 1), date(2025, 1, 2)],
        )

    def test_drops_invalid_and_unanchored_values(self) -> None:
        """Represent invalid or unanchored dates as missing values."""
        values = parse_dates(["7/1", "6/6/2024", "2/29", "invalid"])

        self.assertEqual(values, [None, date(2024, 6, 6), None, None])


class ParseWeightDataTests(unittest.TestCase):
    """Test pure CSV selection, cleanup, and sorting."""

    def test_selects_and_sorts_valid_measurements(self) -> None:
        """Keep valid measurements and ignore extra or malformed columns."""
        csv_bytes = (
            b"Date,Weight (lbs),Extra\n"
            b"6/7/2024,157.6,ignored\n"
            b"6/6/2024,158,ignored\n"
            b"1/1,157.5,ignored\n"
            b"6/7,not-a-number,ignored\n"
        )

        data = parse_weight_data(csv_bytes)

        self.assertEqual(
            data.to_dicts(),
            [
                {"date": date(2024, 6, 6), "weight_lbs": 158.0},
                {"date": date(2024, 6, 7), "weight_lbs": 157.6},
                {"date": date(2025, 1, 1), "weight_lbs": 157.5},
            ],
        )

    def test_rejects_missing_columns(self) -> None:
        """Reject CSV content without the required measurement columns."""
        with self.assertRaises(ValueError):
            parse_weight_data(b"date,value\n6/6/2024,158\n")


class BuildChartSpecTests(unittest.TestCase):
    """Test pure chart-specification construction."""

    def test_builds_immutable_chart_spec(self) -> None:
        """Copy measurements into immutable chart values and a date-range title."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 6, 6), date(2024, 6, 7)],
                "weight_lbs": [158.0, 157.5],
            }
        )

        spec = build_chart_spec(data)

        self.assertEqual(
            spec,
            ChartSpec(
                dates=(date(2024, 6, 6), date(2024, 6, 7)),
                weights=(158.0, 157.5),
                title="Weight Over Time (Jun 06, 2024 - Jun 07, 2024)",
            ),
        )

    def test_rejects_empty_data(self) -> None:
        """Reject an empty DataFrame before chart rendering begins."""
        with self.assertRaises(ValueError):
            build_chart_spec(pl.DataFrame({"date": [], "weight_lbs": []}))


if __name__ == "__main__":
    unittest.main()
