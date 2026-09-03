# Copyright (c) 2026 Rajeev Atla
"""Tests for sparse time-series preparation, models, and validation."""

import math
import unittest
from datetime import date, timedelta

import polars as pl

from weight_forecasting import (
    available_models,
    backtest_models,
    candidate_periods,
    forecast_model,
    make_forecast_dates,
    prepare_model_data,
)


def synthetic_measurements() -> pl.DataFrame:
    """Build a deterministic sparse series with trend and weekly seasonality.

    Returns:
        Synthetic weight measurements with several missing calendar days.
    """
    start = date(2024, 1, 1)
    rows = [
        {
            "date": start + timedelta(days=offset),
            "weight_lbs": 160.0
            + 0.02 * offset
            + 1.5 * math.sin(2.0 * math.pi * offset / 7.0),
        }
        for offset in range(140)
        if offset not in {12, 13, 41, 42, 87, 88, 89}
    ]
    return pl.DataFrame(rows)


class PreparationTests(unittest.TestCase):
    """Test model input normalization and seasonal eligibility."""

    def test_aggregates_duplicate_dates_and_sorts(self) -> None:
        """Use the daily median when multiple measurements share a date."""
        data = pl.DataFrame(
            {
                "date": [date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 2)],
                "weight_lbs": [164.0, 160.0, 162.0],
            }
        )

        prepared = prepare_model_data(data)

        self.assertEqual(
            prepared.to_dicts(),
            [
                {"date": date(2024, 1, 1), "weight_lbs": 160.0},
                {"date": date(2024, 1, 2), "weight_lbs": 163.0},
            ],
        )

    def test_requires_multiple_cycles_for_candidate_periods(self) -> None:
        """Offer only seasonal periods supported by the available history."""
        periods = candidate_periods(prepare_model_data(synthetic_measurements()))

        self.assertEqual(periods, (7, 14, 28))

    def test_available_models_contains_core_families(self) -> None:
        """Expose baselines, state-space, harmonic, and GP model families."""
        models = available_models(prepare_model_data(synthetic_measurements()))

        self.assertIn("last_value", models)
        self.assertIn("kalman_trend", models)
        self.assertIn("harmonic_ridge_7d", models)
        self.assertIn("gaussian_process_7d", models)


class ForecastModelTests(unittest.TestCase):
    """Test finite forecasts from the supported model implementations."""

    def test_models_forecast_sparse_data(self) -> None:
        """Fit each core model without filling the missing observations."""
        data = prepare_model_data(synthetic_measurements())
        last_date = data["date"].to_list()[-1]
        future = make_forecast_dates(last_date, 5)
        model_names = (
            "last_value",
            "seasonal_naive_7d",
            "damped_holt",
            "kalman_trend",
            "kalman_weekly",
            "harmonic_ridge_7d",
            "gaussian_process_7d",
        )

        for model_name in model_names:
            with self.subTest(model=model_name):
                forecast = forecast_model(data, model_name, future)

                self.assertEqual(forecast.dates, future)
                self.assertEqual(len(forecast.values), len(future))
                self.assertTrue(all(math.isfinite(value) for value in forecast.values))
                self.assertTrue(
                    all(
                        lower <= value <= upper
                        for lower, value, upper in zip(
                            forecast.lower,
                            forecast.values,
                            forecast.upper,
                            strict=True,
                        )
                    )
                )

    def test_damped_holt_rejects_long_gaps(self) -> None:
        """Avoid treating a highly irregular sequence as equally spaced."""
        data = pl.DataFrame(
            {
                "date": [
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                    date(2024, 6, 1),
                ],
                "weight_lbs": [160.0, 160.1, 160.2, 161.0],
            }
        )

        with self.assertRaises(ValueError):
            forecast_model(
                data,
                "damped_holt",
                (date(2024, 6, 2),),
            )


class BacktestTests(unittest.TestCase):
    """Test temporal validation and its reported metrics."""

    def test_backtest_uses_expanding_temporal_folds(self) -> None:
        """Report finite metrics for a model evaluated only on future rows."""
        data = prepare_model_data(synthetic_measurements())

        scores = backtest_models(
            data,
            horizon_days=14,
            model_names=("last_value",),
        )

        self.assertEqual(len(scores), 1)
        score = scores[0]
        self.assertEqual(score.model_name, "last_value")
        self.assertGreaterEqual(score.folds, 2)
        self.assertGreater(score.observations, 0)
        self.assertTrue(math.isfinite(score.mae))
        self.assertTrue(math.isfinite(score.rmse))
        self.assertTrue(math.isfinite(score.mase))
        self.assertGreaterEqual(score.coverage, 0.0)
        self.assertLessEqual(score.coverage, 1.0)
        self.assertGreater(score.mean_interval_width, 0.0)


if __name__ == "__main__":
    unittest.main()
