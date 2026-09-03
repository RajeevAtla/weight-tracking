# Copyright (c) 2026 Rajeev Atla
"""Forecast bodyweight with conservative models for sparse time series."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise
from typing import cast

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    ExpSineSquared,
    Matern,
    WhiteKernel,
)
from sklearn.linear_model import Ridge
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning as StatsmodelsConvergenceWarning,
)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.structural import UnobservedComponents

FloatArray = npt.NDArray[np.float64]

FORECAST_HORIZON_DAYS = 28
MAX_BACKTEST_ORIGINS = 12
MIN_TRAINING_POINTS = 28
MIN_HOLT_POINTS = 3
MIN_FEATURE_POINTS = 10
WEEKLY_PERIOD = 7
INTERVAL_Z = 1.96
MIN_INTERVAL_WIDTH = 0.5
SEASONAL_PERIODS = (7, 14, 28)


@dataclass(frozen=True, slots=True)
class Forecast:
    """Point forecast and approximate 95% intervals for future dates."""

    model_name: str
    dates: tuple[date, ...]
    values: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ModelScore:
    """Rolling-origin score for one forecasting model."""

    model_name: str
    mae: float
    rmse: float
    mase: float
    coverage: float
    mean_interval_width: float
    observations: int
    folds: int


@dataclass(frozen=True, slots=True)
class ModelSelection:
    """Selected forecast and the scores used to select its model."""

    forecast: Forecast
    score: ModelScore
    scores: tuple[ModelScore, ...]


def prepare_model_data(data: pl.DataFrame) -> pl.DataFrame:
    """Normalize measurements for model fitting.

    Args:
        data: DataFrame containing ``date`` and ``weight_lbs`` columns.

    Returns:
        Sorted, non-null measurements with one median value per date.

    Raises:
        ValueError: If required columns are missing or no valid measurements
            remain.
    """
    required_columns = {"date", "weight_lbs"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        message = f"Model data is missing required column(s): {missing}"
        raise ValueError(message)

    try:
        prepared = (
            data.select(
                pl.col("date").cast(pl.Date),
                pl.col("weight_lbs").cast(pl.Float64, strict=False),
            )
            .drop_nulls()
            .group_by("date")
            .agg(pl.col("weight_lbs").median())
            .sort("date")
        )
    except pl.exceptions.PolarsError as error:
        message = f"Could not prepare model data: {error}"
        raise ValueError(message) from error

    if prepared.is_empty():
        message = "No valid measurements remain for model fitting."
        raise ValueError(message)
    return prepared


def candidate_periods(data: pl.DataFrame) -> tuple[int, ...]:
    """Return seasonal periods supported by the available history.

    Args:
        data: Prepared, sorted measurement data.

    Returns:
        Candidate periods in days. A period is included only when at least four
        cycles and enough observations are available.
    """
    dates = data["date"].to_list()
    if not dates:
        return ()

    span_days = (dates[-1] - dates[0]).days
    observations = len(dates)
    return tuple(
        period
        for period in SEASONAL_PERIODS
        if span_days >= 4 * period and observations >= max(20, 2 * period)
    )


def make_forecast_dates(last_date: date, horizon_days: int) -> tuple[date, ...]:
    """Build a daily forecast horizon after the last measurement.

    Args:
        last_date: Date of the latest observed measurement.
        horizon_days: Number of future calendar days to forecast.

    Returns:
        Consecutive dates after ``last_date``.

    Raises:
        ValueError: If ``horizon_days`` is not positive.
    """
    if horizon_days < 1:
        message = "Forecast horizon must be positive."
        raise ValueError(message)
    return tuple(
        last_date + timedelta(days=offset) for offset in range(1, horizon_days + 1)
    )


def available_models(data: pl.DataFrame) -> tuple[str, ...]:
    """List models suitable for the available measurement history.

    Args:
        data: Prepared, sorted measurement data.

    Returns:
        Model names to evaluate. Seasonal variants are included only when the
        data contains enough history for their period.
    """
    periods = candidate_periods(data)
    models = ["last_value", "damped_holt", "kalman_trend"]
    if WEEKLY_PERIOD in periods:
        models.append("seasonal_naive_7d")
        models.append("kalman_weekly")
    models.extend(f"harmonic_ridge_{period}d" for period in periods)
    models.extend(f"gaussian_process_{period}d" for period in periods)
    return tuple(models)


def forecast_model(
    data: pl.DataFrame,
    model_name: str,
    forecast_dates: tuple[date, ...],
) -> Forecast:
    """Fit one model and forecast the requested future dates.

    Args:
        data: Raw or prepared DataFrame with ``date`` and ``weight_lbs``.
        model_name: Name returned by :func:`available_models`.
        forecast_dates: Sorted dates after the latest measurement.

    Returns:
        Forecast values and approximate 95% intervals.

    Raises:
        ValueError: If the data, model name, or forecast dates are invalid.
        RuntimeError: If the selected model cannot be fitted.
    """
    prepared = prepare_model_data(data)
    return _forecast_prepared(prepared, model_name, forecast_dates)


def backtest_models(
    data: pl.DataFrame,
    horizon_days: int = FORECAST_HORIZON_DAYS,
    model_names: tuple[str, ...] | None = None,
) -> tuple[ModelScore, ...]:
    """Evaluate models with expanding-window rolling-origin validation.

    Args:
        data: Raw or prepared DataFrame with ``date`` and ``weight_lbs``.
        horizon_days: Maximum number of future calendar days in each fold.
        model_names: Optional explicit model names to evaluate.

    Returns:
        Scores sorted by increasing mean absolute error. Models that fail on
        every fold are omitted.

    Raises:
        ValueError: If ``horizon_days`` is not positive or the data is invalid.
    """
    if horizon_days < 1:
        message = "Backtest horizon must be positive."
        raise ValueError(message)

    prepared = prepare_model_data(data)
    dates = tuple(prepared["date"].to_list())
    names = model_names or available_models(prepared)
    folds = _backtest_folds(dates, horizon_days)
    scores: list[ModelScore] = []

    for name in names:
        errors: list[float] = []
        mase_errors: list[float] = []
        interval_hits: list[bool] = []
        interval_widths: list[float] = []
        completed_folds = 0
        for cutoff, test_dates in folds:
            train = prepared.filter(pl.col("date") <= cutoff)
            try:
                forecast = _forecast_prepared(train, name, test_dates)
            except (
                FloatingPointError,
                np.linalg.LinAlgError,
                RuntimeError,
                ValueError,
            ):
                continue

            predictions = dict(zip(forecast.dates, forecast.values, strict=True))
            actual = dict(
                zip(
                    test_dates,
                    prepared.filter(pl.col("date").is_in(test_dates))[
                        "weight_lbs"
                    ].to_list(),
                    strict=True,
                )
            )
            lower = dict(zip(forecast.dates, forecast.lower, strict=True))
            upper = dict(zip(forecast.dates, forecast.upper, strict=True))
            fold_errors = [actual[day] - predictions[day] for day in test_dates]
            if not all(math.isfinite(error) for error in fold_errors):
                continue
            train_weights = np.asarray(train["weight_lbs"].to_list(), dtype=np.float64)
            scale = max(
                MIN_INTERVAL_WIDTH,
                float(np.mean(np.abs(np.diff(train_weights))))
                if len(train_weights) > 1
                else MIN_INTERVAL_WIDTH,
            )
            errors.extend(fold_errors)
            mase_errors.extend(abs(error) / scale for error in fold_errors)
            interval_hits.extend(
                lower[day] <= actual[day] <= upper[day] for day in test_dates
            )
            interval_widths.extend(upper[day] - lower[day] for day in test_dates)
            completed_folds += 1

        if errors:
            error_array = np.asarray(errors, dtype=np.float64)
            scores.append(
                ModelScore(
                    model_name=name,
                    mae=float(np.mean(np.abs(error_array))),
                    rmse=float(np.sqrt(np.mean(error_array**2))),
                    mase=float(np.mean(mase_errors)),
                    coverage=float(np.mean(interval_hits)),
                    mean_interval_width=float(np.mean(interval_widths)),
                    observations=len(errors),
                    folds=completed_folds,
                )
            )

    return tuple(sorted(scores, key=_score_key))


def select_forecast(
    data: pl.DataFrame,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> ModelSelection:
    """Select the best validated model and fit it to all measurements.

    Args:
        data: Raw or prepared DataFrame with ``date`` and ``weight_lbs``.
        horizon_days: Number of future calendar days to forecast.

    Returns:
        The selected forecast, its validation score, and all available scores.

    Raises:
        ValueError: If ``horizon_days`` is not positive or the data is invalid.
    """
    if horizon_days < 1:
        message = "Forecast horizon must be positive."
        raise ValueError(message)

    prepared = prepare_model_data(data)
    last_date = prepared["date"].to_list()[-1]
    forecast_dates = make_forecast_dates(last_date, horizon_days)
    scores = backtest_models(prepared, horizon_days=horizon_days)

    if scores:
        score = scores[0]
        try:
            forecast = _forecast_prepared(prepared, score.model_name, forecast_dates)
        except (
            FloatingPointError,
            np.linalg.LinAlgError,
            RuntimeError,
            ValueError,
        ):
            score = _fallback_score(scores)
            forecast = _forecast_prepared(prepared, score.model_name, forecast_dates)
    else:
        score = ModelScore(
            model_name="last_value",
            mae=math.nan,
            rmse=math.nan,
            mase=math.nan,
            coverage=math.nan,
            mean_interval_width=math.nan,
            observations=0,
            folds=0,
        )
        forecast = _forecast_prepared(prepared, score.model_name, forecast_dates)

    return ModelSelection(forecast=forecast, score=score, scores=scores)


def _forecast_prepared(
    data: pl.DataFrame,
    model_name: str,
    forecast_dates: tuple[date, ...],
) -> Forecast:
    """Fit a model using already prepared data.

    Args:
        data: Prepared, sorted measurement data.
        model_name: Model identifier.
        forecast_dates: Sorted future dates.

    Returns:
        Forecast for every requested date.

    Raises:
        ValueError: If dates are invalid or the model identifier is unknown.
        RuntimeError: If a model fitting routine fails.
    """
    dates = tuple(data["date"].to_list())
    weights = np.asarray(data["weight_lbs"].to_list(), dtype=np.float64)
    if not forecast_dates or any(day <= dates[-1] for day in forecast_dates):
        message = "Forecast dates must be after the latest measurement."
        raise ValueError(message)
    if tuple(sorted(forecast_dates)) != forecast_dates:
        message = "Forecast dates must be sorted."
        raise ValueError(message)

    if model_name == "last_value":
        forecast = _last_value_forecast(dates, weights, forecast_dates)
    elif model_name == "seasonal_naive_7d":
        forecast = _seasonal_naive_forecast(
            dates, weights, forecast_dates, period=WEEKLY_PERIOD
        )
    elif model_name == "damped_holt":
        forecast = _damped_holt_forecast(dates, weights, forecast_dates)
    elif model_name == "kalman_trend":
        forecast = _kalman_forecast(dates, weights, forecast_dates, period=None)
    elif model_name == "kalman_weekly":
        forecast = _kalman_forecast(
            dates, weights, forecast_dates, period=WEEKLY_PERIOD
        )
    elif model_name.startswith("harmonic_ridge_"):
        period = _period_from_model_name(model_name, "harmonic_ridge_")
        forecast = _harmonic_ridge_forecast(dates, weights, forecast_dates, period)
    elif model_name.startswith("gaussian_process_"):
        period = _period_from_model_name(model_name, "gaussian_process_")
        forecast = _gaussian_process_forecast(dates, weights, forecast_dates, period)
    else:
        message = f"Unknown forecasting model: {model_name}"
        raise ValueError(message)
    return _validate_forecast(forecast)


def _validate_forecast(forecast: Forecast) -> Forecast:
    """Reject non-finite predictions or malformed prediction intervals.

    Args:
        forecast: Forecast to validate.

    Returns:
        The unchanged valid forecast.

    Raises:
        RuntimeError: If values, intervals, or lengths are invalid.
    """
    lengths = {
        len(forecast.dates),
        len(forecast.values),
        len(forecast.lower),
        len(forecast.upper),
    }
    valid = len(lengths) == 1 and all(
        math.isfinite(value)
        for values in (forecast.values, forecast.lower, forecast.upper)
        for value in values
    )
    valid = valid and all(
        lower <= value <= upper
        for lower, value, upper in zip(
            forecast.lower, forecast.values, forecast.upper, strict=True
        )
    )
    if not valid:
        message = f"Model produced an invalid forecast: {forecast.model_name}"
        raise RuntimeError(message)
    return forecast


def _backtest_folds(
    dates: tuple[date, ...], horizon_days: int
) -> tuple[tuple[date, tuple[date, ...]], ...]:
    """Build recent expanding-window validation folds.

    Args:
        dates: Sorted unique measurement dates.
        horizon_days: Maximum future span in each fold.

    Returns:
        Cutoff dates paired with observed future dates, limited to recent folds.
    """
    if len(dates) <= MIN_TRAINING_POINTS:
        return ()

    folds = []
    for index in range(MIN_TRAINING_POINTS, len(dates)):
        cutoff = dates[index - 1]
        test_dates = tuple(
            day for day in dates[index:] if day <= cutoff + timedelta(days=horizon_days)
        )
        if test_dates:
            folds.append((cutoff, test_dates))

    return tuple(folds[-MAX_BACKTEST_ORIGINS:])


def _fallback_score(scores: tuple[ModelScore, ...]) -> ModelScore:
    """Return the simplest successfully scored model.

    Args:
        scores: Scores sorted by validation performance.

    Returns:
        The last-value score when available, otherwise the best score.
    """
    for score in scores:
        if score.model_name == "last_value":
            return score
    return scores[0]


def _score_key(score: ModelScore) -> tuple[float, float]:
    """Return the validation metrics used to order model scores.

    Args:
        score: Model score to order.

    Returns:
        Mean absolute error followed by root mean squared error.
    """
    return score.mae, score.rmse


def _period_from_model_name(model_name: str, prefix: str) -> int:
    """Extract a positive day period from a model name.

    Args:
        model_name: Model name containing a numeric day suffix.
        prefix: Prefix before the numeric period.

    Returns:
        Parsed period in days.

    Raises:
        ValueError: If the suffix is missing or not positive.
    """
    try:
        period = int(model_name.removeprefix(prefix).removesuffix("d"))
    except ValueError as error:
        message = f"Invalid seasonal model name: {model_name}"
        raise ValueError(message) from error
    if period < 1:
        message = f"Seasonal period must be positive: {model_name}"
        raise ValueError(message)
    return period


def _last_value_forecast(
    dates: tuple[date, ...], weights: FloatArray, forecast_dates: tuple[date, ...]
) -> Forecast:
    """Forecast the latest measurement at every future date.

    Args:
        dates: Observed measurement dates.
        weights: Observed weight values.
        forecast_dates: Future dates to forecast.

    Returns:
        Constant last-value forecast with a robust residual interval.
    """
    del dates
    value = float(weights[-1])
    width = INTERVAL_Z * _residual_scale(np.diff(weights))
    return _constant_interval_forecast(
        "last_value", forecast_dates, np.full(len(forecast_dates), value), width
    )


def _seasonal_naive_forecast(
    dates: tuple[date, ...],
    weights: FloatArray,
    forecast_dates: tuple[date, ...],
    period: int,
) -> Forecast:
    """Repeat the latest exact measurement from one seasonal period earlier.

    Args:
        dates: Observed measurement dates.
        weights: Observed weight values.
        forecast_dates: Future dates to forecast.
        period: Seasonal period in days.

    Returns:
        Seasonal-naive forecast with a robust residual interval.
    """
    observed = dict(zip(dates, weights.tolist(), strict=True))
    latest = float(weights[-1])
    values = np.asarray(
        [observed.get(day - timedelta(days=period), latest) for day in forecast_dates],
        dtype=np.float64,
    )
    residuals = [
        value - observed[day - timedelta(days=period)]
        for day, value in zip(dates, weights, strict=True)
        if day - timedelta(days=period) in observed
    ]
    width = INTERVAL_Z * _residual_scale(np.asarray(residuals, dtype=np.float64))
    return _constant_interval_forecast(
        f"seasonal_naive_{period}d", forecast_dates, values, width
    )


def _damped_holt_forecast(
    dates: tuple[date, ...], weights: FloatArray, forecast_dates: tuple[date, ...]
) -> Forecast:
    """Fit a damped additive trend to the observed sequence.

    Args:
        dates: Observed measurement dates.
        weights: Observed weight values.
        forecast_dates: Future dates to forecast.

    Returns:
        Damped trend forecast with a residual-based interval.

    Raises:
        ValueError: If observations are too sparse or too few for the model.
        RuntimeError: If the exponential-smoothing fit fails.
    """
    if len(weights) < MIN_HOLT_POINTS:
        message = "Damped Holt requires at least three measurements."
        raise ValueError(message)
    if len(dates) > 1:
        gaps = [(right - left).days for left, right in pairwise(dates)]
        cadence = max(1, round(float(np.median(gaps))))
        if max(gaps) > 4 * cadence:
            message = "Damped Holt requires mostly regular observations."
            raise ValueError(message)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=StatsmodelsConvergenceWarning)
            fitted = ExponentialSmoothing(
                weights,
                trend="add",
                damped_trend=True,
                initialization_method="estimated",
            ).fit(optimized=True, remove_bias=False)
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as error:
        message = "Damped Holt could not be fitted."
        raise RuntimeError(message) from error

    fitted_values = np.asarray(fitted.fittedvalues, dtype=np.float64)
    residuals = weights - fitted_values
    width = INTERVAL_Z * _residual_scale(residuals)
    return _sequence_forecast("damped_holt", fitted, dates, forecast_dates, width)


def _sequence_forecast(
    model_name: str,
    fitted: object,
    dates: tuple[date, ...],
    forecast_dates: tuple[date, ...],
    width: float,
) -> Forecast:
    """Project a sequence model onto future calendar dates.

    Args:
        model_name: Name to store in the result.
        fitted: Statsmodels fitted smoothing result.
        dates: Observed measurement dates.
        forecast_dates: Future dates to forecast.
        width: Half-width of the prediction interval.

    Returns:
        Forecast values selected at the estimated observation cadence.

    Raises:
        RuntimeError: If the fitted model does not expose a forecast method.
    """
    if not hasattr(fitted, "forecast"):
        message = "Fitted sequence model has no forecast method."
        raise RuntimeError(message)
    forecast_method = fitted.forecast
    cadence = 1
    if len(dates) > 1:
        gaps = np.asarray(
            [(right - left).days for left, right in pairwise(dates)],
            dtype=np.float64,
        )
        cadence = max(1, round(float(np.median(gaps))))
    horizon_days = (forecast_dates[-1] - dates[-1]).days
    steps = max(1, math.ceil(horizon_days / cadence))
    values = np.asarray(forecast_method(steps), dtype=np.float64)
    indexes = [
        min(
            len(values) - 1,
            max(0, math.ceil((day - dates[-1]).days / cadence) - 1),
        )
        for day in forecast_dates
    ]
    predictions = values[indexes]
    return _constant_interval_forecast(model_name, forecast_dates, predictions, width)


def _kalman_forecast(
    dates: tuple[date, ...],
    weights: FloatArray,
    forecast_dates: tuple[date, ...],
    period: int | None,
) -> Forecast:
    """Fit a missing-aware local trend state-space model.

    Args:
        dates: Observed measurement dates.
        weights: Observed weight values.
        forecast_dates: Future dates to forecast.
        period: Optional stochastic seasonal period in days.

    Returns:
        State-space forecast with model-derived 95% intervals.

    Raises:
        RuntimeError: If the state-space model cannot be fitted.
    """
    if period is not None and len(weights) < 2 * period:
        message = "Seasonal Kalman model has too few measurements."
        raise ValueError(message)

    last_target = forecast_dates[-1]
    start_date = dates[0]
    grid_length = (last_target - start_date).days + 1
    grid_values = np.full(grid_length, np.nan, dtype=np.float64)
    for observed_date, weight in zip(dates, weights, strict=True):
        grid_values[(observed_date - start_date).days] = weight

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=StatsmodelsConvergenceWarning)
            fitted = UnobservedComponents(
                grid_values,
                level=True,
                trend=True,
                seasonal=period,
                irregular=True,
                stochastic_level=True,
                stochastic_trend=True,
                stochastic_seasonal=period is not None,
            ).fit(disp=False, maxiter=200)
        prediction = fitted.get_forecast(steps=(last_target - dates[-1]).days)
        values = np.asarray(prediction.predicted_mean, dtype=np.float64)
        intervals = np.asarray(prediction.conf_int(alpha=0.05), dtype=np.float64)
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as error:
        message = "Kalman model could not be fitted."
        raise RuntimeError(message) from error

    indexes = [(day - dates[-1]).days - 1 for day in forecast_dates]
    return Forecast(
        model_name="kalman_weekly" if period is not None else "kalman_trend",
        dates=forecast_dates,
        values=tuple(float(values[index]) for index in indexes),
        lower=tuple(float(intervals[index, 0]) for index in indexes),
        upper=tuple(float(intervals[index, 1]) for index in indexes),
    )


def _harmonic_ridge_forecast(
    dates: tuple[date, ...],
    weights: FloatArray,
    forecast_dates: tuple[date, ...],
    period: int,
) -> Forecast:
    """Fit a regularized trend plus harmonic seasonal regression.

    Args:
        dates: Observed measurement dates.
        weights: Observed weight values.
        forecast_dates: Future dates to forecast.
        period: Seasonal period in days.

    Returns:
        Ridge forecast with a robust residual-based interval.

    Raises:
        ValueError: If too few observations are available for the features.
    """
    if len(weights) < MIN_FEATURE_POINTS:
        message = "Harmonic regression requires at least ten measurements."
        raise ValueError(message)
    origin = dates[0]
    train_features = _harmonic_features(_elapsed_days(dates, origin), period)
    future_features = _harmonic_features(_elapsed_days(forecast_dates, origin), period)
    model = Ridge(alpha=1.0)
    model.fit(train_features, weights)
    fitted_values = np.asarray(model.predict(train_features), dtype=np.float64)
    predictions = np.asarray(model.predict(future_features), dtype=np.float64)
    width = INTERVAL_Z * _residual_scale(weights - fitted_values)
    return _constant_interval_forecast(
        f"harmonic_ridge_{period}d", forecast_dates, predictions, width
    )


def _gaussian_process_forecast(
    dates: tuple[date, ...],
    weights: FloatArray,
    forecast_dates: tuple[date, ...],
    period: int,
) -> Forecast:
    """Fit a Matérn plus quasi-periodic Gaussian Process.

    Args:
        dates: Observed measurement dates.
        weights: Observed weight values.
        forecast_dates: Future dates to forecast.
        period: Seasonal period in days.

    Returns:
        Gaussian Process forecast with model-derived 95% intervals.

    Raises:
        RuntimeError: If the Gaussian Process cannot be fitted.
    """
    if len(weights) < MIN_FEATURE_POINTS:
        message = "Gaussian Process requires at least ten measurements."
        raise ValueError(message)

    origin = dates[0]
    scale = 365.25
    train_days = _elapsed_days(dates, origin) / scale
    future_days = _elapsed_days(forecast_dates, origin) / scale
    periodicity = period / scale
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=0.15, length_scale_bounds=(0.01, 2.0), nu=1.5)
        + ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=0.5,
            length_scale_bounds=(0.05, 2.0),
            nu=1.5,
        )
        * ExpSineSquared(
            length_scale=0.03,
            periodicity=periodicity,
            length_scale_bounds=(0.003, 0.5),
            periodicity_bounds="fixed",
        )
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-4, 25.0))
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=0,
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SklearnConvergenceWarning)
            model.fit(train_days.reshape(-1, 1), weights)
            predictions, standard_deviation = cast(
                "tuple[FloatArray, FloatArray]",
                model.predict(future_days.reshape(-1, 1), return_std=True),
            )
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
        message = "Gaussian Process could not be fitted."
        raise RuntimeError(message) from error

    standard_deviation = np.maximum(
        np.asarray(standard_deviation, dtype=np.float64), 0.0
    )
    predictions = np.asarray(predictions, dtype=np.float64)
    return Forecast(
        model_name=f"gaussian_process_{period}d",
        dates=forecast_dates,
        values=tuple(float(value) for value in predictions),
        lower=tuple(
            float(value) for value in predictions - INTERVAL_Z * standard_deviation
        ),
        upper=tuple(
            float(value) for value in predictions + INTERVAL_Z * standard_deviation
        ),
    )


def _elapsed_days(dates: tuple[date, ...], origin: date) -> FloatArray:
    """Convert dates to elapsed floating-point days.

    Args:
        dates: Dates to convert.
        origin: Date used as day zero.

    Returns:
        One-dimensional floating-point elapsed-day array.
    """
    return np.asarray([(day - origin).days for day in dates], dtype=np.float64)


def _harmonic_features(days: FloatArray, period: int) -> FloatArray:
    """Build trend and two Fourier harmonics for a seasonal period.

    Args:
        days: Elapsed days from the training origin.
        period: Seasonal period in days.

    Returns:
        Feature matrix with a scaled trend and sine/cosine terms.
    """
    years = days / 365.25
    features = [years]
    for harmonic in (1, 2):
        angle = 2.0 * np.pi * harmonic * days / period
        features.extend((np.sin(angle), np.cos(angle)))
    return np.column_stack(features).astype(np.float64)


def _residual_scale(residuals: FloatArray) -> float:
    """Estimate a robust scale for an approximate prediction interval.

    Args:
        residuals: Model or naive residuals.

    Returns:
        Positive scale in pounds, with a 0.5-pound lower bound.
    """
    finite = residuals[np.isfinite(residuals)]
    if len(finite) == 0:
        return MIN_INTERVAL_WIDTH
    median = float(np.median(finite))
    scale = 1.4826 * float(np.median(np.abs(finite - median)))
    if not math.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(finite))
    if not math.isfinite(scale) or scale <= 0.0:
        scale = MIN_INTERVAL_WIDTH
    return max(scale, MIN_INTERVAL_WIDTH)


def _constant_interval_forecast(
    model_name: str,
    forecast_dates: tuple[date, ...],
    values: FloatArray,
    width: float,
) -> Forecast:
    """Create a forecast with a constant symmetric interval.

    Args:
        model_name: Name to store in the result.
        forecast_dates: Future dates to forecast.
        values: Point predictions.
        width: Positive half-width of the interval.

    Returns:
        Immutable forecast values and intervals.

    Raises:
        ValueError: If predictions are non-finite or lengths do not match.
    """
    predictions = np.asarray(values, dtype=np.float64)
    if len(predictions) != len(forecast_dates) or not np.all(np.isfinite(predictions)):
        message = "Forecast predictions must be finite and match dates."
        raise ValueError(message)
    half_width = max(MIN_INTERVAL_WIDTH, float(width))
    return Forecast(
        model_name=model_name,
        dates=forecast_dates,
        values=tuple(float(value) for value in predictions),
        lower=tuple(float(value - half_width) for value in predictions),
        upper=tuple(float(value + half_width) for value in predictions),
    )
