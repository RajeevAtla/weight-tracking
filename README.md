# Weight Tracking

Fetches weight data from the configured public Google Sheet and saves a
historical chart as `weight_over_time.png` plus a 28-day forecast chart as
`weight_forecast.png`.

## Setup

Install Python and create the uv environment:

```powershell
uv python install 3.13
uv sync --locked
```

`uv sync` installs Polars, Matplotlib, NumPy, scikit-learn, and statsmodels
from `pyproject.toml`.

## Run

```powershell
uv run python weight_tracking.py
```

The Google Sheet must remain publicly readable.

## Forecasting

The script evaluates these conservative time-series models with recent
expanding-window backtests:

- Last-value and seven-day seasonal-naive baselines.
- Damped Holt trend when the observed cadence is mostly regular.
- Local trend and weekly seasonal state-space models using a daily grid with
  missing observations retained.
- Ridge regression with trend and Fourier terms for irregular dates.
- Matérn plus quasi-periodic Gaussian Process models for irregular dates and
  uncertainty intervals.

Models are scored with MAE, RMSE, MASE, and interval coverage. The best
validated model is used for the forecast chart. Missing measurements are not
interpolated into training targets, and seasonal periods are considered only
when enough cycles are present. The forecast is a descriptive estimate, not a
medical prediction.

## Quality Checks

```powershell
uv run python -m unittest discover --verbose
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check
```

To format the Python source locally:

```powershell
uv run ruff format .
```
