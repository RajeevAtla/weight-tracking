# Weight Tracking

Fetches weight data from the configured public Google Sheet and saves a line
chart as `weight_over_time.png`.

## Setup

Install Python and create the uv environment:

```powershell
uv python install 3.13
uv sync
```

`uv sync` installs Polars and Matplotlib from `pyproject.toml`.

## Run

```powershell
uv run python weight_tracking.py
```

The Google Sheet must remain publicly readable.

## Quality Checks

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check
```

To format the Python source locally:

```powershell
uv run ruff format .
```
