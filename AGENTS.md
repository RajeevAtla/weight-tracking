# Repository Guide

This file is the operating contract for coding agents and maintainers working
in this repository. Prefer the repository files and commands below over
assumptions from another project.

## Project Map

- `weight_tracking.py`: Downloads the first Google Sheet tab and writes the
  tracked historical and forecast charts.
- `weight_forecasting.py`: Pure preparation, model fitting, forecasting, and
  rolling-origin validation for sparse weight measurements.
- `test_weight_tracking.py`: Standard-library tests for the pure data and
  historical chart-specification functions.
- `test_weight_forecasting.py`: Standard-library tests for model preparation,
  forecasts, and temporal validation.
- `pyproject.toml`: Project metadata, runtime dependencies, and Ruff/Pyrefly
  configuration.
- `uv.lock`: Reproducible dependency resolution. Update it with uv when
  dependencies change.
- `.python-version`: Pins the project to Python 3.13.
- `.github/workflows/quality.yml`: Runs the local quality checks on pull
  requests and pushes to `main`.
- `README.md`: Human-facing setup and usage instructions.
- `.opencode/`: Local OpenCode configuration. It is intentionally ignored and
  must not be added to commits.

## Environment

- Use uv for Python installation, environments, dependency changes, and
  command execution.
- Do not install project packages globally or edit `uv.lock` by hand.
- The Google Sheet must be publicly readable because the script uses its CSV
  export and does not authenticate.

## Commands

Run these from the repository root:

```powershell
uv python install 3.13
uv sync --locked
uv run python weight_tracking.py
uv run python -m unittest discover --verbose
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check
```

Use `uv run ruff format .` when a formatting change is intended. The chart
generator writes `weight_over_time.png` and `weight_forecast.png` next to the
script.

## Code Rules

- Keep functions small and give every function, including private helpers, a
  Google-style docstring with useful `Args`, `Returns`, and `Raises` sections
  where they apply.
- Keep parsing, validation, and chart-specification functions pure. Isolate
  network, Matplotlib, filesystem, and stdout effects in boundary functions.
- Keep `parse_dates`, `parse_weight_data`, and `build_chart_spec` free of I/O;
  keep `fetch_csv`, `render_chart`, and `main` as the explicit effect boundary.
- Keep type annotations complete and compatible with Python 3.13. Pyrefly is
  configured with its strict preset; do not silence a type error without a
  specific reason.
- Use comments to explain non-obvious intent, data quirks, or tradeoffs. Do
  not comment code by restating what the next line already says.
- Run Ruff formatting before committing Python changes. Keep Ruff lint and
  Pyrefly clean rather than accumulating suppressions.
- Preserve the data contract: use the `Date` and `Weight (lbs)` columns, drop
  invalid measurements, sort by date, and infer omitted years from source row
  order after the first full date.
- Preserve one median model measurement per calendar date when duplicate rows
  exist. Keep missing calendar dates missing for regular-grid models; never
  interpolate them into training targets.
- Use actual elapsed dates for irregular-date harmonic and Gaussian Process
  models. Use expanding-window rolling-origin validation and compare every
  candidate with the last-value baseline.
- Keep generated chart output reproducible. Regenerate and commit the PNG
  files when chart code or source data behavior changes.

## Change Workflow

1. Inspect the relevant source, configuration, and current Git status.
2. Make the smallest change that satisfies the request.
3. Update `README.md` and this file when commands, behavior, dependencies, or
   project structure change.
4. Run `uv sync --locked`, `uv run python -m unittest discover --verbose`,
   `uv run ruff check .`, `uv run ruff format --check .`, and
   `uv run pyrefly check`.
5. Run the chart generator when its behavior or input contract changes.
6. Review the diff, ensure no secrets or local configuration are included,
   then commit only intentional files and push the completed commit to `origin`.

## Delivery

- When implementation and verification are complete, commit and push the
  completed changes to the configured remote; do not leave finished work only
  in the local worktree.
- Never force-push or rewrite shared history. If a push fails, resolve the
  underlying issue and verify the remote branch before reporting completion.

## Living Documentation

`AGENTS.md` is deliberately maintained as part of the codebase rather than
treated as static boilerplate. Whenever a change makes any instruction here
false, update or remove that instruction in the same commit. Add a concise
project-map entry for new durable files, a command for new developer actions,
and an invariant for new behavior that future agents must preserve. Prefer
current facts and rationale over historical notes, and delete stale guidance
instead of letting exceptions accumulate.

## Definition Of Done

A change is complete when the implementation, README, `AGENTS.md`, tool
configuration, lockfile, and CI commands agree; the documented checks pass;
the completed commit is pushed to `origin`; and the working tree contains only
intentional tracked artifacts.
