# PropertyLens

House price estimates for Bangalore, Mumbai and Pune, with a natural-language Price Assistant and a Help & Support assistant. See `implementation_plan.md` for the full design.

## Setup

```
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

## Build data and model

```
.venv/Scripts/python -m core.data_gen   # writes data/housing_data.csv (6,000 rows)
.venv/Scripts/python -m core.train      # benchmarks 5 models, saves LightGBM + metrics to models/
```

## Run

```
.venv/Scripts/streamlit run app.py
```

## Optional: Claude-enhanced assistants

Set `ANTHROPIC_API_KEY` (environment variable or `.streamlit/secrets.toml`). The model defaults to `claude-opus-5-5`; override with `CLAUDE_MODEL`. Without a key, both assistants use the built-in rule-based engines.

## Test

```
.venv/Scripts/python -m pytest
```

## Data note

The bundled dataset is synthetic, calibrated to approximate 2024-25 locality price levels. To use real data, provide a CSV with the same columns at `data/housing_data.csv` and re-run training.
