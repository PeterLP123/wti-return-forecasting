# CLAUDE.md

## Project

WTI crude oil next-day log-return forecasting. Models: LSTM (point + probabilistic) and XGBoost baseline. See README.md for full context.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run grid search (from project root)
python scripts/grid_search_nll.py

# Launch notebooks
jupyter notebook
```

Scripts must be run from the **project root** — `Path(__file__).parent` resolves relative to the script location.

## Data

- **Raw:** `data/raw/final_daily_data_enriched_12_countries.csv` — 227 columns, daily 2016–2026. Do not modify.
- **Processed:** `data/processed/*.parquet` — three feature-set variants (core / full / pca). These are rebuilt by `notebooks/daily/feature_engineering.ipynb`.
- **Results:** `results/nll_grid/` — grid search outputs, created at runtime, not committed.

### Leakage rule

`RobustScaler` (both `scaler_X` and `scaler_y`) must be **fit only on the training split** and then applied to validation and test. Never refit on val/test data.

### Train/val/test split dates

| Split | End date |
|-------|----------|
| Train | 2022-12-31 |
| Validation | 2024-06-30 |
| Test | 2026-03-03 (latest) |

`NON_FEATURE_COLS = ['Date', 'Target', 'WTI_Close', 'Brent_Close']` — exclude these from feature matrices.

## Notebook Order

1. `data_processing.ipynb` — refresh raw data via `yfinance`
2. `data_analysis.ipynb` — EDA (read-only, no outputs saved)
3. `feature_engineering.ipynb` — writes the three parquet files
4. `return_prediction/lstm_model.ipynb` — trains and evaluates LSTM

## Key Conventions

- **Seed:** `SEED = 42` everywhere (`torch.manual_seed`, `np.random.seed`).
- **Device:** auto-detected (`cuda` if available, else `cpu`).
- **Target variable:** always `'Target'` (next-day WTI log return).
- **Default feature dataset for scripts:** `sanity_data_pca.parquet` (Fed text PCA-compressed).
- Grid search saves results **incrementally** to avoid recomputation — check `results/nll_grid/` before re-running.

## Tech Stack

Python 3.9+, PyTorch (LSTM), XGBoost, scikit-learn, pandas, pyarrow, yfinance, matplotlib/seaborn.
