# WTI Crude Oil Next-Day Return Forecasting

A machine learning project for predicting next-day log returns of West Texas Intermediate (WTI) crude oil. The project integrates market data, EIA inventory statistics, Federal Reserve text sentiment, and GDELT geopolitical sentiment across 12 oil-relevant countries.

---

## Project Overview

The target variable is the next-day WTI log return:

```
Target_t = log(WTI_t+1 / WTI_t)
```

Returns are highly non-normal (skewness = −3.80, excess kurtosis = 166), motivating the use of deep learning with uncertainty quantification over linear baselines.

**Time coverage:** 2017-01-31 to 2026-03-03 (~9 years of daily data)

---

## Repository Structure

```
des-moines/
├── data/
│   ├── raw/
│   │   └── final_daily_data_enriched_12_countries.csv   # 227-column daily dataset
│   └── processed/
│       ├── model_dataset_full.parquet                   # Full feature set
│       ├── sanity_data.parquet                          # 80 features (all sources)
│       ├── sanity_data_core.parquet                     # 17 core market features
│       └── sanity_data_pca.parquet                      # 55 features (Fed text PCA-compressed)
├── notebooks/
│   └── daily/
│       ├── data_processing.ipynb         # Data loading and refresh
│       ├── data_analysis.ipynb           # EDA and statistical analysis
│       ├── feature_engineering.ipynb     # Feature derivation and dataset creation
│       └── return_prediction/
│           └── lstm_model.ipynb          # LSTM model training and evaluation
├── scripts/
│   ├── grid_search_nll.py               # XGBoost + LSTM hyperparameter grid search
│   ├── _plot_final.py                   # Final model prediction visualizations
│   ├── _split_plots.py                  # Plotting utilities
│   └── _fix_acf.py                      # ACF and autocorrelation analysis
├── Group 6 DS.pdf                        # Project report
└── requirements.txt
```

---

## Data Sources

| Source | Description | Features |
|--------|-------------|----------|
| Market prices | WTI spot/futures, Brent crude, S&P 500, VIX, DXY | Price levels, log returns |
| EIA inventories | Commercial stocks, SPR, days of supply | Weekly frequency, forward-filled |
| Federal Reserve text | Machine-readable sentiment from Fed documents | 28 features across 3 categories (`FedAll_*`, `FedEP_*`, `FedOG_*`) |
| GDELT | News event sentiment for 12 oil-producing/consuming countries | GoldsteinScale, AvgTone, NumMentions, NumArticles per country |

### Feature Sets (processed datasets)

| Dataset | Columns | Description |
|---------|---------|-------------|
| `sanity_data_core` | 17 | Core market features only (WTI, Brent, S&P, VIX, DXY log returns; lagged returns; rolling volatility) |
| `sanity_data` | 80 | All sources: market + inventory + Fed text + GDELT sentiment |
| `sanity_data_pca` | 55 | Same as `sanity_data` but 28 Fed text features compressed to 3 PCA components (80% variance explained) |

---

## Engineered Features

- **Log returns:** WTI, Brent, S&P 500, VIX, DXY
- **Lagged WTI returns:** Lags 1–10
- **Rolling volatility:** 5-day, 10-day, 20-day, 60-day windows
- **Spot momentum:** 5-day, 20-day, 60-day percent changes
- **Brent-WTI spread:** Level and daily change
- **Fed text PCA:** 28 features → 3 components

---

## Models

### LSTM Neural Network (`notebooks/daily/return_prediction/lstm_model.ipynb`)

A multi-layer LSTM with configurable architecture trained on sliding input sequences.

**Architecture:**
- Input: sequences of length 3–60 timesteps
- LSTM layers: 1–3 layers, hidden size 16–128
- Dropout: 0–0.5
- Optional fully-connected hidden layer (0 or 32 units)
- Output: point estimate (MSE loss) or mean + variance (NLL loss)

**Training setup:**
- Optimizer: Adam (lr = 5×10⁻⁴ to 1×10⁻³)
- Early stopping on validation loss
- Feature scaling: `RobustScaler` fit on training set only

### XGBoost Baseline (`scripts/grid_search_nll.py`)

Gradient boosting regressor with hyperparameter tuning via `TimeSeriesSplit` (3 folds).

### Probabilistic LSTM Variants (`scripts/grid_search_nll.py`)

- **LSTM (NLL):** Predicts mean and variance; trained with Negative Log-Likelihood loss
- **LSTM (NLL + Dirichlet):** Adds Dirichlet-based uncertainty quantification

---

## Evaluation Protocol

Walk-forward (non-leaking) train/validation/test split:

| Split | Period |
|-------|--------|
| Train | Up to 2022-12-31 |
| Validation | 2023-01-01 – 2024-06-30 |
| Test | 2024-07-01 – 2026-03-03 |

**Metrics:** RMSE, MAE, R²

---

## Key EDA Findings

- **Non-normality:** Fat tails and extreme kurtosis make linear models ill-suited
- **Weak linear autocorrelation in returns:** Consistent with Efficient Market Hypothesis
- **Strong autocorrelation in squared returns:** ARCH effects motivate volatility-aware models
- **Time-varying correlations:** Rolling correlations between WTI and macro/sentiment features are unstable, reinforcing the need for walk-forward evaluation
- **Volatility regimes:** Clearly visible around 2020 COVID collapse, 2021–2022 rally, and 2023+ decline

---

## Setup

**Requirements:** Python 3.9+

```bash
pip install -r requirements.txt
```

Key dependencies: `pandas`, `numpy`, `scikit-learn`, `torch`, `xgboost`, `yfinance`, `pyarrow`, `matplotlib`, `seaborn`, `scipy`

### Recommended workflow

1. **Refresh raw data** — `notebooks/daily/data_processing.ipynb`
2. **Explore the data** — `notebooks/daily/data_analysis.ipynb`
3. **Build feature datasets** — `notebooks/daily/feature_engineering.ipynb`
4. **Train and evaluate LSTM** — `notebooks/daily/return_prediction/lstm_model.ipynb`
5. **Run grid search** — `python scripts/grid_search_nll.py`

---

## Project Report

See `Group 6 DS.pdf` for the full methodology, results, and analysis.
