# Data layout

The consolidated research data is intentionally not versioned. It combines third-party-derived market, inventory, central-bank text, and GDELT inputs.

## Required working inputs

Place these files locally before running the pipeline:

```text
data/
├── raw/
│   └── final_daily_data_enriched_12_countries.csv
└── processed/
    └── model_dataset_full.parquet
```

The submitted data snapshot had the following shapes:

| File | Rows | Columns | Date span | Purpose |
|---|---:|---:|---|---|
| `raw/final_daily_data_enriched_12_countries.csv` | 3,535 | 216 | 2 Jul 2016 – 7 Mar 2026 | consolidated source and validation panel |
| `processed/model_dataset_full.parquet` | 1,958 | 72 | 18 Oct 2016 – 4 Mar 2026 | starting feature panel used by preprocessing |

The source snapshot is retained outside this public repository. The notebooks do not contain credentials or download every upstream source from scratch.

## Generated files

Running `data_processing.ipynb` and `feature_engineering.ipynb` creates:

| File | Columns in submitted snapshot | Description |
|---|---:|---|
| `processed_base.parquet` | 60 | refreshed market prices plus consolidated features |
| `sanity_data_core.parquet` | 17 | core market feature set |
| `sanity_data.parquet` | 80 | full modelling feature set |
| `sanity_data_pca.parquet` | 55 | full set with Federal Reserve text features compressed to three principal components |

The three modelling datasets contain 1,895 daily rows from 31 January 2017 to 3 March 2026. Excel mirrors may also be written for inspection. All generated data paths are ignored by Git.

## Leakage rule

Fit preprocessing transformations on the training period only (through 30 December 2022). Apply the fitted transformations unchanged to validation and test data.
