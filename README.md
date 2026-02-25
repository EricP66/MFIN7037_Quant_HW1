# MFIN7037 Quantitative Finance Homework 1

This repository contains the full implementation and analysis for Homework 1 of the MFIN7037 course.

The project applies quantitative financial analysis techniques including:

- Data preprocessing
- Performance evaluation
- Factor analysis
- Macro asset allocation modelling
- Regression-based performance attribution
- Visualization and reporting

The workflow is implemented using Python and Jupyter Notebook to ensure full reproducibility.

---

# 📁 Repository Structure

```
.
├── Code/
│   ├── Homework1_Q1.ipynb          # Question 1 analysis
│   ├── Homework1_Q2.ipynb          # Question 2 analysis
│   └── Q3/
│       ├── analysis.py             # Core quantitative analysis pipeline
│       ├── fetch_data.py           # Data loading and preprocessing
│       └── visualize.py            # Visualization utilities
│
├── Data/
│   ├── Q1/
│   │   ├── histretSP.csv
│   │   └── mpf_category_annual_returns.csv
│   └── Q3/
│       ├── benchmarks_monthly.csv
│       ├── ff5_factors.csv
│       ├── mom_factor.csv
│       ├── hfgm_monthly.csv
│       ├── analysis_results.pkl
│       └── HFGM_Quantitative_Analysis_Report.xlsx
│
└── README.md
```

---

# 📌 Project Overview

This project aims to analyze investment performance using quantitative finance methods.

Main objectives include:

- Evaluate MPF and market returns
- Perform macro factor analysis
- Decompose performance using Fama-French models
- Construct and analyze multi-asset benchmark portfolios
- Compare live ETF returns with factor-based models

---

# 📊 Dataset Description

## Q1 Analysis Datasets

### 1️⃣ histretSP.csv

Historical asset-class return dataset used as benchmark comparison.

Only analysis-relevant columns are documented below.

| Column Name | Type | Description |
|---|---|---|
| Year | int | Observation year |
| S&P 500 (includes dividends) | float (converted from %) | Total return of US equity market; used as primary equity benchmark |
| US Small cap (bottom decile) | float | Small-cap equity return proxy |
| 3-month T.Bill | float | Risk-free rate proxy |
| US T. Bond (10-year) | float | Long-duration treasury bond return |
| Baa Corporate Bond | float | Credit risk premium proxy |
| Real Estate | float | Real estate asset class return |
| Gold | float | Commodity/inflation hedge proxy |

Notes:

- Returns are stored as percentage strings and converted to numeric values during preprocessing.

---

### 2️⃣ mpf_category_annual_returns.csv

Annual MPF category return dataset used for comparative performance analysis.

| Column Name | Type | Description |
|---|---|---|
| hk_year | int | Calendar year |
| hk_AggressiveAllocation | float | Aggressive allocation MPF category |
| hk_GlobalEquityLargeCap | float | Global large-cap equity exposure |
| hk_GlobalFixedIncome | float | Global bond exposure |
| hk_GreaterChinaEquity | float | Greater China equity category |
| hk_USEquityLargeCapBlend | float | US equity exposure |
| hk_ModerateAllocation | float | Balanced allocation proxy |

Columns prefixed with: “cal_*” represent calibrated or adjusted return series used for analytical comparison.

---

## Q2 Analysis Datasets

Factor model datasets used for asset pricing analysis.

---

### ff.five_factor.parquet

Fama-French Five-Factor dataset.

| Column | Type | Description |
|---|---|---|
| date | datetime | Monthly observation date |
| Mkt-RF | float | Market excess return |
| SMB | float | Size factor (small minus big) |
| HML | float | Value factor |
| RMW | float | Profitability factor |
| CMA | float | Investment factor |
| RF | float | Risk-free rate |

Role in analysis:

- Used for multi-factor regression
- Performance attribution
- Factor exposure estimation

---

### ff.four_factor.parquet

Fama-French Four-Factor dataset (Carhart model).

| Column | Type | Description |
|---|---|---|
| date | datetime | Monthly observation date |
| Mkt-RF | float | Market factor |
| SMB | float | Size factor |
| HML | float | Value factor |
| MOM | float | Momentum factor |

Role:

- Extension of FF3 model
- Momentum-adjusted regression analysis

---

### crsp_202501.dsenames.parquet

CRSP security metadata dataset.

Only analysis-relevant columns:

| Column | Type | Description |
|---|---|---|
| permno | int | CRSP permanent security identifier |
| ticker | string | Stock ticker symbol |
| namedt | datetime | Name start date |
| nameendt | datetime | Name end date |

Role:

- Mapping securities to identifiers
- Ensuring correct asset matching during analysis

---

### crsp_202501.msf.parquet

CRSP monthly stock file containing security-level return and price information.

| Column | Type | Description |
|---|---|---|
| permno | int | CRSP permanent security identifier |
| date | datetime | Observation date (month-end) |
| ret | float | Monthly total return (decimal format, includes dividends) |
| prc | float | Closing price; negative values indicate bid-ask midpoint pricing |
| shrout | float | Shares outstanding (in thousands) |
| hexcd | int | Exchange code (1 = NYSE, 2 = AMEX, 3 = NASDAQ) |

Role in analysis:

- Security-level return dataset
- Used for factor regression and asset-level analysis
- Provides market microstructure context (exchange classification)

---

## Q3 Analysis Datasets

The following datasets are used for macro factor replication and performance attribution analysis.

Only analysis-relevant columns are documented.

---

### benchmarks_monthly.csv

Monthly benchmark asset returns used to construct macro factor exposures.

| Column | Type | Description |
|---|---|---|
| Date | datetime | Monthly observation date |
| SPY | float | US equity market proxy |
| TLT | float | Long-duration US treasury bond proxy |
| GLD | float | Gold ETF (inflation hedge) |
| DBC | float | Commodity index proxy |
| UUP | float | USD index exposure |
| AGG | float | Aggregate bond index |
| EFA | float | Developed international equities |
| VWO | float | Emerging market equities |

Role in analysis:

- Construction of macro factor proxies
- Multi-asset diversification analysis
- Regression benchmark inputs

---

### ff5_factors.csv

Fama-French Five-Factor dataset.

| Column | Type | Description |
|---|---|---|
| Date | datetime | Monthly observation date |
| Mkt-RF | float | Market excess return |
| SMB | float | Size factor (Small Minus Big) |
| HML | float | Value factor |
| RMW | float | Profitability factor |
| CMA | float | Investment factor |
| RF | float | Risk-free rate |

Role:

- Factor regression model
- Performance attribution

---

### mom_factor.csv

Momentum factor dataset.

| Column | Type | Description |
|---|---|---|
| Date | datetime | Monthly observation date |
| MOM | float | Momentum factor return |

Role:

- Extension from FF5 to momentum-adjusted model (FF6)

---

### hfgm_monthly.csv

HFGM ETF monthly return dataset.

| Column | Type | Description |
|---|---|---|
| Date | datetime | Monthly observation date |
| HFGM_Return | float | ETF return series |

Role:

- Target strategy proxy
- Performance comparison vs factor models

---

### analysis_results.pkl

Serialized analysis output.

Contains:

- Regression coefficients
- Performance statistics
- Tracking error metrics
- Risk measures

---

### HFGM_Quantitative_Analysis_Report.xlsx

Final report output generated from analysis pipeline.

Includes:

- Summary tables
- Performance metrics
- Regression outputs

---

# 📈 Analysis Workflow

The analysis pipeline follows a systematic quantitative process:

1. Load and clean datasets.
2. Convert return formats and align time series.
3. Construct benchmark and factor datasets.
4. Perform multi-factor regression analysis.
5. Evaluate performance and risk metrics.
6. Generate visualization and final reports.

---

# 🔁 Reproducibility

- All datasets are stored locally under `/Data`.
- Analysis results can be reproduced by running scripts and notebooks in order.
- Fixed preprocessing ensures consistent output.

---

# 📜 License

This repository is intended for academic use.
