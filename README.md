<div align="center">

# 🏦 iyzico Merchant Payment Volume Forecasting

**92-day ahead daily payment volume prediction for individual merchants**  
*LightGBM · Baseline ensemble · COVID-19 anomaly handling · Turkish holiday features*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-green)](https://lightgbm.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-PEP8-black)](https://peps.python.org/pep-0008/)

</div>

---

## Overview

This pipeline forecasts **92 days of daily `Total_Paid`** for each iyzico merchant, using only information that is genuinely available at forecast time (no leakage). It handles:

- **Turkish public holidays & religious festivals** (Ramazan, Kurban Bayramı) with half-day features
- **Shopping seasons** (Black Friday ±3 days, 11/11)
- **COVID-19 pandemic anomaly** via three configurable modes
- **Automatic model selection** with a simplicity-first tolerance rule

The full pipeline runs in ~2 minutes on a laptop for 7 merchants.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Feature Engineering](#feature-engineering)
- [COVID-19 Handling](#covid-19-handling)
- [Model Selection](#model-selection)
- [Outputs & Figures](#outputs--figures)
- [Configuration Reference](#configuration-reference)
- [Contributing](#contributing)

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/senturkilayda/iyzico-forecasting.git
cd iyzico-forecasting

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your data file
mkdir data
cp /path/to/iyzico_data.csv data/

# 5. Run the pipeline
python iyzico_forecasting.py
```

Results are written to `figures/` and `outputs/`. That's it.

> **Data format:** The pipeline expects a CSV with columns  
> `merchant_id`, `transaction_date`, `Total_Transaction`, `Total_Paid`.

---

## Project Structure

```
iyzico-forecasting/
│
├── iyzico_forecasting.py   # Single-file pipeline (all logic here)
├── requirements.txt
├── LICENSE
├── README.md
│
├── data/
│   └── iyzico_data.csv     # ← place your data here
│
├── figures/                # auto-created on first run
│   ├── 01_monthly_volume.png
│   ├── 02_yearly_seasonality.png
│   ├── 03_weekday_heatmap.png
│   ├── 04_data_availability.png
│   ├── 05_model_comparison.png
│   ├── 06_validation_fold_a.png
│   ├── 06_validation_fold_b.png
│   ├── 07_feature_importance.png
│   ├── 08_test_forecast.png
│   ├── 09_learning_curves_include.png
│   ├── 09_learning_curves_flag.png
│   └── 09_learning_curves_exclude.png
│
└── outputs/                # auto-created on first run
    ├── iyzico_test_forecast.csv
    ├── validation_results.csv
    ├── holiday_ablation.csv
    └── covid_ablation.csv
```

---

## Pipeline Architecture

```
Raw CSV
  │
  ▼
Data Quality Report
  │  (duplicate check, missing %, per-merchant lifetime)
  ▼
Daily Panel (merchant × date)
  │  (no rows before a merchant's first observation)
  ▼
Feature Engineering
  │  calendar · holiday/festival · shopping · lag · rolling · EWM
  ▼
COVID Mode Branching ──────────────────────────────────────────┐
  include │   flag │   exclude                                  │
          │        │                                            │
          ▼        ▼        ▼                                   │
     Cross-Validation (2 Folds × 3 COVID Modes)                │
          │  Fold A: Oct–Dec 2019 (pre-COVID)                   │
          │  Fold B: Jul–Sep 2020 (post-COVID)                  │
          │  Same rows evaluated across ALL candidates ─────────┘
          ▼
  All Candidates Ranked by Macro SMAPE
  (6 baselines + 18 LightGBM variants)
          │
          ▼  Tolerance Rule (±0.10 SMAPE pts → pick simplest)
  Final Model Selected
          │
          ▼
  Oct–Dec 2020 Forecast  →  outputs/iyzico_test_forecast.csv
```

---

## Feature Engineering

### Lag Features

| Feature | Looks back | Why |
|---|---|---|
| `paid_lag_92` | 92 days | Exact horizon boundary |
| `paid_lag_98` | 98 days | ~14 weeks |
| `paid_lag_182` | 182 days | ~6 months |
| `paid_lag_364` | 364 days | Same weekday, prior year |
| `paid_lag_366` | 366 days | Leap-year safe prior year |

All lag/rolling/EWM features are shifted by **≥ 92 days** — guaranteeing no leakage at a 92-day forecast horizon.

### Rolling & EWM Features

Rolling means and exponential weighted means at 7, 28, and 91-day windows, all shifted by the forecast horizon before windowing.

### Holiday & Calendar Features

| Feature | Description |
|---|---|
| `is_holiday` | Turkish fixed public holidays (New Year, ANZAC, Labour Day, etc.) |
| `is_bayram` | Ramazan & Kurban Bayramı days (dates manually verified) |
| `is_half_day` | Festival eves + October 28 (Republic Day eve) |
| `days_to_bayram` | Countdown to next festival (capped at 30) |
| `days_since_bayram` | Days since last festival ended (capped at 30) |
| `is_black_friday_window` | Black Friday ±3 days (2018–2020) |
| `is_1111` | November 11 (Singles Day / 11.11) |

### Two Target Definitions

```python
# "plain" — direct log-scale forecast
y = log1p(Total_Paid)

# "level" — detrended forecast (removes level shifts)
y = log1p(Total_Paid) - log1p(rolling_mean_91_shifted)
```

The `level` target helps tree models handle periods where the payment level drifts far from the training distribution (e.g., post-COVID recovery).

---

## COVID-19 Handling

The test period (Oct–Dec 2020) inherits COVID-era values via lag features:

```
Test day: 2020-10-01
  lag_182  → references: ~2020-04-02  ← COVID peak
  roll_91  → covers:     2020-07-02 to 2020-09-30  ← partial COVID
```

Three strategies are compared **on the same validation rows** to ensure fairness:

| Mode | What it does | Best when |
|---|---|---|
| `include` | COVID rows treated normally | Low impact on merchant |
| `flag` | Adds `is_covid` feature; test uses flag=0 | Model should learn the regime shift |
| `exclude` | COVID rows set to NaN; excluded from targets **and** lag sources | High impact; cleaner lag features needed |

```python
# Configure at the top of the script
COVID_START = pd.Timestamp("2020-03-11")
COVID_END   = pd.Timestamp("2020-06-01")
COVID_MODES = ["include", "flag", "exclude"]   # all three compared
```

Ablation results are saved to `outputs/covid_ablation.csv`. Negative difference = that mode reduces error vs. `include`.

> **Note:** Only Fold B (Jul–Sep 2020) is affected by COVID mode. Fold A (Oct–Dec 2019) gives identical scores across all three modes. Don't over-interpret small differences.

---

## Model Selection

All 24 candidates (6 baselines + 18 LightGBM variants) are ranked by **2-fold average Macro SMAPE**:

```
Macro SMAPE = mean of per-merchant SMAPE values
```

### Baselines

| Baseline | Feature used |
|---|---|
| Lag 92 | `paid_lag_92` |
| Lag 98 | `paid_lag_98` |
| Lag 364 | `paid_lag_364` |
| Lag 366 | `paid_lag_366` |
| Rolling Mean 28 | `paid_roll_mean_28` |
| Rolling Mean 91 | `paid_roll_mean_91` |

### Tolerance Rule

```python
SELECTION_TOLERANCE = 0.10   # SMAPE points

# Among candidates within 0.10 pts of the best score,
# pick the SIMPLEST one.
near = mean_scores[mean_scores <= best_score + SELECTION_TOLERANCE]
chosen = min(near, key=complexity_score)
```

Complexity order: `baseline < LightGBM(no holidays) < +holidays < +BF/11.11`

This prevents selecting a complex model when a simple one is statistically indistinguishable.

### Early Stopping Warning

If LightGBM stops before **20 iterations**, it means validation loss rises immediately after the first trees — the model is effectively making a constant prediction:

```
val l2: first 0.0412 → best 0.0412   ← WARN: very few trees
```

Causes: distribution shift between train and validation, or a target too noisy for the features. In this case the tolerance rule typically selects a baseline automatically.

---

## Outputs & Figures

### CSV Outputs

| File | Description |
|---|---|
| `iyzico_test_forecast.csv` | Merchant × date × predicted_total_paid for Oct–Dec 2020 |
| `validation_results.csv` | SMAPE / Macro SMAPE / WAPE / MAE for every candidate × fold |
| `holiday_ablation.csv` | Holiday feature group contribution vs. "no holidays" baseline |
| `covid_ablation.csv` | COVID mode contribution vs. `include` baseline |

### Figures

| File | What you see |
|---|---|
| `01_monthly_volume.png` | Monthly Total_Paid per merchant, COVID shaded |
| `02_yearly_seasonality.png` | Day-of-year patterns, one line per year |
| `03_weekday_heatmap.png` | Relative daily payment intensity (Mon–Sun) |
| `04_data_availability.png` | Calendar heatmap of observed-day coverage |
| `05_model_comparison.png` | Bar chart of all candidates by Macro SMAPE |
| `06_validation_*.png` | Actual vs. predicted for the chosen model, per fold |
| `07_feature_importance.png` | Top-15 features by LightGBM gain |
| `08_test_forecast.png` | Final Oct–Dec 2020 forecast with history |
| `09_learning_curves_*.png` | Validation L2 curves with early-stopping markers |

---

## Configuration Reference

All settings live at the top of `iyzico_forecasting.py`:

```python
# ── Data ─────────────────────────────────────────────────────────────
DATA_PATH_OVERRIDE = None          # set to Path(r"C:\...\iyzico_data.csv") if needed
DATA_FILE_NAME     = "iyzico_data.csv"

# ── Forecast horizon ─────────────────────────────────────────────────
HORIZON        = 92
LAST_OBSERVED  = pd.Timestamp("2020-09-30")
FORECAST_END   = pd.Timestamp("2020-12-31")

# ── Validation folds ─────────────────────────────────────────────────
FOLDS = {
    "Fold A (Oct-Dec 2019)": ("2019-10-01", "2019-12-31"),
    "Fold B (Jul-Sep 2020)": ("2020-07-01", "2020-09-30"),
}

# ── COVID anomaly ────────────────────────────────────────────────────
COVID_START = pd.Timestamp("2020-03-11")
COVID_END   = pd.Timestamp("2020-06-01")
COVID_MODES = ["include", "flag", "exclude"]

# ── Model selection ──────────────────────────────────────────────────
SELECTION_TOLERANCE = 0.10         # SMAPE points

# ── Display ──────────────────────────────────────────────────────────
SHOW_PLOTS = True                  # set False on headless servers
```

---

## Requirements

```
pandas>=1.5
numpy>=1.23
lightgbm>=4.0
matplotlib>=3.6
```

Full list: `requirements.txt`

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- Holiday dates verified against official Turkish government calendars and news archives
- Pipeline design inspired by Makridakis et al. (M5 competition), with a leakage-safe lag architecture
- COVID period definition (March 11 – June 1, 2020) aligns with Turkey's initial lockdown phase
