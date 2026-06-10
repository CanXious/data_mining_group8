# Data Mining Group 8

This repository contains the analysis workflow and outputs for the project:

**Predicting Building Energy Use Intensity in Chicago via Spatio-Temporal Data Mining**

The project uses the Chicago Energy Benchmarking dataset for 2018-2023 to predict annual building-level `Site EUI (kBtu/sq ft)`.

## Contents

- `model.py`: main entry point for rerunning the analysis.
- `src/run_energy_benchmarking_analysis.py`: full data download, preprocessing, feature engineering, model training, random search tuning, evaluation, plots, and report generation.
- `data/processed/chicago_energy_modeling_table_2018_2023.csv`: cleaned modeling table after filtering, outlier handling, leakage removal, and feature engineering.
- `outputs/*.csv`: model metrics, random search results, predictions, feature importance, and group-level error analysis.
- `outputs/plots/*.png`: evaluation and analysis figures.
- `reports/chicago_energy_benchmarking_report.md`: final written report.

## Method Summary

- Dataset: Chicago Energy Benchmarking.
- Scope: 2018-2023.
- Target: `site_eui_kbtu_sq_ft`.
- Split:
  - Train: 2018-2021
  - Validation: 2022
  - Test: 2023
- Baselines:
  - Global Mean Baseline
  - Community Area + Building Type Group Mean Baseline
  - Ridge Regression
- Models:
  - Random Forest Regressor
  - XGBoost Regressor
- Tuning: random search on the validation set.
- Metrics: RMSE, MAE, and R-squared.

## Key Test Results

| Model | RMSE | MAE | R-squared |
|---|---:|---:|---:|
| Random Forest | 19.4437 | 13.1277 | 0.7908 |
| XGBoost | 20.1889 | 13.8336 | 0.7745 |
| Ridge Regression | 22.2865 | 16.1149 | 0.7252 |
| Group Mean Baseline | 29.0747 | 20.4876 | 0.5323 |
| Global Mean Baseline | 43.3843 | 29.9623 | -0.0413 |

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full workflow:

```bash
python model.py
```

The script downloads raw data from the Chicago Data Portal, rebuilds the cleaned modeling table, trains models, and regenerates outputs.
