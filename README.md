# Chicago Building Energy EUI Modeling

This repository contains the modeling code and the raw/processed data used for the Chicago Energy Benchmarking data-mining experiments.

The repository is intentionally limited to:

- model and experiment code
- raw input data used by the scripts
- processed modeling tables and external-feature tables

It intentionally excludes generated reports, PDFs, plots, logs, cached bytecode, and broad intermediate output folders.

## Project Scope

- Task: predict annual building-level `Site EUI (kBtu/sq ft)`
- Unit of analysis: building-year record
- Main source: Chicago Energy Benchmarking
- Years: 2018-2023
- Train: 2018-2021
- Validation: 2022
- Test: 2023
- Metrics: RMSE, MAE, R-squared

## Repository Structure

```text
code/
  base_model/                  base workflow copy
  external_ablation/           weather, footprints, permits, socioeconomic, ACS experiments
  aligned_experiments/         complete-aligned and footprint-aligned experiments
  high_eui_experiments/        high-EUI and two-stage experiments
src/                           runnable source tree
model.py                       base workflow entry point

data/
  raw/
    base_energy/
    weather/
    footprints/
    permits/
    socioeconomic/
  processed/
    base_model/
    external_features/
    aligned_experiments/
```

Use root-level `src/` for execution because the scripts expect the repository root to be one level above `src/`. The experiment-specific folders under `code/` duplicate the same scripts by analysis topic for easier review.

## Experiments

| Experiment | Runnable script |
|---|---|
| Base model | `python model.py` |
| Weather, footprints, permits ablation | `python src/compare_external_features.py` |
| All external features | `python src/run_all_external_feature_comparison.py` |
| Community-area socioeconomic external ablation | `python src/run_socioeconomic_occupancy_comparison.py` |
| ACS tract socioeconomic external ablation | `python src/run_acs_tract_socioeconomic_occupancy_comparison.py` |
| Weather subgroup error analysis | `python src/analyze_weather_subgroup_errors.py` |
| Complete aligned external comparison | `python src/run_aligned_external_feature_comparison.py` |
| Footprint-aligned weather + footprints comparison | `python src/run_footprint_aligned_weather_footprints.py` |
| High-EUI weather experiment | `python src/run_high_eui_weather_experiment.py` |
| Weather improvement experiments | `python src/run_weather_improvement_experiments.py` |
| Two-stage high-EUI experiment | `python src/run_two_stage_high_eui_experiment.py` |

## Data

Raw data includes cached source extracts from:

- Chicago Energy Benchmarking
- Open-Meteo historical weather
- Chicago Building Footprints
- Chicago Building Permits
- ACS/Census tract data and Chicago selected socioeconomic indicators

Processed data includes:

- base modeling table
- annual weather features
- ACS tract lookup and socioeconomic features
- complete-aligned external modeling table
- footprint-aligned modeling table

The permits raw CSV is intentionally included because permit-history experiments depend on it. It is the largest file in the repository and remains below GitHub's 100 MB single-file limit.

## Setup

```bash
pip install -r requirements.txt
```

Run commands from the repository root.

## Notes

The main model code no longer requires the local report-generation module. If that module is unavailable, the model still writes metrics, predictions, feature importance, and plots.
