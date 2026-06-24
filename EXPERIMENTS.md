# Experiment Index

## 1. Base Model

Purpose: train the base Chicago Energy Benchmarking model using engineered building, temporal, spatial, and historical EUI features.

Code:

- `src/run_energy_benchmarking_analysis.py`
- `model.py`

Data:

- `data/raw/base_energy/chicago_energy_benchmarking_2018_2023_raw.csv`
- `data/processed/base_model/chicago_energy_modeling_table_2018_2023.csv`

## 2. External Feature Ablation

Purpose: compare the base model against weather, building footprints, building permits, socioeconomic/occupancy variables, ACS tract variables, and combined external feature sets.

Code:

- `src/compare_external_features.py`
- `src/run_all_external_feature_comparison.py`
- `src/run_socioeconomic_occupancy_comparison.py`
- `src/run_acs_tract_socioeconomic_occupancy_comparison.py`

Data:

- `data/raw/weather/open_meteo_chicago_daily_2018_2023.csv`
- `data/raw/footprints/chicago_building_footprints_selected.csv`
- `data/raw/permits/chicago_building_permits_2015_2022_selected.csv`
- `data/raw/socioeconomic/`
- `data/processed/external_features/external_weather_features_2018_2023.csv`
- `data/processed/external_features/external_socioeconomic_occupancy_features_community_area.csv`
- `data/processed/external_features/external_acs_tract_lookup_2018_2023.csv`
- `data/processed/external_features/external_acs_tract_socioeconomic_occupancy_features_2018_2023.csv`

## 3. Aligned External Experiments

Purpose: separate external feature value from missing-coverage effects.

Complete aligned sample:

```text
footprint_matched == 1
permit_matched == 1
```

Footprint-aligned sample:

```text
footprint_matched == 1
```

Code:

- `src/run_aligned_external_feature_comparison.py`
- `src/run_footprint_aligned_weather_footprints.py`

Data:

- `data/processed/aligned_experiments/aligned_external_modeling_table_2018_2023.csv`
- `data/processed/aligned_experiments/footprint_aligned_modeling_table_2018_2023.csv`

## 4. High-EUI Experiments

Purpose: evaluate whether weather features and two-stage modeling improve high-EUI and special building-type errors.

Code:

- `src/analyze_weather_subgroup_errors.py`
- `src/run_high_eui_weather_experiment.py`
- `src/run_weather_improvement_experiments.py`
- `src/run_two_stage_high_eui_experiment.py`

Data:

- `data/processed/base_model/chicago_energy_modeling_table_2018_2023.csv`
- `data/processed/external_features/external_weather_features_2018_2023.csv`
