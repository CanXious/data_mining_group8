# Model Features and Formulas

本文件整理本專案實際建模使用的 features 與公式。內容以程式實作為準，主要來源為：

- `src/run_energy_benchmarking_analysis.py`
- `src/compare_external_features.py`
- `src/run_all_external_feature_comparison.py`

## 1. Modeling Target and Data Scope

### Target Variable

模型預測目標為：

```text
y = site_eui_kbtu_sq_ft
```

也就是 Chicago Energy Benchmarking dataset 中的：

```text
Site EUI (kBtu/sq ft)
```

這是一個連續數值，因此本專案是 regression task。

### Main Dataset

原始主資料集：

```text
Chicago Energy Benchmarking
https://data.cityofchicago.org/resource/xq83-jr8c.json
```

使用範圍：

```text
data_year = 2018-2023
```

資料篩選規則：

```text
reporting_status in {"Submitted", "Submitted Data"}
site_eui_kbtu_sq_ft is not null
gross_floor_area_buildings_sq_ft is not null
community_area is not null
primary_property_type is not null
latitude is not null
longitude is not null
1 <= site_eui_kbtu_sq_ft <= 500
```

篩選後資料量：

```text
Total rows: 15,177
Distinct buildings: 3,312
Train: 2018-2021, 10,130 rows
Validation: 2022, 2,468 rows
Test: 2023, 2,579 rows
```

## 2. Feature Sets Used in Experiments

本專案實際比較了五組 feature set：

| Feature Set | Description |
|---|---|
| `base` | 只使用 Chicago Energy Benchmarking 原始資料與衍生特徵 |
| `weather` | `base` + Open-Meteo weather features |
| `footprints` | `base` + Chicago Building Footprints features |
| `permits` | `base` + Chicago Building Permits features |
| `all_external` | `base` + weather + footprints + permits |

## 3. Base Features

以下 features 來自 Chicago Energy Benchmarking 主資料集。

### 3.1 Raw Numeric Features

| Feature | Formula / Definition |
|---|---|
| `data_year` | 原始資料年份 |
| `gross_floor_area_buildings_sq_ft` | 原始建築樓地板面積 |
| `year_built` | 原始建造年份 |
| `of_buildings` | 原始建築棟數，實作中小於 1 的值會被 clip 到 1 |
| `latitude` | 原始緯度 |
| `longitude` | 原始經度 |

### 3.2 Raw Categorical Features

| Feature | Processing |
|---|---|
| `community_area_clean` | `community_area` 去除空白並轉大寫 |
| `primary_property_type_clean` | `primary_property_type` 去除空白 |
| `zip_code_clean` | `zip_code` 去除空白，缺失值設為 `UNKNOWN` |

類別特徵在模型 pipeline 中使用 one-hot encoding：

```text
category = k  ->  one_hot_k = 1, other categories = 0
```

### 3.3 Time Features

| Feature | Formula |
|---|---|
| `years_since_2018` | `data_year - 2018` |
| `regulation_stage` | 2018-2019: `pre_covid_mature_policy`; 2020-2021: `covid_period`; 2022-2023: `post_covid_recent` |

### 3.4 Building Attribute Features

| Feature | Formula |
|---|---|
| `log_floor_area` | `ln(1 + max(gross_floor_area_buildings_sq_ft, 0))` |
| `building_age` | `data_year - year_built` |

`building_age` 若小於 0 或大於 250，會設為 missing value。

### 3.5 Building-Level Historical EUI Features

這些 feature 只使用同一棟建築物在當年以前的 EUI，避免使用未來資料。

| Feature | Formula |
|---|---|
| `prev_year_site_eui` | 若同一建築上一筆資料年份剛好是 `t-1`，則為 `Site EUI(i, t-1)`；否則為 missing |
| `prior_building_mean_eui` | 同一建築在所有過去年份的平均 EUI |

公式：

```text
prev_year_site_eui(i, t) =
    Site EUI(i, t-1), if previous observed year = t-1
    missing, otherwise
```

```text
prior_building_mean_eui(i, t)
    = mean(Site EUI(i, tau)) for all tau < t
```

其中 `i` 表示 building id，`t` 表示 data year。

### 3.6 Historical Group Mean Features

這些 feature 使用同一 group 在「過去年份」的歷史平均值，不使用當年 target。

| Feature | Formula |
|---|---|
| `prior_community_mean_eui` | 同一 community 在過去年份的平均 EUI |
| `prior_property_type_mean_eui` | 同一 property type 在過去年份的平均 EUI |
| `prior_comm_property_mean_eui` | 同一 community + property type 在過去年份的平均 EUI |

公式：

```text
prior_community_mean_eui(c, t)
    = sum(Site EUI for community c and year tau < t)
      / count(records for community c and year tau < t)
```

```text
prior_property_type_mean_eui(p, t)
    = sum(Site EUI for property type p and year tau < t)
      / count(records for property type p and year tau < t)
```

```text
prior_comm_property_mean_eui(c, p, t)
    = sum(Site EUI for community c, property type p, and year tau < t)
      / count(records for community c, property type p, and year tau < t)
```

其中：

```text
c = community area
p = primary property type
t = data year
tau = previous year
```

### 3.7 Historical High-EUI Ratio Feature

程式中定義：

```text
high_eui_flag = 1 if Site EUI >= 100 else 0
```

| Feature | Formula |
|---|---|
| `prior_community_high_eui_ratio` | 同一 community 過去年份中高 EUI 建築比例 |

公式：

```text
prior_community_high_eui_ratio(c, t)
    = sum(high_eui_flag for community c and year tau < t)
      / count(records for community c and year tau < t)
```

### 3.8 Same-Year Group Structure Features

這些 feature 描述同一年、同一 group 的樣本結構。

| Feature | Formula |
|---|---|
| `community_year_record_count` | `count(records in community c at year t)` |
| `community_year_building_count` | `count(unique building ids in community c at year t)` |
| `community_year_avg_floor_area` | `mean(gross_floor_area_buildings_sq_ft in community c at year t)` |
| `property_year_record_count` | `count(records in property type p at year t)` |

公式：

```text
community_year_record_count(c, t)
    = number of records where community = c and data_year = t
```

```text
community_year_building_count(c, t)
    = number of unique building ids where community = c and data_year = t
```

```text
community_year_avg_floor_area(c, t)
    = mean(gross_floor_area_buildings_sq_ft for community c and year t)
```

```text
property_year_record_count(p, t)
    = number of records where property type = p and data_year = t
```

## 4. External Dataset 1: Weather Features

### Data Source

補充資料集：

```text
Open-Meteo Historical Weather API
https://archive-api.open-meteo.com/v1/archive
```

使用位置：

```text
Chicago city center
latitude = 41.8781
longitude = -87.6298
```

使用日期：

```text
2018-01-01 to 2023-12-31
```

原始 daily variables：

```text
temperature_2m_mean
temperature_2m_max
temperature_2m_min
precipitation_sum
```

Join key：

```text
data_year
```

Weather features 對所有 modeling rows 都能 join 到：

```text
matched rows = 15,177
matched row rate = 100%
```

### Weather Feature Formulas

Base temperature：

```text
base_temp_c = (65 - 32) * 5 / 9 = 18.3333 Celsius
```

Daily heating degree day：

```text
hdd65_d = max(base_temp_c - temperature_2m_mean_d, 0)
```

Daily cooling degree day：

```text
cdd65_d = max(temperature_2m_mean_d - base_temp_c, 0)
```

Annual weather features：

| Feature | Formula |
|---|---|
| `weather_temp_mean_c` | `mean(temperature_2m_mean_d)` within year |
| `weather_temp_max_mean_c` | `mean(temperature_2m_max_d)` within year |
| `weather_temp_min_mean_c` | `mean(temperature_2m_min_d)` within year |
| `weather_precip_sum_mm` | `sum(precipitation_sum_d)` within year |
| `weather_hdd65` | `sum(hdd65_d)` within year |
| `weather_cdd65` | `sum(cdd65_d)` within year |
| `weather_hot_days_30c` | `sum(1 if temperature_2m_max_d >= 30 else 0)` within year |
| `weather_cold_days_minus10c` | `sum(1 if temperature_2m_min_d <= -10 else 0)` within year |

其中 `d` 表示 daily record。

## 5. External Dataset 2: Building Footprints Features

### Data Source

補充資料集：

```text
Chicago Building Footprints
https://data.cityofchicago.org/resource/syp8-uezg.json
```

使用原始欄位：

```text
bldg_id
f_add1
t_add1
pre_dir1
st_name1
st_type1
stories
no_stories
year_built
bldg_sq_fo
shape_area
shape_len
```

使用條件：

```text
f_add1 > 0
st_name1 is not null
```

### Address Matching Rule

主資料與 footprint 資料都會先做 address normalization。

主資料 address 會被拆成：

```text
address_number
street_key = direction | street_name | street_type
loose_street_key = direction | street_name
address_key = address_number | direction | street_name | street_type
```

Footprint address 會被拆成：

```text
fp_address_number
fp_street_key
fp_loose_street_key
fp_address_key
```

優先使用 `street_key` match；如果找不到，再使用 `loose_street_key` match。

Footprint match 條件：

```text
f_add1 <= address_number <= t_add1
```

如果找不到 range match，則嘗試：

```text
f_add1 == address_number
```

Coverage：

```text
matched rows = 11,481
matched row rate = 75.65%
matched unique addresses = 3,085
unique addresses = 4,248
```

### Footprint Feature Formulas

| Feature | Formula / Definition |
|---|---|
| `footprint_matched` | 1 if footprint was matched, otherwise 0 |
| `footprint_shape_area` | matched footprint `shape_area` |
| `footprint_shape_len` | matched footprint `shape_len` |
| `footprint_compactness` | `4 * pi * shape_area / shape_len^2` |
| `footprint_stories` | `stories` if positive, otherwise `no_stories` |
| `footprint_year_built` | matched footprint `year_built` |
| `footprint_bldg_sqft` | matched footprint `bldg_sq_fo` |
| `gross_to_footprint_area_ratio` | `gross_floor_area_buildings_sq_ft / footprint_shape_area` |
| `footprint_age` | `data_year - footprint_year_built` |

Invalid `footprint_age` values are set to missing:

```text
footprint_age < 0 or footprint_age > 250  ->  missing
```

## 6. External Dataset 3: Building Permits Features

### Data Source

補充資料集：

```text
Chicago Building Permits
https://data.cityofchicago.org/resource/ydr8-5enu.json
```

使用日期：

```text
issue_date between 2015-01-01 and 2022-12-31
```

使用原始欄位：

```text
id
permit_
permit_status
permit_type
issue_date
street_number
street_direction
street_name
work_type
work_description
reported_cost
total_fee
latitude
longitude
```

Join key：

```text
address_key + data_year
```

Coverage：

```text
permit matched rows = 10,706
permit matched row rate = 70.54%
recent permit rows = 9,613
recent permit row rate = 63.34%
```

### Permit Text Flags

Permit text is created by concatenating:

```text
work_type + work_description + permit_type
```

Then converted to uppercase.

| Flag | Definition |
|---|---|
| `mechanical_flag` | text contains `MECHANICAL`, `HVAC`, `VENTILATION`, `REFRIGERATION`, `FURNACE`, `BOILER`, `AIR CONDITION`, `COOLING`, or `HEATING` |
| `electrical_flag` | text contains `ELECTRIC`, `WIRING`, `POWER`, or `LIGHTING` |
| `renovation_flag` | text contains `RENOVATION`, `ALTERATION`, `REHAB`, `REMODEL`, or `REPAIR` |
| `major_cost_flag` | `reported_cost >= 100000` |

### Annual Permit Aggregation

First, permits are aggregated by:

```text
address_key + issue_year
```

Annual raw aggregations:

```text
permit_count = count(permit records)
reported_cost_sum = sum(reported_cost)
total_fee_sum = sum(total_fee)
mechanical_count = sum(mechanical_flag)
electrical_count = sum(electrical_flag)
renovation_count = sum(renovation_flag)
major_cost_count = sum(major_cost_flag)
```

### Permit Feature Formulas

For each building-year `t`, the recent permit window is:

```text
[t - 3, t - 1]
```

This means the model only uses permits issued before the prediction year.

| Feature | Formula |
|---|---|
| `permit_count_3yr` | `sum(permit_count)` for issue years `t-3` to `t-1` |
| `permit_reported_cost_sum_3yr` | `sum(reported_cost_sum)` for issue years `t-3` to `t-1` |
| `permit_total_fee_sum_3yr` | `sum(total_fee_sum)` for issue years `t-3` to `t-1` |
| `permit_mechanical_count_3yr` | `sum(mechanical_count)` for issue years `t-3` to `t-1` |
| `permit_electrical_count_3yr` | `sum(electrical_count)` for issue years `t-3` to `t-1` |
| `permit_renovation_count_3yr` | `sum(renovation_count)` for issue years `t-3` to `t-1` |
| `permit_major_cost_count_3yr` | `sum(major_cost_count)` for issue years `t-3` to `t-1` |
| `permit_count_all_prior` | `sum(permit_count)` for all issue years `<= t-1` |
| `years_since_last_permit` | `data_year - last_permit_year` |
| `has_recent_permit` | `1 if permit_count_3yr > 0 else 0` |
| `permit_matched` | `1 if permit_count_all_prior > 0 else 0` |

All permit count and cost features are filled with 0 when no matching permit history exists.

## 7. Preprocessing Before Modeling

The model uses a `ColumnTransformer` pipeline.

### Numeric Features

Numeric features are processed by:

```text
median imputation -> standard scaling
```

Median imputation:

```text
x_missing = median(x_train)
```

Standard scaling:

```text
z = (x - mean_train) / std_train
```

### Categorical Features

Categorical features are processed by:

```text
most frequent imputation -> one-hot encoding
```

Most frequent imputation:

```text
x_missing = mode(x_train)
```

One-hot encoding:

```text
category k -> binary indicator column
```

## 8. Models Actually Trained

### Baseline

Baseline model:

```text
Group Mean Baseline
```

Prediction fallback order:

```text
1. mean Site EUI for same community_area_clean + primary_property_type_clean
2. mean Site EUI for same primary_property_type_clean
3. mean Site EUI for same community_area_clean
4. global mean Site EUI
```

### Machine Learning Models

實際訓練的主要模型：

```text
RandomForestRegressor
XGBRegressor
```

Tuning method:

```text
Random search
```

Split:

```text
Train: 2018-2021
Validation: 2022
Test: 2023
```

Random Forest search space:

```text
n_estimators: [200, 300, 500]
max_depth: [8, 12, 16, 24, None]
min_samples_leaf: [1, 3, 5, 10]
max_features: ["sqrt", 0.5, 0.8]
n_iter: 20
```

XGBoost search space:

```text
n_estimators: [200, 400, 600]
max_depth: [2, 3, 4, 5, 6]
learning_rate: [0.03, 0.05, 0.08, 0.1]
subsample: [0.7, 0.85, 1.0]
colsample_bytree: [0.7, 0.85, 1.0]
reg_lambda: [1, 5, 10]
min_child_weight: [1, 3, 5]
n_iter: 25
```

## 9. Final Feature List by Dataset

### Base Dataset Features

```text
data_year
years_since_2018
gross_floor_area_buildings_sq_ft
log_floor_area
year_built
building_age
of_buildings
latitude
longitude
prev_year_site_eui
prior_building_mean_eui
prior_community_mean_eui
prior_property_type_mean_eui
prior_comm_property_mean_eui
prior_community_high_eui_ratio
community_year_record_count
community_year_building_count
community_year_avg_floor_area
property_year_record_count
community_area_clean
primary_property_type_clean
zip_code_clean
regulation_stage
```

### Weather Features

```text
weather_temp_mean_c
weather_temp_max_mean_c
weather_temp_min_mean_c
weather_precip_sum_mm
weather_hdd65
weather_cdd65
weather_hot_days_30c
weather_cold_days_minus10c
```

### Building Footprints Features

```text
footprint_matched
footprint_shape_area
footprint_shape_len
footprint_compactness
footprint_stories
footprint_year_built
footprint_bldg_sqft
gross_to_footprint_area_ratio
footprint_age
```

### Building Permits Features

```text
permit_matched
permit_count_3yr
permit_count_all_prior
permit_reported_cost_sum_3yr
permit_total_fee_sum_3yr
permit_mechanical_count_3yr
permit_electrical_count_3yr
permit_renovation_count_3yr
permit_major_cost_count_3yr
years_since_last_permit
has_recent_permit
```

## 10. Notes About Leakage Control

The following fields were excluded from model inputs because they directly describe energy consumption or energy-performance outcomes and could cause data leakage:

```text
energy_star_score
electricity_use_kbtu
natural_gas_use_kbtu
district_steam_use_kbtu
district_chilled_water_use_kbtu
all_other_fuel_use_kbtu
source_eui_kbtu_sq_ft
weather_normalized_site_eui_kbtu_sq_ft
weather_normalized_source_eui_kbtu_sq_ft
total_ghg_emissions_metric_tons_co2e
ghg_intensity_kg_co2e_sq_ft
water_use_kgal
```

Historical EUI features use only records from years earlier than the prediction year. Permit features also use only permits issued before the prediction year.

## 11. Best Test Results for Context

The best result among the compared feature sets was:

```text
Feature set: weather
Model: XGBoost
Test RMSE: 17.8748
Test MAE: 11.0647
Test R2: 0.8232
```

The all-external feature set result was:

```text
Feature set: all_external
Model: Random Forest
Test RMSE: 17.9975
Test MAE: 11.0789
Test R2: 0.8208
```

Interpretation:

```text
Weather features gave the strongest improvement.
Adding footprints and permits together with weather did not further improve over weather-only.
This may be related to address matching noise and incomplete coverage in the footprint and permit joins.
```
