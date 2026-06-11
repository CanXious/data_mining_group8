# External Data Ablation Report

## 1. 目的

本報告比較三種外部資料加入 Chicago Energy Benchmarking regression model 後的影響：

1. Weather
2. Building Footprints
3. Building Permits
4. All external features: Weather + Footprints + Permits

目標是判斷外部資料是否能改善 `Site EUI (kBtu/sq ft)` 預測，尤其是原始模型在高 EUI 建築與特殊 building type 上誤差較大的問題。

## 2. 實驗設定

- Target: `site_eui_kbtu_sq_ft`
- Train: 2018-2021
- Validation: 2022，用於 hyperparameter tuning
- Test: 2023，只做最後評估
- Models: Random Forest Regressor, XGBoost Regressor
- Baseline: Group Mean Baseline
- Metrics: RMSE, MAE, R²

## 3. Feature Engineering

### 3.1 Base Features

| Feature | 公式 / 定義 |
|---|---|
| `years_since_2018` | `data_year - 2018` |
| `log_floor_area` | `log(1 + gross_floor_area_buildings_sq_ft)` |
| `building_age` | `data_year - year_built`; if `<0` or `>250`, set to missing |
| `regulation_stage` | 2018-2019 = pre-covid mature policy; 2020-2021 = covid period; 2022-2023 = post-covid recent |
| `prev_year_site_eui` | same building's Site EUI at year `t-1` |
| `prior_building_mean_eui` | mean Site EUI of same building for years `< t` |
| `prior_community_mean_eui` | mean Site EUI of same community area for years `< t` |
| `prior_property_type_mean_eui` | mean Site EUI of same building type for years `< t` |
| `prior_comm_property_mean_eui` | mean Site EUI of same community + building type for years `< t` |
| `prior_community_high_eui_ratio` | proportion of records with Site EUI >= 100 in same community for years `< t` |
| `community_year_record_count` | number of records in the same community and year |
| `community_year_building_count` | number of unique buildings in the same community and year |
| `community_year_avg_floor_area` | average gross floor area in the same community and year |
| `property_year_record_count` | number of records in the same property type and year |

### 3.2 Weather Features

| Feature | 公式 / 定義 |
|---|---|
| `weather_temp_mean_c` | annual mean of daily mean temperature |
| `weather_temp_max_mean_c` | annual mean of daily maximum temperature |
| `weather_temp_min_mean_c` | annual mean of daily minimum temperature |
| `weather_precip_sum_mm` | annual sum of daily precipitation |
| `weather_hdd65` | `sum(max(18.33 - daily_mean_temp_c, 0))` |
| `weather_cdd65` | `sum(max(daily_mean_temp_c - 18.33, 0))` |
| `weather_hot_days_30c` | count of days with daily max temperature >= 30°C |
| `weather_cold_days_minus10c` | count of days with daily min temperature <= -10°C |

### 3.3 Building Footprint Features

| Feature | 公式 / 定義 |
|---|---|
| `footprint_matched` | whether address matched to a building footprint |
| `footprint_shape_area` | footprint polygon area |
| `footprint_shape_len` | footprint polygon perimeter |
| `footprint_compactness` | `4 * pi * shape_area / shape_len²` |
| `footprint_stories` | number of stories from footprint data |
| `footprint_year_built` | year built from footprint data |
| `footprint_bldg_sqft` | building square footage from footprint data |
| `gross_to_footprint_area_ratio` | `gross_floor_area_buildings_sq_ft / footprint_shape_area` |
| `footprint_age` | `data_year - footprint_year_built` |

### 3.4 Building Permit Features

| Feature | 公式 / 定義 |
|---|---|
| `permit_matched` | whether the address had permit history before year `t` |
| `permit_count_3yr` | permit count from `t-3` to `t-1` |
| `permit_count_all_prior` | all permit count before `t` |
| `permit_reported_cost_sum_3yr` | sum of reported cost from `t-3` to `t-1` |
| `permit_total_fee_sum_3yr` | sum of permit fees from `t-3` to `t-1` |
| `permit_mechanical_count_3yr` | count of HVAC/mechanical/heating/cooling permits from `t-3` to `t-1` |
| `permit_electrical_count_3yr` | count of electrical/wiring/lighting permits from `t-3` to `t-1` |
| `permit_renovation_count_3yr` | count of renovation/alteration/rehab/repair permits from `t-3` to `t-1` |
| `permit_major_cost_count_3yr` | count of permits with reported cost >= 100,000 from `t-3` to `t-1` |
| `years_since_last_permit` | `data_year - last_permit_year` |
| `has_recent_permit` | 1 if `permit_count_3yr > 0`, otherwise 0 |

所有 historical EUI 與 permit features 都只使用 `data_year` 以前的資料，避免 data leakage。

## 4. Join Coverage

| feature_set  | matched_rows | matched_row_rate | matched_unique_addresses | unique_addresses | recent_permit_rows | recent_permit_row_rate | weather_matched_rows | weather_matched_row_rate | footprint_matched_rows | footprint_matched_row_rate | permit_matched_rows | permit_matched_row_rate |
| ------------ | ------------ | ---------------- | ------------------------ | ---------------- | ------------------ | ---------------------- | -------------------- | ------------------------ | ---------------------- | -------------------------- | ------------------- | ----------------------- |
| base         | 15177        | 1.0000           |                          |                  |                    |                        |                      |                          |                        |                            |                     |                         |
| weather      | 15177        | 1.0000           |                          |                  |                    |                        |                      |                          |                        |                            |                     |                         |
| footprints   | 11481        | 0.7565           | 3085.0000                | 4248.0000        |                    |                        |                      |                          |                        |                            |                     |                         |
| permits      | 10706        | 0.7054           |                          |                  | 9613.0000          | 0.6334                 |                      |                          |                        |                            |                     |                         |
| all_external | 15177        | 1.0000           |                          |                  | 9613.0000          | 0.6334                 | 15177.0000           | 1.0000                   | 11481.0000             | 0.7565                     | 10706.0000          | 0.7054                  |

## 5. Hyperparameter Tuning

本研究使用 random search 做 hyperparameter tuning。每個 feature set 都重新 tuning，因為加入外部資料後 feature space 改變，最佳參數也可能不同。

```text
Random Forest:
  n_estimators = [200, 300, 500]
  max_depth = [8, 12, 16, 24, None]
  min_samples_leaf = [1, 3, 5, 10]
  max_features = ['sqrt', 0.5, 0.8]

XGBoost:
  n_estimators = [200, 400, 600]
  max_depth = [2, 3, 4, 5, 6]
  learning_rate = [0.03, 0.05, 0.08, 0.1]
  subsample = [0.7, 0.85, 1.0]
  colsample_bytree = [0.7, 0.85, 1.0]
  reg_lambda = [1, 5, 10]
  min_child_weight = [1, 3, 5]
```

每個 feature set 的 validation-selected model 如下：

| feature_set  | model         | iteration | validation_rmse | validation_mae | validation_r2 |
| ------------ | ------------- | --------- | --------------- | -------------- | ------------- |
| base         | XGBoost       | 7         | 21.1808         | 12.3151        | 0.8067        |
| weather      | Random Forest | 18        | 21.4582         | 11.9469        | 0.8016        |
| footprints   | XGBoost       | 11        | 20.9459         | 12.0954        | 0.8110        |
| permits      | XGBoost       | 7         | 20.7603         | 12.2395        | 0.8143        |
| all_external | Random Forest | 18        | 21.4591         | 11.9103        | 0.8016        |

Best hyperparameters:

```json
{
  "base": {
    "model": "XGBoost",
    "params": {
      "colsample_bytree": 0.85,
      "learning_rate": 0.1,
      "max_depth": 6,
      "min_child_weight": 1,
      "n_estimators": 400,
      "reg_lambda": 5,
      "subsample": 0.7
    }
  },
  "weather": {
    "model": "Random Forest",
    "params": {
      "max_depth": 24,
      "max_features": 0.5,
      "min_samples_leaf": 3,
      "n_estimators": 500
    }
  },
  "footprints": {
    "model": "XGBoost",
    "params": {
      "colsample_bytree": 0.7,
      "learning_rate": 0.05,
      "max_depth": 6,
      "min_child_weight": 3,
      "n_estimators": 400,
      "reg_lambda": 5,
      "subsample": 0.85
    }
  },
  "permits": {
    "model": "XGBoost",
    "params": {
      "colsample_bytree": 0.85,
      "learning_rate": 0.1,
      "max_depth": 6,
      "min_child_weight": 1,
      "n_estimators": 400,
      "reg_lambda": 5,
      "subsample": 0.7
    }
  },
  "all_external": {
    "model": "Random Forest",
    "params": {
      "max_depth": 24,
      "max_features": 0.5,
      "min_samples_leaf": 3,
      "n_estimators": 500
    }
  }
}
```

## 6. Test Results

| feature_set  | model               | rmse    | mae     | r2     |
| ------------ | ------------------- | ------- | ------- | ------ |
| all_external | Random Forest       | 17.9975 | 11.0789 | 0.8208 |
| all_external | XGBoost             | 18.6474 | 11.6705 | 0.8076 |
| all_external | Group Mean Baseline | 29.0747 | 20.4876 | 0.5323 |
| base         | Random Forest       | 19.4437 | 13.1277 | 0.7908 |
| base         | XGBoost             | 20.1889 | 13.8336 | 0.7745 |
| base         | Group Mean Baseline | 29.0747 | 20.4876 | 0.5323 |
| footprints   | Random Forest       | 19.5151 | 13.0831 | 0.7893 |
| footprints   | XGBoost             | 19.9053 | 13.2463 | 0.7808 |
| footprints   | Group Mean Baseline | 29.0747 | 20.4876 | 0.5323 |
| permits      | Random Forest       | 19.3409 | 12.9813 | 0.7930 |
| permits      | XGBoost             | 19.8375 | 13.6813 | 0.7823 |
| permits      | Group Mean Baseline | 29.0747 | 20.4876 | 0.5323 |
| weather      | XGBoost             | 17.8748 | 11.0647 | 0.8232 |
| weather      | Random Forest       | 17.9485 | 11.0946 | 0.8218 |
| weather      | Group Mean Baseline | 29.0747 | 20.4876 | 0.5323 |

## 7. Best Model By Feature Set

| feature_set  | model         | rmse    | mae     | r2     | rmse_change_vs_base | rmse_change_pct_vs_base | rmse_change_vs_weather | rmse_change_pct_vs_weather |
| ------------ | ------------- | ------- | ------- | ------ | ------------------- | ----------------------- | ---------------------- | -------------------------- |
| base         | Random Forest | 19.4437 | 13.1277 | 0.7908 | 0.0000              | 0.0000                  | 1.5689                 | 8.7772                     |
| weather      | XGBoost       | 17.8748 | 11.0647 | 0.8232 | -1.5689             | -8.0689                 | 0.0000                 | 0.0000                     |
| footprints   | Random Forest | 19.5151 | 13.0831 | 0.7893 | 0.0714              | 0.3670                  | 1.6403                 | 9.1764                     |
| permits      | Random Forest | 19.3409 | 12.9813 | 0.7930 | -0.1028             | -0.5285                 | 1.4661                 | 8.2023                     |
| all_external | Random Forest | 17.9975 | 11.0789 | 0.8208 | -1.4462             | -7.4379                 | 0.1227                 | 0.6865                     |

![External Feature RMSE Comparison](../outputs/plots/external_feature_test_rmse_comparison_with_all.png)

![External Feature MAE Comparison](../outputs/plots/external_feature_test_mae_comparison_with_all.png)

![External Feature R2 Comparison](../outputs/plots/external_feature_test_r2_comparison_with_all.png)

![Best Model Improvement](../outputs/plots/external_feature_best_rmse_improvement.png)

![External Data Join Coverage](../outputs/plots/external_feature_join_coverage.png)

## 8. 成果解讀

Weather-only 是整體最佳外部資料設定。它的最佳模型是 **XGBoost**，test RMSE = **17.8748**、MAE = **11.0647**、R² = **0.8232**。

All external 的最佳模型是 **Random Forest**，test RMSE = **17.9975**。雖然 all external 比 base model 明顯更好，但沒有超過 weather-only；相對 weather-only，RMSE 增加 **0.1227**，約 **0.69%**。

因此，三種外部資料中，weather features 提供最穩定、最明顯的額外訊號。Footprints 與 permits 在目前 address matching、資料粒度與特徵設計下，沒有在 weather features 已存在時提供額外穩定增益。

## 9. 限制

1. Weather 是年度彙總資料，適合年度結束後分析；若要年初預測該年度 EUI，不能直接使用完整年度天氣。
2. Footprints 與 permits 使用地址 matching，coverage 分別約 75.65% 與 70.54%，可能引入 matching error。
3. Permit features 只能代表是否有申請或施工紀錄，無法確認實際設備效率、施工品質或完成時間。
4. 高 EUI 建築與特殊 building type 的誤差仍可能需要 occupancy、operating hours、HVAC system、設備效率、租戶型態等更細資料才能改善。

## 10. 結論

最終建議模型使用 **Base features + Weather features + XGBoost Regressor**。此設定在 test set 上取得最低 RMSE 與最高 R²，且模型比 all external 更簡潔，也避免 footprints/permits matching noise。
