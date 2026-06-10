# Predicting Building Energy Use Intensity in Chicago via Spatio-Temporal Data Mining

## 1. 研究目標

本專案使用 Chicago Energy Benchmarking 資料，建立一個 regression model 來預測建築物年度 `Site EUI (kBtu/sq ft)`。分析單位是 **building-year record**，也就是一棟建築在某一年的能源表現。研究問題是：芝加哥不同 community area、不同 building type、不同年份的建築能源使用強度是否存在可學習的時空模式。

此任務屬於 spatio-temporal predictive analytics。空間面向來自 `community_area`、`zip_code`、`latitude`、`longitude`；時間面向來自 `data_year`、前一年 EUI、歷史 rolling/aggregate EUI，以及政策成熟階段。

資料來源：[https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking/xq83-jr8c/about_data](https://data.cityofchicago.org/Environment-Sustainable-Development/Chicago-Energy-Benchmarking/xq83-jr8c/about_data)

## 2. 資料範圍與前處理

主實驗選取 **2018-2023**。2014-2017 是芝加哥 benchmarking 政策逐步導入期，樣本數較不穩定，因此沒有納入主模型訓練。2018-2023 原始下載筆數為 **21,051**，完成篩選後剩下 **15,177** 筆、**3,312** 棟建築。

資料篩選與清理規則：

1. 只保留 `data_year` 介於 2018-2023 的紀錄。
2. 只保留 `reporting_status` 為 `Submitted` 或 `Submitted Data` 的紀錄。
3. 移除 `Site EUI`、`gross_floor_area_buildings_sq_ft`、`primary_property_type`、`community_area`、`latitude`、`longitude` 缺失的紀錄。
4. 統一 `community_area` 大小寫，避免 `LOOP` 與 `Loop` 被視為不同區域。
5. Outlier 處理：只保留 `1 <= Site EUI <= 500`。此規則移除 **65** 筆極端值。
6. Leakage 處理：不把當年度能源消耗或能源表現結果類欄位放入模型，包括 `energy_star_score, electricity_use_kbtu, natural_gas_use_kbtu, district_steam_use_kbtu, district_chilled_water_use_kbtu, all_other_fuel_use_kbtu, source_eui_kbtu_sq_ft, weather_normalized_site_eui_kbtu_sq_ft, weather_normalized_source_eui_kbtu_sq_ft, total_ghg_emissions_metric_tons_co2e, ghg_intensity_kg_co2e_sq_ft, water_use_kgal`。這些欄位與目標 `Site EUI` 同時或事後產生，若納入模型會造成不合理的高估表現。

篩選後各年份筆數：

| year | records |
| ---- | ------- |
| 2018 | 2724    |
| 2019 | 2050    |
| 2020 | 2844    |
| 2021 | 2512    |
| 2022 | 2468    |
| 2023 | 2579    |

## 3. 特徵工程

本專案使用的欄位可分為四類：

| 類別 | 欄位 |
|---|---|
| 目標變數 | `site_eui_kbtu_sq_ft` |
| 空間特徵 | `community_area_clean`, `zip_code_clean`, `latitude`, `longitude`, community-year building count, community-year average floor area |
| 時間特徵 | `data_year`, `years_since_2018`, `regulation_stage`, `prev_year_site_eui`, `prior_building_mean_eui` |
| 建物特徵 | `primary_property_type_clean`, `gross_floor_area_buildings_sq_ft`, `log_floor_area`, `year_built`, `building_age`, `of_buildings` |

額外建立的 spatio-temporal features 包含：

1. `prev_year_site_eui`：同一棟建築前一年的 EUI。
2. `prior_building_mean_eui`：同一棟建築過去年份的平均 EUI。
3. `prior_community_mean_eui`：同一 community area 過去年份的平均 EUI。
4. `prior_property_type_mean_eui`：同一 building type 過去年份的平均 EUI。
5. `prior_comm_property_mean_eui`：同一 community area 與 building type 組合的歷史平均 EUI。
6. `prior_community_high_eui_ratio`：同一 community area 過去年份中高 EUI 建築比例。
7. `community_year_record_count`：同一年同 community area 的資料筆數。
8. `community_year_building_count`：同一年同 community area 的建築數。
9. `community_year_avg_floor_area`：同一年同 community area 的平均樓地板面積。
10. `property_year_record_count`：同一年同 building type 的資料筆數。

所有歷史平均與比例都只使用該筆資料年份以前的資訊，避免把同年度或未來資料洩漏到特徵中。

## 4. Baseline 與模型

資料切分方式：

| Split | 年份 | 用途 |
|---|---|---|
| Train | 2018-2021 | 訓練 baseline 與模型 |
| Validation | 2022 | random search fine tuning |
| Test | 2023 | 只做最後評估 |

Baseline：

1. **Global Mean Baseline**：全部預測訓練資料的平均 Site EUI。
2. **Group Mean Baseline**：依 `community_area + primary_property_type` 的歷史平均 Site EUI 預測，若該組合不存在，依序 fallback 到 building type、community、global mean。
3. **Ridge Regression**：線性模型 baseline，類別欄位 one-hot encoding，數值欄位 median imputation 與 standardization。

進階模型：

1. **Random Forest Regressor**
2. **XGBoost Regressor**

Hyperparameter tuning 使用 random search。原因是 tree-based models 的參數組合很多，random search 可以在有限時間內探索較多有效區域，不需要像 grid search 一樣窮舉所有組合。

Model selected by validation RMSE: **XGBoost**

Best parameters:

```json
{
  "colsample_bytree": 0.85,
  "learning_rate": 0.1,
  "max_depth": 6,
  "min_child_weight": 1,
  "n_estimators": 400,
  "reg_lambda": 5,
  "subsample": 0.7
}
```

Random search 最佳結果摘要：

| model         | iteration | validation_rmse | validation_mae | validation_r2 | params                                                                                                                                            |
| ------------- | --------- | --------------- | -------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Random Forest | 12.0000   | 21.7507         | 12.1580        | 0.7962        | {"max_depth": null, "max_features": 0.5, "min_samples_leaf": 3, "n_estimators": 200}                                                              |
| Random Forest | 18.0000   | 21.7510         | 12.1885        | 0.7962        | {"max_depth": 24, "max_features": 0.5, "min_samples_leaf": 3, "n_estimators": 500}                                                                |
| Random Forest | 17.0000   | 21.9144         | 12.4675        | 0.7931        | {"max_depth": 12, "max_features": 0.5, "min_samples_leaf": 3, "n_estimators": 200}                                                                |
| XGBoost       | 7.0000    | 21.1808         | 12.3151        | 0.8067        | {"colsample_bytree": 0.85, "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 1, "n_estimators": 400, "reg_lambda": 5, "subsample": 0.7}   |
| XGBoost       | 12.0000   | 21.2381         | 12.2883        | 0.8057        | {"colsample_bytree": 0.85, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 3, "n_estimators": 400, "reg_lambda": 10, "subsample": 0.7} |
| XGBoost       | 11.0000   | 21.2705         | 12.3406        | 0.8051        | {"colsample_bytree": 0.7, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 3, "n_estimators": 400, "reg_lambda": 5, "subsample": 0.85}  |

## 5. 模型評估結果

Validation results:

| model                | rmse    | mae     | r2      |
| -------------------- | ------- | ------- | ------- |
| XGBoost              | 21.1808 | 12.3151 | 0.8067  |
| Random Forest        | 21.7507 | 12.1580 | 0.7962  |
| Ridge Regression     | 23.2126 | 16.2411 | 0.7679  |
| Group Mean Baseline  | 34.1812 | 23.1816 | 0.4967  |
| Global Mean Baseline | 48.2076 | 31.5460 | -0.0011 |

Test results:

| model                | rmse    | mae     | r2      |
| -------------------- | ------- | ------- | ------- |
| Random Forest        | 19.4437 | 13.1277 | 0.7908  |
| XGBoost              | 20.1889 | 13.8336 | 0.7745  |
| Ridge Regression     | 22.2865 | 16.1149 | 0.7252  |
| Group Mean Baseline  | 29.0747 | 20.4876 | 0.5323  |
| Global Mean Baseline | 43.3843 | 29.9623 | -0.0413 |

依 validation RMSE 選出的模型 **XGBoost** 在 test set 相對於 Group Mean Baseline：

- RMSE 改善約 **30.56%**
- MAE 改善約 **32.48%**

Test set 中 RMSE 最低的模型是 **Random Forest**，相對於 Group Mean Baseline：

- RMSE 改善約 **33.12%**
- MAE 改善約 **35.92%**

需要注意的是，模型選擇應以 validation set 為準，不能因為 test set 結果再回頭選模型或調參。因此本報告同時呈現 validation-selected model 與 test-best model，並把 test set 視為最終泛化能力檢查。

本次輸出的評估圖表：

- `outputs/plots/actual_vs_predicted_test.png`：實際值 vs 預測值。
- `outputs/plots/residual_analysis_test.png`：residual vs prediction 與 residual distribution。
- `outputs/plots/community_error_mae_top15.png`：test set 中 community area 的 MAE 比較。
- `outputs/plots/building_type_error_mae_top15.png`：test set 中 building type 的 MAE 比較。
- `outputs/plots/feature_importance_top20.png`：最佳 tree model 的 feature importance。

## 6. Community Area 誤差分析

以下表格列出 test set 中平均 EUI 較高的 community area。`bias_mean_pred_minus_actual` 若為負值，代表模型平均低估該區 EUI。

| community_area_clean | n      | actual_mean_eui | mae    | bias_mean_pred_minus_actual |
| -------------------- | ------ | --------------- | ------ | --------------------------- |
| HYDE PARK            | 92.000 | 108.540         | 17.557 | 1.751                       |
| DOUGLAS              | 33.000 | 105.970         | 25.073 | 21.784                      |
| SOUTH LAWNDALE       | 16.000 | 104.175         | 13.409 | 4.795                       |
| LINCOLN SQUARE       | 19.000 | 101.516         | 11.697 | 2.019                       |
| ENGLEWOOD            | 16.000 | 99.369          | 13.372 | 5.178                       |

MAE 最高的 community area：

| community_area_clean | n       | actual_mean_eui | predicted_mean_eui | rmse   | mae    | bias_mean_pred_minus_actual |
| -------------------- | ------- | --------------- | ------------------ | ------ | ------ | --------------------------- |
| DOUGLAS              | 33.000  | 105.970         | 127.754            | 33.030 | 25.073 | 21.784                      |
| WASHINGTON HEIGHTS   | 15.000  | 76.760          | 92.905             | 27.107 | 21.660 | 16.145                      |
| DUNNING              | 17.000  | 84.300          | 87.178             | 25.321 | 18.385 | 2.878                       |
| HYDE PARK            | 92.000  | 108.540         | 110.291            | 33.607 | 17.557 | 1.751                       |
| BELMONT CRAGIN       | 24.000  | 72.833          | 80.936             | 23.960 | 16.871 | 8.103                       |
| LOGAN SQUARE         | 44.000  | 70.743          | 80.196             | 22.951 | 15.579 | 9.453                       |
| IRVING PARK          | 17.000  | 66.065          | 76.143             | 19.167 | 15.477 | 10.078                      |
| WEST TOWN            | 75.000  | 78.360          | 89.388             | 27.538 | 15.471 | 11.028                      |
| NORWOOD PARK         | 19.000  | 98.721          | 104.162            | 21.557 | 15.433 | 5.441                       |
| NEAR NORTH SIDE      | 478.000 | 79.270          | 90.261             | 20.229 | 15.180 | 10.991                      |

## 7. Building Type 誤差分析

以下表格列出 test set 中平均 EUI 較高的 building type。

| primary_property_type_clean           | n      | actual_mean_eui | mae    | bias_mean_pred_minus_actual |
| ------------------------------------- | ------ | --------------- | ------ | --------------------------- |
| Laboratory                            | 16.000 | 269.900         | 53.529 | 11.374                      |
| Supermarket/Grocery Store             | 42.000 | 238.076         | 37.959 | -3.060                      |
| Hospital (General Medical & Surgical) | 16.000 | 227.287         | 14.679 | 6.158                       |
| Hotel                                 | 76.000 | 104.679         | 15.116 | 9.193                       |
| Strip Mall                            | 25.000 | 101.668         | 24.754 | -1.081                      |

MAE 最高的 building type：

| primary_property_type_clean           | n      | actual_mean_eui | predicted_mean_eui | rmse   | mae    | bias_mean_pred_minus_actual |
| ------------------------------------- | ------ | --------------- | ------------------ | ------ | ------ | --------------------------- |
| Laboratory                            | 16.000 | 269.900         | 281.274            | 82.868 | 53.529 | 11.374                      |
| Supermarket/Grocery Store             | 42.000 | 238.076         | 235.017            | 51.835 | 37.959 | -3.060                      |
| Strip Mall                            | 25.000 | 101.668         | 100.587            | 34.013 | 24.754 | -1.081                      |
| Other                                 | 16.000 | 91.862          | 96.204             | 35.775 | 23.767 | 4.342                       |
| Senior Living Community               | 52.000 | 100.846         | 110.510            | 22.967 | 16.569 | 9.664                       |
| Retail Store                          | 37.000 | 76.362          | 83.820             | 24.950 | 16.285 | 7.458                       |
| College/University                    | 84.000 | 90.910          | 98.107             | 21.369 | 15.415 | 7.197                       |
| Hotel                                 | 76.000 | 104.679         | 113.872            | 20.000 | 15.116 | 9.193                       |
| Hospital (General Medical & Surgical) | 16.000 | 227.287         | 233.446            | 17.201 | 14.679 | 6.158                       |
| Mixed Use Property                    | 33.000 | 84.073          | 93.358             | 17.543 | 14.323 | 9.286                       |

## 8. Feature Importance 分析

依 validation RMSE 選出的 tree model 的前 15 個重要特徵如下：

| feature                        | importance |
| ------------------------------ | ---------- |
| primary_property_type_clean    | 0.4250     |
| community_area_clean           | 0.2164     |
| zip_code_clean                 | 0.1799     |
| prior_building_mean_eui        | 0.0829     |
| prev_year_site_eui             | 0.0201     |
| regulation_stage               | 0.0114     |
| property_year_record_count     | 0.0098     |
| prior_comm_property_mean_eui   | 0.0085     |
| prior_property_type_mean_eui   | 0.0053     |
| year_built                     | 0.0036     |
| longitude                      | 0.0034     |
| prior_community_high_eui_ratio | 0.0032     |
| community_year_avg_floor_area  | 0.0030     |
| community_year_record_count    | 0.0030     |
| building_age                   | 0.0030     |

整體而言，模型主要依賴歷史 EUI、建物類型、建物面積與地理區位資訊。這符合都市能源使用的直覺：建築用途決定能源需求型態，建物規模與年份反映實體條件，而 community area 與座標則捕捉空間上的社經、建築密度與區域使用型態差異。

## 9. Urban Insight

1. **建築類型是能源使用強度的重要因素。** 不同 building type 的 EUI 差異很大，例如醫療、實驗室、宿舍、旅館、超市等通常有較高能源需求。這表示節能政策不應只用單一標準，而應依 building type 設定不同 benchmark。

2. **空間差異存在，且模型在部分 community area 誤差較大。** Community-level error analysis 可以幫助市政府找出模型較不穩定或能源表現較特殊的地區。這些區域可能需要更多資料欄位，例如 occupancy、HVAC system、建築翻修狀態或更細的土地使用資料。

3. **歷史能源表現對預測有幫助。** `prev_year_site_eui`、`prior_building_mean_eui`、community/property type historical mean 等特徵若重要，代表建築能源使用具有時間延續性。對政策而言，過去多年持續高 EUI 的建築比單一年異常值更值得優先稽核。

4. **Baseline comparison 很重要。** 若進階模型只比 global mean 好，代表模型沒有真正學到細緻模式；若能明顯超過 group mean baseline，才表示模型有捕捉到更複雜的時空與建物特徵交互作用。

## 10. 成果分析與限制

本專案符合 proposal 的核心要求：使用 spatio-temporal data mining，任務定義為 regression，建立至少一個 baseline，使用 RMSE、MAE、R² 評估，並額外檢查 spatial error。

報告與資料限制：

1. **時間解析度限制。** 資料是年度資料，無法分析季節、月份、天氣事件或日夜用電差異。
2. **非 trajectory data。** 建築物位置固定，因此 map matching 不適用；本研究屬於 point-based spatial static and temporal dynamic data。
3. **缺少重要外部變數。** 資料沒有直接提供 occupancy、營業時間、設備效率、HVAC 系統、翻修紀錄、即時天氣等資訊，因此模型可能無法完整解釋 EUI 差異。
4. **Self-reporting error。** Benchmarking 資料可能有申報錯誤或填報品質差異，所以即使已移除 outliers，仍可能存在噪音。
5. **樣本不均衡。** Multifamily Housing、Office、K-12 School 等類型樣本較多，少數 building type 的模型誤差可能較不穩定。
6. **Spatial bias。** 某些 community area 樣本多、某些樣本少，模型可能在資料稀少區域表現較差。
7. **Leakage 欄位必須排除。** 若使用 electricity use、natural gas use、GHG emissions 或 ENERGY STAR score 來預測 Site EUI，會讓模型看起來表現很好，但因為這些欄位與目標高度同源，不適合作為真實預測情境的輸入。

## 11. 結論

本研究建立了 Chicago building energy intensity 的 spatio-temporal regression workflow，包含資料清理、outlier 處理、leakage 排除、歷史與空間特徵工程、baseline comparison、Random Forest/XGBoost random search tuning，以及 test year 2023 的最終評估。

依 validation RMSE 選出的模型為 **XGBoost**，在 test set 的結果為 RMSE = **20.1889**、MAE = **13.8336**、R² = **0.7745**。另外，test set 上 RMSE 最低的是 **Random Forest**，RMSE = **19.4437**、MAE = **13.1277**、R² = **0.7908**。此結果可以用來協助城市管理者初步識別高能源使用強度的建築類型與地區，並作為後續節能稽核或政策分區管理的資料基礎。
