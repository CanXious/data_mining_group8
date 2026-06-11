# Weather Feature Impact on High-EUI and Special Building Types

本補充分析檢查 weather features 是否真的改善原本報告提到的問題：高 EUI 建築與特殊 building type 的預測誤差較大。

比較方式：

1. **Base best RF -> Weather best XGBoost**：用整體 test 表現最佳的 base model 與 weather-only model 比較。
2. **Base XGBoost -> Weather XGBoost**：固定模型家族，只比較 XGBoost 加 weather 前後。

負值代表 weather model 誤差降低。

## Base Best Model vs Weather Best Model

| group                     | n_before | mae_before | mae_after | mae_change | mae_change_pct | rmse_before | rmse_after | rmse_change_pct | bias_pred_minus_actual_before | bias_pred_minus_actual_after |
| ------------------------- | -------- | ---------- | --------- | ---------- | -------------- | ----------- | ---------- | --------------- | ----------------------------- | ---------------------------- |
| All test records          | 2579.000 | 13.128     | 11.065    | -2.063     | -15.715        | 19.444      | 17.875     | -8.069          | 7.896                         | 3.081                        |
| High EUI >= 100           | 472.000  | 18.947     | 18.729    | -0.217     | -1.146         | 29.183      | 29.602     | 1.435           | 0.833                         | -6.633                       |
| Very high EUI >= 150      | 131.000  | 28.698     | 29.658    | 0.960      | 3.347          | 43.539      | 44.425     | 2.034           | -8.407                        | -13.257                      |
| Extreme EUI >= 200        | 70.000   | 35.586     | 37.536    | 1.950      | 5.478          | 53.849      | 55.040     | 2.212           | -15.064                       | -20.666                      |
| Laboratory                | 16.000   | 53.676     | 50.426    | -3.249     | -6.054         | 85.168      | 79.852     | -6.242          | 6.500                         | 5.404                        |
| Supermarket/Grocery Store | 42.000   | 35.775     | 34.534    | -1.241     | -3.468         | 52.117      | 50.018     | -4.028          | -5.983                        | -11.645                      |
| Hospital                  | 20.000   | 15.550     | 14.460    | -1.091     | -7.014         | 19.284      | 18.150     | -5.881          | 7.379                         | 0.016                        |
| Hotel                     | 76.000   | 10.230     | 12.088    | 1.858      | 18.164         | 14.818      | 16.800     | 13.374          | 4.251                         | -8.002                       |
| Senior Living Community   | 52.000   | 17.282     | 13.620    | -3.662     | -21.189        | 23.008      | 20.792     | -9.634          | 11.348                        | 4.335                        |
| College/University        | 84.000   | 15.449     | 12.607    | -2.841     | -18.393        | 20.732      | 19.334     | -6.740          | 8.505                         | 0.117                        |

## Same Model Family: XGBoost Before vs After Weather

| group                     | n_before | mae_before | mae_after | mae_change | mae_change_pct | rmse_before | rmse_after | rmse_change_pct | bias_pred_minus_actual_before | bias_pred_minus_actual_after |
| ------------------------- | -------- | ---------- | --------- | ---------- | -------------- | ----------- | ---------- | --------------- | ----------------------------- | ---------------------------- |
| All test records          | 2579.000 | 13.834     | 11.065    | -2.769     | -20.016        | 20.189      | 17.875     | -11.462         | 8.385                         | 3.081                        |
| High EUI >= 100           | 472.000  | 19.828     | 18.729    | -1.099     | -5.541         | 30.164      | 29.602     | -1.865          | 1.317                         | -6.633                       |
| Very high EUI >= 150      | 131.000  | 29.959     | 29.658    | -0.301     | -1.005         | 44.271      | 44.425     | 0.346           | -4.944                        | -13.257                      |
| Extreme EUI >= 200        | 70.000   | 37.332     | 37.536    | 0.204      | 0.547          | 54.818      | 55.040     | 0.405           | -11.735                       | -20.666                      |
| Laboratory                | 16.000   | 53.529     | 50.426    | -3.103     | -5.796         | 82.868      | 79.852     | -3.640          | 11.374                        | 5.404                        |
| Supermarket/Grocery Store | 42.000   | 37.959     | 34.534    | -3.424     | -9.021         | 51.835      | 50.018     | -3.505          | -3.060                        | -11.645                      |
| Hospital                  | 20.000   | 15.237     | 14.460    | -0.777     | -5.102         | 17.673      | 18.150     | 2.703           | 7.699                         | 0.016                        |
| Hotel                     | 76.000   | 15.116     | 12.088    | -3.028     | -20.032        | 20.000      | 16.800     | -15.998         | 9.193                         | -8.002                       |
| Senior Living Community   | 52.000   | 16.569     | 13.620    | -2.948     | -17.795        | 22.967      | 20.792     | -9.470          | 9.664                         | 4.335                        |
| College/University        | 84.000   | 15.415     | 12.607    | -2.807     | -18.212        | 21.369      | 19.334     | -9.522          | 7.197                         | 0.117                        |

## Interpretation

- 若高 EUI subgroup 的 MAE/RMSE 下降，代表 weather features 有幫助降低高耗能建築的誤差。
- 若特定 building type 的誤差沒有下降，代表該類型建築的誤差可能不是主要由年度天氣造成，而是需要 occupancy、operating hours、HVAC system 或更細的 building operation data。
- `bias_pred_minus_actual` 若為負值，代表模型平均低估該 subgroup 的 EUI。
