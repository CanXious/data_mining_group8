from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import compare_external_features as ext
import run_energy_benchmarking_analysis as base


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"


def load_best_params(feature_set: str, model_name: str) -> dict:
    tuning = pd.read_csv(OUTPUT_DIR / "external_feature_tuning_results.csv")
    row = (
        tuning[(tuning["feature_set"].eq(feature_set)) & (tuning["model"].eq(model_name))]
        .sort_values("validation_rmse")
        .iloc[0]
    )
    return json.loads(row["params"])


def set_features(extra_numeric: list[str]):
    original = (base.NUMERIC_FEATURES.copy(), base.CATEGORICAL_FEATURES.copy(), base.FEATURES.copy())
    base.NUMERIC_FEATURES = list(dict.fromkeys(base.NUMERIC_FEATURES + extra_numeric))
    base.FEATURES = base.NUMERIC_FEATURES + base.CATEGORICAL_FEATURES
    return original


def restore_features(original) -> None:
    base.NUMERIC_FEATURES, base.CATEGORICAL_FEATURES, base.FEATURES = original


def predict_variant(df: pd.DataFrame, feature_set: str, model_name: str, extra_numeric: list[str]) -> pd.DataFrame:
    original = set_features(extra_numeric)
    try:
        params = load_best_params(feature_set, model_name)
        trainval = df[df["data_year"].between(2018, 2022)].copy()
        test = df[df["data_year"].eq(2023)].copy()
        model = base.fit_pipeline(model_name, params, trainval)
        out = test[
            [
                "id",
                "address",
                "data_year",
                base.TARGET,
                "community_area_clean",
                "primary_property_type_clean",
            ]
        ].copy()
        out["feature_set"] = feature_set
        out["model"] = model_name
        out["prediction"] = model.predict(test[base.FEATURES])
        out["abs_error"] = (out["prediction"] - out[base.TARGET]).abs()
        out["squared_error"] = (out["prediction"] - out[base.TARGET]) ** 2
        return out
    finally:
        restore_features(original)


def summarize_group(df: pd.DataFrame, label: str) -> dict:
    return {
        "group": label,
        "n": len(df),
        "actual_mean_eui": df[base.TARGET].mean(),
        "predicted_mean_eui": df["prediction"].mean(),
        "mae": df["abs_error"].mean(),
        "rmse": float(np.sqrt(df["squared_error"].mean())),
        "bias_pred_minus_actual": df["prediction"].mean() - df[base.TARGET].mean(),
    }


def compare_subgroups(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subgroups = {
        "All test records": lambda d: pd.Series(True, index=d.index),
        "High EUI >= 100": lambda d: d[base.TARGET] >= 100,
        "Very high EUI >= 150": lambda d: d[base.TARGET] >= 150,
        "Extreme EUI >= 200": lambda d: d[base.TARGET] >= 200,
        "Laboratory": lambda d: d["primary_property_type_clean"].eq("Laboratory"),
        "Supermarket/Grocery Store": lambda d: d["primary_property_type_clean"].eq("Supermarket/Grocery Store"),
        "Hospital": lambda d: d["primary_property_type_clean"].str.contains("Hospital", case=False, na=False),
        "Hotel": lambda d: d["primary_property_type_clean"].eq("Hotel"),
        "Senior Living Community": lambda d: d["primary_property_type_clean"].eq("Senior Living Community"),
        "College/University": lambda d: d["primary_property_type_clean"].eq("College/University"),
    }
    for (feature_set, model_name), group_df in predictions.groupby(["feature_set", "model"]):
        for label, mask_fn in subgroups.items():
            selected = group_df[mask_fn(group_df)]
            if selected.empty:
                continue
            rows.append({"feature_set": feature_set, "model": model_name, **summarize_group(selected, label)})
    return pd.DataFrame(rows)


def add_delta_table(summary: pd.DataFrame, left_key: tuple[str, str], right_key: tuple[str, str], label: str) -> pd.DataFrame:
    left = summary[(summary["feature_set"].eq(left_key[0])) & (summary["model"].eq(left_key[1]))].copy()
    right = summary[(summary["feature_set"].eq(right_key[0])) & (summary["model"].eq(right_key[1]))].copy()
    merged = left.merge(right, on="group", suffixes=("_before", "_after"))
    merged["comparison"] = label
    merged["mae_change"] = merged["mae_after"] - merged["mae_before"]
    merged["mae_change_pct"] = merged["mae_change"] / merged["mae_before"] * 100
    merged["rmse_change"] = merged["rmse_after"] - merged["rmse_before"]
    merged["rmse_change_pct"] = merged["rmse_change"] / merged["rmse_before"] * 100
    return merged[
        [
            "comparison",
            "group",
            "n_before",
            "mae_before",
            "mae_after",
            "mae_change",
            "mae_change_pct",
            "rmse_before",
            "rmse_after",
            "rmse_change",
            "rmse_change_pct",
            "bias_pred_minus_actual_before",
            "bias_pred_minus_actual_after",
        ]
    ]


def md_table(df: pd.DataFrame, cols: list[str], digits: int = 3) -> str:
    out = df[cols].copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
    rows = [cols] + out.fillna("").astype(str).values.tolist()
    widths = [max(len(row[i]) for row in rows) for i in range(len(cols))]
    lines = ["| " + " | ".join(rows[0][i].ljust(widths[i]) for i in range(len(cols))) + " |"]
    lines.append("| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) + " |")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(ext.MODELING_TABLE, low_memory=False)
    df = ext.add_address_keys(df)
    weather_df, _ = ext.add_weather_features(df.copy())

    predictions = pd.concat(
        [
            predict_variant(df, "base", "Random Forest", []),
            predict_variant(df, "base", "XGBoost", []),
            predict_variant(weather_df, "weather", "XGBoost", ext.WEATHER_FEATURES),
            predict_variant(weather_df, "weather", "Random Forest", ext.WEATHER_FEATURES),
        ],
        ignore_index=True,
    )
    predictions.to_csv(OUTPUT_DIR / "weather_subgroup_predictions.csv", index=False, encoding="utf-8-sig")

    summary = compare_subgroups(predictions)
    summary.to_csv(OUTPUT_DIR / "weather_subgroup_error_summary.csv", index=False, encoding="utf-8-sig")

    best_delta = add_delta_table(
        summary,
        ("base", "Random Forest"),
        ("weather", "XGBoost"),
        "Base best RF -> Weather best XGBoost",
    )
    xgb_delta = add_delta_table(
        summary,
        ("base", "XGBoost"),
        ("weather", "XGBoost"),
        "Base XGBoost -> Weather XGBoost",
    )
    delta = pd.concat([best_delta, xgb_delta], ignore_index=True)
    delta.to_csv(OUTPUT_DIR / "weather_subgroup_error_delta.csv", index=False, encoding="utf-8-sig")

    report = f"""# Weather Feature Impact on High-EUI and Special Building Types

本補充分析檢查 weather features 是否真的改善原本報告提到的問題：高 EUI 建築與特殊 building type 的預測誤差較大。

比較方式：

1. **Base best RF -> Weather best XGBoost**：用整體 test 表現最佳的 base model 與 weather-only model 比較。
2. **Base XGBoost -> Weather XGBoost**：固定模型家族，只比較 XGBoost 加 weather 前後。

負值代表 weather model 誤差降低。

## Base Best Model vs Weather Best Model

{md_table(best_delta, ["group", "n_before", "mae_before", "mae_after", "mae_change", "mae_change_pct", "rmse_before", "rmse_after", "rmse_change_pct", "bias_pred_minus_actual_before", "bias_pred_minus_actual_after"], digits=3)}

## Same Model Family: XGBoost Before vs After Weather

{md_table(xgb_delta, ["group", "n_before", "mae_before", "mae_after", "mae_change", "mae_change_pct", "rmse_before", "rmse_after", "rmse_change_pct", "bias_pred_minus_actual_before", "bias_pred_minus_actual_after"], digits=3)}

## Interpretation

- 若高 EUI subgroup 的 MAE/RMSE 下降，代表 weather features 有幫助降低高耗能建築的誤差。
- 若特定 building type 的誤差沒有下降，代表該類型建築的誤差可能不是主要由年度天氣造成，而是需要 occupancy、operating hours、HVAC system 或更細的 building operation data。
- `bias_pred_minus_actual` 若為負值，代表模型平均低估該 subgroup 的 EUI。
"""
    (REPORT_DIR / "weather_subgroup_error_analysis.md").write_text(report, encoding="utf-8")
    print("Done weather subgroup analysis.")
    print(best_delta.to_string(index=False))


if __name__ == "__main__":
    main()
