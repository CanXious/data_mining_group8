from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import compare_external_features as ext


base = ext.base

INTERACTION_BASE_FEATURES = [
    "weather_hdd65_x_building_age",
    "weather_cdd65_x_building_age",
    "weather_hot_days_x_building_age",
    "weather_cold_days_x_building_age",
    "weather_hdd65_x_log_floor_area",
    "weather_cdd65_x_log_floor_area",
    "weather_hot_days_x_log_floor_area",
    "weather_cold_days_x_log_floor_area",
    "weather_hdd65_x_prev_year_site_eui",
    "weather_cdd65_x_prev_year_site_eui",
    "weather_hdd65_x_prior_property_type_mean_eui",
    "weather_cdd65_x_prior_property_type_mean_eui",
]


def sanitize_token(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return re.sub(r"_+", "_", text)


def add_weather_interactions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["weather_hdd65_x_building_age"] = out["weather_hdd65"] * out["building_age"]
    out["weather_cdd65_x_building_age"] = out["weather_cdd65"] * out["building_age"]
    out["weather_hot_days_x_building_age"] = out["weather_hot_days_30c"] * out["building_age"]
    out["weather_cold_days_x_building_age"] = out["weather_cold_days_minus10c"] * out["building_age"]
    out["weather_hdd65_x_log_floor_area"] = out["weather_hdd65"] * out["log_floor_area"]
    out["weather_cdd65_x_log_floor_area"] = out["weather_cdd65"] * out["log_floor_area"]
    out["weather_hot_days_x_log_floor_area"] = out["weather_hot_days_30c"] * out["log_floor_area"]
    out["weather_cold_days_x_log_floor_area"] = out["weather_cold_days_minus10c"] * out["log_floor_area"]
    out["weather_hdd65_x_prev_year_site_eui"] = out["weather_hdd65"] * out["prev_year_site_eui"]
    out["weather_cdd65_x_prev_year_site_eui"] = out["weather_cdd65"] * out["prev_year_site_eui"]
    out["weather_hdd65_x_prior_property_type_mean_eui"] = (
        out["weather_hdd65"] * out["prior_property_type_mean_eui"]
    )
    out["weather_cdd65_x_prior_property_type_mean_eui"] = (
        out["weather_cdd65"] * out["prior_property_type_mean_eui"]
    )

    train_types = (
        out[out["data_year"].between(2018, 2021)]["primary_property_type_clean"]
        .value_counts()
        .head(5)
        .index.tolist()
    )
    property_interactions = []
    for property_type in train_types:
        token = sanitize_token(str(property_type))
        hdd_col = f"weather_hdd65_x_property_type_{token}"
        cdd_col = f"weather_cdd65_x_property_type_{token}"
        indicator = out["primary_property_type_clean"].eq(property_type).astype(int)
        out[hdd_col] = out["weather_hdd65"] * indicator
        out[cdd_col] = out["weather_cdd65"] * indicator
        property_interactions.extend([hdd_col, cdd_col])

    return out, INTERACTION_BASE_FEATURES + property_interactions


def rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def metrics_row(feature_set: str, split: str, model: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "feature_set": feature_set,
        "split": split,
        "model": model,
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def get_param_spaces() -> tuple[dict, dict]:
    rf_param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [8, 12, 16, 24, None],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_features": ["sqrt", 0.5, 0.8],
    }
    xgb_param_dist = {
        "n_estimators": [200, 400, 600],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_lambda": [1, 5, 10],
        "min_child_weight": [1, 3, 5],
    }
    return rf_param_dist, xgb_param_dist


def estimator_for(model_name: str, params: dict):
    if model_name == "Random Forest":
        return RandomForestRegressor(random_state=base.RANDOM_STATE, n_jobs=-1, **params)
    if model_name == "XGBoost":
        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            eval_metric="rmse",
            random_state=base.RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            **params,
        )
    raise ValueError(f"Unknown model: {model_name}")


def target_train_values(y: pd.Series, target_mode: str) -> np.ndarray:
    if target_mode == "raw":
        return y.to_numpy()
    if target_mode == "log":
        return np.log1p(y.to_numpy())
    raise ValueError(f"Unknown target mode: {target_mode}")


def inverse_predictions(pred: np.ndarray, target_mode: str) -> np.ndarray:
    if target_mode == "raw":
        return pred
    if target_mode == "log":
        return np.clip(np.expm1(pred), 0, None)
    raise ValueError(f"Unknown target mode: {target_mode}")


def make_pipeline(model_name: str, params: dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", base.build_preprocessor()),
            ("model", estimator_for(model_name, params)),
        ]
    )


def tune_model(
    feature_set: str,
    model_name: str,
    param_dist: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_mode: str,
    n_iter: int,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    best_params = None
    best_rmse = math.inf
    samples = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=base.RANDOM_STATE))

    for i, params in enumerate(samples, start=1):
        model = make_pipeline(model_name, params)
        model.fit(train_df[base.FEATURES], target_train_values(train_df[base.TARGET], target_mode))
        pred = inverse_predictions(model.predict(val_df[base.FEATURES]), target_mode)
        row = {
            "feature_set": feature_set,
            "target_mode": target_mode,
            "model": model_name,
            "iteration": i,
            "validation_rmse": rmse(val_df[base.TARGET], pred),
            "validation_mae": float(mean_absolute_error(val_df[base.TARGET], pred)),
            "validation_r2": float(r2_score(val_df[base.TARGET], pred)),
            "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        }
        rows.append(row)
        if row["validation_rmse"] < best_rmse:
            best_rmse = row["validation_rmse"]
            best_params = params
        print(
            f"{feature_set} {model_name} {target_mode} random search "
            f"{i:02d}/{n_iter}: RMSE={row['validation_rmse']:.4f}"
        )

    assert best_params is not None
    return best_params, pd.DataFrame(rows).sort_values("validation_rmse")


def fit_predict(
    model_name: str,
    params: dict,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    target_mode: str,
) -> tuple[np.ndarray, Pipeline]:
    model = make_pipeline(model_name, params)
    model.fit(train_df[base.FEATURES], target_train_values(train_df[base.TARGET], target_mode))
    pred = inverse_predictions(model.predict(eval_df[base.FEATURES]), target_mode)
    return pred, model


def evaluate_improvement_variant(
    feature_set: str,
    df: pd.DataFrame,
    extra_numeric: list[str],
    target_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = ext.set_feature_globals(extra_numeric)
    try:
        train_df = df[df["data_year"].between(2018, 2021)].copy()
        val_df = df[df["data_year"].eq(2022)].copy()
        trainval_df = df[df["data_year"].between(2018, 2022)].copy()
        test_df = df[df["data_year"].eq(2023)].copy()

        rf_param_dist, xgb_param_dist = get_param_spaces()
        rf_params, rf_tuning = tune_model(
            feature_set, "Random Forest", rf_param_dist, train_df, val_df, target_mode, n_iter=20
        )
        xgb_params, xgb_tuning = tune_model(
            feature_set, "XGBoost", xgb_param_dist, train_df, val_df, target_mode, n_iter=25
        )
        tuning = pd.concat([rf_tuning, xgb_tuning], ignore_index=True)

        metrics = []
        predictions = {}
        for model_name, params in {"Random Forest": rf_params, "XGBoost": xgb_params}.items():
            val_pred, _ = fit_predict(model_name, params, train_df, val_df, target_mode)
            test_pred, _ = fit_predict(model_name, params, trainval_df, test_df, target_mode)
            metrics.append(metrics_row(feature_set, "validation", model_name, val_df[base.TARGET], val_pred))
            metrics.append(metrics_row(feature_set, "test", model_name, test_df[base.TARGET], test_pred))
            predictions[model_name] = {"validation": val_pred, "test": test_pred}

        val_ensemble = (predictions["Random Forest"]["validation"] + predictions["XGBoost"]["validation"]) / 2
        test_ensemble = (predictions["Random Forest"]["test"] + predictions["XGBoost"]["test"]) / 2
        metrics.append(metrics_row(feature_set, "validation", "RF_XGBoost_Ensemble", val_df[base.TARGET], val_ensemble))
        metrics.append(metrics_row(feature_set, "test", "RF_XGBoost_Ensemble", test_df[base.TARGET], test_ensemble))

        return pd.DataFrame(metrics), tuning
    finally:
        ext.restore_feature_globals(original)


def md_table(df: pd.DataFrame, cols: list[str], digits: int = 4) -> str:
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


def plot_results(test_metrics: pd.DataFrame) -> None:
    best = test_metrics.sort_values(["feature_set", "rmse"]).groupby("feature_set", as_index=False).first()
    order = [
        "weather_raw",
        "weather_log_target",
        "weather_interactions",
        "weather_interactions_log_target",
    ]
    best = best.set_index("feature_set").reindex(order).reset_index()
    plt.figure(figsize=(10, 5.5))
    plt.bar(best["feature_set"], best["rmse"], color="#2F6F73")
    plt.ylabel("Best Test RMSE")
    plt.xlabel("Improvement experiment")
    plt.title("Weather Model Improvement Experiments")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(ext.PLOT_DIR / "weather_improvement_experiments_rmse.png", dpi=180)
    plt.close()


def write_report(metrics: pd.DataFrame, tuning: pd.DataFrame, interaction_features: list[str]) -> None:
    test = metrics[metrics["split"].eq("test")].copy()
    best_by_set = test.sort_values(["feature_set", "rmse"]).groupby("feature_set", as_index=False).first()
    weather_best = best_by_set.loc[best_by_set["feature_set"].eq("weather_raw"), "rmse"].iloc[0]
    best_by_set["rmse_change_vs_weather_raw"] = best_by_set["rmse"] - weather_best
    best_by_set["rmse_change_pct_vs_weather_raw"] = best_by_set["rmse_change_vs_weather_raw"] / weather_best * 100

    report = f"""# Weather Model Improvement Experiments

This experiment tested three ways to improve the current best model family:

1. Log-transforming the target: `log1p(Site EUI)`.
2. Adding weather interaction features.
3. Averaging Random Forest and XGBoost predictions as an ensemble.

The split and random search spaces are kept consistent with the previous external-feature experiments:

- Train: 2018-2021
- Validation: 2022
- Test: 2023
- Random Forest random search: 20 iterations
- XGBoost random search: 25 iterations

## Interaction Features

```text
{chr(10).join(interaction_features)}
```

## Test Metrics

{md_table(test.sort_values(["feature_set", "rmse"]), ["feature_set", "model", "rmse", "mae", "r2"], digits=4)}

## Best Model by Experiment

{md_table(best_by_set, ["feature_set", "model", "rmse", "mae", "r2", "rmse_change_vs_weather_raw", "rmse_change_pct_vs_weather_raw"], digits=4)}

## Plot

![Weather Improvement Experiments RMSE](../outputs/plots/weather_improvement_experiments_rmse.png)

## Interpretation

- `weather_raw` is a rerun of the current weather-only setup.
- `weather_log_target` tests whether reducing target skew improves original-scale RMSE.
- `weather_interactions` tests whether buildings with different size, age, and property type respond differently to weather.
- `weather_interactions_log_target` combines the target transform and interaction features.
- For each feature set, `RF_XGBoost_Ensemble` is the simple average of the tuned Random Forest and XGBoost predictions.
"""
    (ext.REPORT_DIR / "weather_improvement_experiments.md").write_text(report, encoding="utf-8")


def main() -> None:
    ext.ensure_dirs()
    model_df = pd.read_csv(ext.MODELING_TABLE, low_memory=False)
    weather_df, weather_summary = ext.add_weather_features(model_df.copy())
    interaction_df, interaction_features = add_weather_interactions(weather_df)

    variants = [
        ("weather_raw", weather_df, ext.WEATHER_FEATURES, "raw"),
        ("weather_log_target", weather_df, ext.WEATHER_FEATURES, "log"),
        ("weather_interactions", interaction_df, ext.WEATHER_FEATURES + interaction_features, "raw"),
        ("weather_interactions_log_target", interaction_df, ext.WEATHER_FEATURES + interaction_features, "log"),
    ]

    metrics_frames = []
    tuning_frames = []
    for feature_set, df, extra_numeric, target_mode in variants:
        metrics, tuning = evaluate_improvement_variant(feature_set, df, extra_numeric, target_mode)
        metrics_frames.append(metrics)
        tuning_frames.append(tuning)

    metrics = pd.concat(metrics_frames, ignore_index=True)
    tuning = pd.concat(tuning_frames, ignore_index=True)
    metrics.to_csv(ext.OUTPUT_DIR / "weather_improvement_experiment_metrics.csv", index=False, encoding="utf-8-sig")
    tuning.to_csv(ext.OUTPUT_DIR / "weather_improvement_experiment_tuning_results.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(
        [
            {
                "feature_set": "weather_improvement_experiments",
                "weather_matched_rows": weather_summary["matched_rows"],
                "weather_matched_row_rate": weather_summary["matched_row_rate"],
                "interaction_feature_count": len(interaction_features),
            }
        ]
    ).to_csv(ext.OUTPUT_DIR / "weather_improvement_experiment_summary.csv", index=False, encoding="utf-8-sig")

    plot_results(metrics[metrics["split"].eq("test")])
    write_report(metrics, tuning, interaction_features)

    print("Done weather improvement experiments.")
    print(metrics[metrics["split"].eq("test")].sort_values(["feature_set", "rmse"]).to_string(index=False))


if __name__ == "__main__":
    main()
