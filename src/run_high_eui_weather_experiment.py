from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for path in [ROOT, SRC_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import compare_external_features as ext
import run_energy_benchmarking_analysis as base


OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = base.RANDOM_STATE

HIGH_EUI_FEATURES = [
    "prior_eui_count",
    "prior_max_site_eui",
    "prior_eui_std",
    "prior_eui_volatility_ratio",
    "eui_trend_prev_vs_prior_mean",
    "was_high_eui_last_year_100",
    "was_high_eui_last_year_150",
    "was_high_eui_last_year_200",
    "prior_high_eui_count_100",
    "prior_high_eui_count_150",
    "prior_high_eui_count_200",
    "prior_high_eui_rate_100",
    "prior_high_eui_rate_150",
    "prior_high_eui_rate_200",
    "prior_community_high_eui_ratio_150",
    "prior_community_high_eui_ratio_200",
    "prior_property_high_eui_ratio_150",
    "prior_property_high_eui_ratio_200",
    "prior_comm_property_high_eui_ratio_150",
    "prior_comm_property_high_eui_ratio_200",
]


XGB_PARAM_DIST = {
    "n_estimators": [200, 400, 600],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "reg_lambda": [1, 5, 10],
    "min_child_weight": [1, 3, 5],
}


def rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def weighted_rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    err = (np.asarray(y_true) - np.asarray(y_pred)) ** 2
    return float(np.sqrt(np.average(err, weights=weights)))


def high_eui_weights(y: pd.Series | np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y)
    weights = np.ones(len(y_arr), dtype=float)
    weights[(y_arr >= 100) & (y_arr < 150)] = 1.5
    weights[(y_arr >= 150) & (y_arr < 200)] = 2.5
    weights[y_arr >= 200] = 4.0
    return weights


def add_high_eui_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["id", "data_year"]).copy()
    grouped = out.groupby("id", group_keys=False)

    out["prior_eui_count"] = grouped.cumcount()
    out["prior_max_site_eui"] = grouped[base.TARGET].transform(
        lambda s: s.shift(1).expanding(min_periods=1).max()
    )
    out["prior_eui_std"] = grouped[base.TARGET].transform(
        lambda s: s.shift(1).expanding(min_periods=2).std()
    )
    out["prior_eui_volatility_ratio"] = out["prior_eui_std"] / out["prior_building_mean_eui"].replace(0, np.nan)
    out["eui_trend_prev_vs_prior_mean"] = out["prev_year_site_eui"] - out["prior_building_mean_eui"]

    for threshold in [100, 150, 200]:
        flag_col = f"high_eui_{threshold}_flag"
        out[flag_col] = (out[base.TARGET] >= threshold).astype(int)
        out[f"was_high_eui_last_year_{threshold}"] = (
            out["prev_year_site_eui"].ge(threshold).fillna(False).astype(int)
        )
        count_col = f"prior_high_eui_count_{threshold}"
        rate_col = f"prior_high_eui_rate_{threshold}"
        out[count_col] = grouped[flag_col].transform(lambda s: s.shift(1).expanding(min_periods=1).sum())
        out[rate_col] = out[count_col] / out["prior_eui_count"].replace(0, np.nan)

    for threshold in [150, 200]:
        flag_col = f"high_eui_{threshold}_flag"
        out = base.add_historical_ratio(
            out,
            ["community_area_clean"],
            f"prior_community_high_eui_ratio_{threshold}",
            flag_col,
        )
        out = base.add_historical_ratio(
            out,
            ["primary_property_type_clean"],
            f"prior_property_high_eui_ratio_{threshold}",
            flag_col,
        )
        out = base.add_historical_ratio(
            out,
            ["community_area_clean", "primary_property_type_clean"],
            f"prior_comm_property_high_eui_ratio_{threshold}",
            flag_col,
        )

    for col in HIGH_EUI_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
    return out


def load_weather_base() -> pd.DataFrame:
    df = pd.read_csv(ext.MODELING_TABLE, low_memory=False)
    df = ext.add_address_keys(df)
    weather_df, _ = ext.add_weather_features(df)
    return weather_df


def load_weather_xgb_params() -> dict:
    candidates = [
        OUTPUT_DIR / "external_feature_tuning_results_with_all.csv",
        OUTPUT_DIR / "external_feature_tuning_results.csv",
    ]
    for path in candidates:
        if path.exists():
            tuning = pd.read_csv(path)
            rows = tuning[(tuning["feature_set"].eq("weather")) & (tuning["model"].eq("XGBoost"))]
            if not rows.empty:
                return json.loads(rows.sort_values("validation_rmse").iloc[0]["params"])
    raise FileNotFoundError("No weather XGBoost tuning results found.")


def set_feature_globals(extra_features: list[str]):
    original = (base.NUMERIC_FEATURES.copy(), base.CATEGORICAL_FEATURES.copy(), base.FEATURES.copy())
    base.NUMERIC_FEATURES = list(dict.fromkeys(base.NUMERIC_FEATURES + ext.WEATHER_FEATURES + extra_features))
    base.FEATURES = base.NUMERIC_FEATURES + base.CATEGORICAL_FEATURES
    return original


def restore_feature_globals(original) -> None:
    base.NUMERIC_FEATURES, base.CATEGORICAL_FEATURES, base.FEATURES = original


def make_xgb(params: dict) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **params,
    )


def fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    pipeline = Pipeline(
        steps=[
            ("preprocess", base.build_preprocessor()),
            ("model", make_xgb(params)),
        ]
    )
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight
    pipeline.fit(train_df[base.FEATURES], train_df[base.TARGET], **fit_kwargs)
    return pipeline.predict(test_df[base.FEATURES])


def tune_xgb(
    variant: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    use_weights: bool,
    selection_metric: str,
    n_iter: int = 25,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    best_params = None
    best_score = math.inf
    train_weights = high_eui_weights(train_df[base.TARGET]) if use_weights else None
    val_weights = high_eui_weights(val_df[base.TARGET])

    samples = list(ParameterSampler(XGB_PARAM_DIST, n_iter=n_iter, random_state=RANDOM_STATE))
    for i, params in enumerate(samples, start=1):
        pred = fit_predict(train_df, val_df, params, sample_weight=train_weights)
        row = {
            "variant": variant,
            "iteration": i,
            "use_sample_weight": use_weights,
            "selection_metric": selection_metric,
            "validation_rmse": rmse(val_df[base.TARGET], pred),
            "validation_mae": float(mean_absolute_error(val_df[base.TARGET], pred)),
            "validation_r2": float(r2_score(val_df[base.TARGET], pred)),
            "validation_weighted_rmse": weighted_rmse(val_df[base.TARGET], pred, val_weights),
            "validation_high150_rmse": rmse(
                val_df.loc[val_df[base.TARGET] >= 150, base.TARGET],
                pred[val_df[base.TARGET].to_numpy() >= 150],
            ),
            "validation_high150_mae": float(
                mean_absolute_error(
                    val_df.loc[val_df[base.TARGET] >= 150, base.TARGET],
                    pred[val_df[base.TARGET].to_numpy() >= 150],
                )
            ),
            "params": json.dumps(params, sort_keys=True),
        }
        rows.append(row)
        score = row[selection_metric]
        if score < best_score:
            best_score = score
            best_params = params
        print(f"{variant} {i:02d}/{n_iter}: {selection_metric}={score:.4f}")

    assert best_params is not None
    return best_params, pd.DataFrame(rows).sort_values(selection_metric)


def prediction_frame(df: pd.DataFrame, variant: str, pred: np.ndarray) -> pd.DataFrame:
    out = df[
        [
            "id",
            "address",
            "data_year",
            base.TARGET,
            "community_area_clean",
            "primary_property_type_clean",
        ]
    ].copy()
    out["variant"] = variant
    out["prediction"] = pred
    out["abs_error"] = (out["prediction"] - out[base.TARGET]).abs()
    out["squared_error"] = (out["prediction"] - out[base.TARGET]) ** 2
    out["bias_pred_minus_actual"] = out["prediction"] - out[base.TARGET]
    return out


def summarize_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in pred_df.groupby("variant"):
        rows.append(
            {
                "variant": variant,
                "rmse": rmse(group[base.TARGET], group["prediction"]),
                "mae": float(group["abs_error"].mean()),
                "r2": float(r2_score(group[base.TARGET], group["prediction"])),
                "bias_mean_pred_minus_actual": float(group["bias_pred_minus_actual"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse")


def summarize_subgroups(pred_df: pd.DataFrame) -> pd.DataFrame:
    subgroup_defs = {
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
    rows = []
    for variant, group in pred_df.groupby("variant"):
        for label, mask_fn in subgroup_defs.items():
            selected = group[mask_fn(group)]
            if selected.empty:
                continue
            rows.append(
                {
                    "variant": variant,
                    "subgroup": label,
                    "n": len(selected),
                    "actual_mean_eui": float(selected[base.TARGET].mean()),
                    "predicted_mean_eui": float(selected["prediction"].mean()),
                    "mae": float(selected["abs_error"].mean()),
                    "rmse": rmse(selected[base.TARGET], selected["prediction"]),
                    "bias_mean_pred_minus_actual": float(selected["bias_pred_minus_actual"].mean()),
                }
            )
    return pd.DataFrame(rows)


def delta_vs_reference(summary: pd.DataFrame, reference: str) -> pd.DataFrame:
    ref = summary[summary["variant"].eq(reference)].copy()
    rows = []
    for variant in summary["variant"].unique():
        if variant == reference:
            continue
        comp = summary[summary["variant"].eq(variant)].merge(ref, on="subgroup", suffixes=("", "_reference"))
        comp["reference_variant"] = reference
        comp["mae_change"] = comp["mae"] - comp["mae_reference"]
        comp["mae_change_pct"] = comp["mae_change"] / comp["mae_reference"] * 100
        comp["rmse_change"] = comp["rmse"] - comp["rmse_reference"]
        comp["rmse_change_pct"] = comp["rmse_change"] / comp["rmse_reference"] * 100
        rows.append(
            comp[
                [
                    "variant",
                    "reference_variant",
                    "subgroup",
                    "n",
                    "mae_reference",
                    "mae",
                    "mae_change",
                    "mae_change_pct",
                    "rmse_reference",
                    "rmse",
                    "rmse_change",
                    "rmse_change_pct",
                    "bias_mean_pred_minus_actual_reference",
                    "bias_mean_pred_minus_actual",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def plot_subgroup(summary: pd.DataFrame, metric: str) -> None:
    ordered_groups = [
        "All test records",
        "High EUI >= 100",
        "Very high EUI >= 150",
        "Extreme EUI >= 200",
        "Laboratory",
        "Supermarket/Grocery Store",
        "Hospital",
        "Hotel",
        "Senior Living Community",
        "College/University",
    ]
    pivot = summary.pivot(index="subgroup", columns="variant", values=metric).reindex(ordered_groups)
    ax = pivot.plot(
        kind="bar",
        figsize=(13, 6.5),
        color=["#7A5195", "#2F6F73", "#C44E52"],
    )
    ax.set_xlabel("Subgroup")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"High-EUI Weather Experiment: Subgroup {metric.upper()}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Variant")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"high_eui_weather_subgroup_{metric}.png", dpi=180)
    plt.close()


def md_table(df: pd.DataFrame, columns: list[str], digits: int = 3) -> str:
    out = df[columns].copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
    rows = [columns] + out.fillna("").astype(str).values.tolist()
    widths = [max(len(row[i]) for row in rows) for i in range(len(columns))]
    lines = ["| " + " | ".join(rows[0][i].ljust(widths[i]) for i in range(len(columns))) + " |"]
    lines.append("| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |")
    return "\n".join(lines)


def write_report(metrics: pd.DataFrame, subgroup: pd.DataFrame, delta: pd.DataFrame, tuning: pd.DataFrame) -> None:
    focus = subgroup[
        subgroup["subgroup"].isin(
            [
                "All test records",
                "High EUI >= 100",
                "Very high EUI >= 150",
                "Extreme EUI >= 200",
                "Laboratory",
                "Supermarket/Grocery Store",
                "Hospital",
                "Hotel",
                "Senior Living Community",
                "College/University",
            ]
        )
    ].sort_values(["subgroup", "variant"])
    weighted_delta = delta[delta["variant"].eq("weather_high_features_weighted_xgb")]
    report = f"""# High-EUI Weather Model Experiment

This experiment uses the final base + weather feature set as the starting point, then tests whether high-EUI-aware historical features and sample weighting improve prediction for extreme high-EUI buildings.

## Variants

1. `weather_xgb_reference`: original weather-only XGBoost model.
2. `weather_high_features_xgb`: weather + high-EUI historical features, unweighted training.
3. `weather_high_features_weighted_xgb`: weather + high-EUI historical features, weighted training.

## High-EUI Historical Features

New features include:

- prior max EUI
- prior EUI standard deviation
- prior EUI volatility ratio
- previous-year high-EUI indicators for thresholds 100, 150, and 200
- prior high-EUI counts and rates for thresholds 100, 150, and 200
- prior community/property/community-property high-EUI ratios for thresholds 150 and 200
- previous-year EUI trend relative to prior building mean

All historical features use only years before the prediction year.

## Sample Weighting

Weights:

| EUI range | weight |
|---|---:|
| EUI < 100 | 1.0 |
| 100 <= EUI < 150 | 1.5 |
| 150 <= EUI < 200 | 2.5 |
| EUI >= 200 | 4.0 |

The weighted model was selected by validation weighted RMSE.

## Overall Test Metrics

{md_table(metrics, ["variant", "rmse", "mae", "r2", "bias_mean_pred_minus_actual"], digits=4)}

## Subgroup Error

{md_table(focus, ["variant", "subgroup", "n", "mae", "rmse", "bias_mean_pred_minus_actual"], digits=3)}

## Weighted Model Change vs Reference

{md_table(weighted_delta, ["subgroup", "n", "mae_reference", "mae", "mae_change", "mae_change_pct", "rmse_reference", "rmse", "rmse_change_pct"], digits=3)}

## Visualizations

![Subgroup MAE](../outputs/plots/high_eui_weather_subgroup_mae.png)

![Subgroup RMSE](../outputs/plots/high_eui_weather_subgroup_rmse.png)

## Tuning Summary

{md_table(tuning.groupby("variant").head(3), ["variant", "iteration", "use_sample_weight", "selection_metric", "validation_rmse", "validation_mae", "validation_weighted_rmse", "validation_high150_rmse", "validation_high150_mae"], digits=4)}
"""
    (REPORT_DIR / "high_eui_weather_experiment.md").write_text(report, encoding="utf-8")


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    weather_df = load_weather_base()
    high_df = add_high_eui_historical_features(weather_df)

    trainval_weather = weather_df[weather_df["data_year"].between(2018, 2022)].copy()
    test_weather = weather_df[weather_df["data_year"].eq(2023)].copy()
    train_high = high_df[high_df["data_year"].between(2018, 2021)].copy()
    val_high = high_df[high_df["data_year"].eq(2022)].copy()
    trainval_high = high_df[high_df["data_year"].between(2018, 2022)].copy()
    test_high = high_df[high_df["data_year"].eq(2023)].copy()

    predictions = []
    tuning_results = []

    original = set_feature_globals([])
    try:
        ref_params = load_weather_xgb_params()
        ref_pred = fit_predict(trainval_weather, test_weather, ref_params)
        predictions.append(prediction_frame(test_weather, "weather_xgb_reference", ref_pred))
    finally:
        restore_feature_globals(original)

    original = set_feature_globals(HIGH_EUI_FEATURES)
    try:
        high_params, high_tuning = tune_xgb(
            "weather_high_features_xgb",
            train_high,
            val_high,
            use_weights=False,
            selection_metric="validation_rmse",
        )
        tuning_results.append(high_tuning)
        high_pred = fit_predict(trainval_high, test_high, high_params)
        predictions.append(prediction_frame(test_high, "weather_high_features_xgb", high_pred))

        weighted_params, weighted_tuning = tune_xgb(
            "weather_high_features_weighted_xgb",
            train_high,
            val_high,
            use_weights=True,
            selection_metric="validation_weighted_rmse",
        )
        tuning_results.append(weighted_tuning)
        weighted_trainval_weights = high_eui_weights(trainval_high[base.TARGET])
        weighted_pred = fit_predict(
            trainval_high,
            test_high,
            weighted_params,
            sample_weight=weighted_trainval_weights,
        )
        predictions.append(prediction_frame(test_high, "weather_high_features_weighted_xgb", weighted_pred))
    finally:
        restore_feature_globals(original)

    pred_df = pd.concat(predictions, ignore_index=True)
    tuning_df = pd.concat(tuning_results, ignore_index=True)
    metrics_df = summarize_metrics(pred_df)
    subgroup_df = summarize_subgroups(pred_df)
    delta_df = delta_vs_reference(subgroup_df, "weather_xgb_reference")

    pred_df.to_csv(OUTPUT_DIR / "high_eui_weather_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(OUTPUT_DIR / "high_eui_weather_metrics.csv", index=False, encoding="utf-8-sig")
    subgroup_df.to_csv(OUTPUT_DIR / "high_eui_weather_subgroup_error.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(OUTPUT_DIR / "high_eui_weather_subgroup_delta.csv", index=False, encoding="utf-8-sig")
    tuning_df.to_csv(OUTPUT_DIR / "high_eui_weather_tuning_results.csv", index=False, encoding="utf-8-sig")

    plot_subgroup(subgroup_df, "mae")
    plot_subgroup(subgroup_df, "rmse")
    write_report(metrics_df, subgroup_df, delta_df, tuning_df.sort_values(["variant", "validation_weighted_rmse"]))

    print("Done high-EUI weather experiment.")
    print(metrics_df.to_string(index=False))
    print(
        delta_df[delta_df["variant"].eq("weather_high_features_weighted_xgb")]
        .sort_values("subgroup")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
