from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for path in [ROOT, SRC_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import run_energy_benchmarking_analysis as base
import run_high_eui_weather_experiment as high_exp


OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = base.RANDOM_STATE
HIGH_EUI_THRESHOLD = 150
THRESHOLD_GRID = np.round(np.arange(0.05, 0.91, 0.05), 2)
SOFT_GAMMA_GRID = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

XGB_REGRESSOR_PARAM_DIST = high_exp.XGB_PARAM_DIST
XGB_CLASSIFIER_PARAM_DIST_BASE = {
    "n_estimators": [200, 400, 600],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "reg_lambda": [1, 5, 10],
    "min_child_weight": [1, 3, 5],
}


def rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def high_eui_flag(df: pd.DataFrame) -> pd.Series:
    return df[base.TARGET].ge(HIGH_EUI_THRESHOLD).astype(int)


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(roc_auc_score(y_true, score))


def safe_average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(average_precision_score(y_true, score))


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "predicted_high_rate": float(pred.mean()),
        "actual_high_rate": float(y_true.mean()),
    }


def make_classifier(params: dict) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **params,
    )


def make_regressor(params: dict) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **params,
    )


def fit_classifier(train_df: pd.DataFrame, params: dict) -> Pipeline:
    model = Pipeline(
        steps=[
            ("preprocess", base.build_preprocessor()),
            ("model", make_classifier(params)),
        ]
    )
    model.fit(train_df[base.FEATURES], high_eui_flag(train_df))
    return model


def fit_regressor(train_df: pd.DataFrame, params: dict) -> Pipeline:
    model = Pipeline(
        steps=[
            ("preprocess", base.build_preprocessor()),
            ("model", make_regressor(params)),
        ]
    )
    model.fit(train_df[base.FEATURES], train_df[base.TARGET])
    return model


def tune_classifier(train_df: pd.DataFrame, val_df: pd.DataFrame, n_iter: int = 20) -> tuple[dict, pd.DataFrame]:
    y_train = high_eui_flag(train_df)
    y_val = high_eui_flag(val_df).to_numpy()
    negative = max(int((y_train == 0).sum()), 1)
    positive = max(int((y_train == 1).sum()), 1)
    imbalance_ratio = negative / positive
    param_dist = {
        **XGB_CLASSIFIER_PARAM_DIST_BASE,
        "scale_pos_weight": [1.0, round(imbalance_ratio * 0.5, 3), round(imbalance_ratio, 3), round(imbalance_ratio * 1.5, 3)],
    }

    rows = []
    best_params = None
    best_f1 = -math.inf
    samples = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=RANDOM_STATE))
    for i, params in enumerate(samples, start=1):
        model = fit_classifier(train_df, params)
        prob = model.predict_proba(val_df[base.FEATURES])[:, 1]
        threshold_rows = [classification_metrics(y_val, prob, threshold) for threshold in THRESHOLD_GRID]
        best_threshold_row = sorted(threshold_rows, key=lambda row: (row["f1"], row["recall"]), reverse=True)[0]
        row = {
            "iteration": i,
            "validation_roc_auc": safe_auc(y_val, prob),
            "validation_average_precision": safe_average_precision(y_val, prob),
            "best_threshold_by_f1": best_threshold_row["threshold"],
            "validation_precision_at_best_f1": best_threshold_row["precision"],
            "validation_recall_at_best_f1": best_threshold_row["recall"],
            "validation_f1_at_best_threshold": best_threshold_row["f1"],
            "params": json.dumps(params, sort_keys=True),
        }
        rows.append(row)
        if row["validation_f1_at_best_threshold"] > best_f1:
            best_f1 = row["validation_f1_at_best_threshold"]
            best_params = params
        print(f"Classifier random search {i:02d}/{n_iter}: best F1={row['validation_f1_at_best_threshold']:.4f}")

    assert best_params is not None
    return best_params, pd.DataFrame(rows).sort_values("validation_f1_at_best_threshold", ascending=False)


def tune_regressor(
    label: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_iter: int = 15,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    best_params = None
    best_rmse = math.inf
    samples = list(ParameterSampler(XGB_REGRESSOR_PARAM_DIST, n_iter=n_iter, random_state=RANDOM_STATE))

    for i, params in enumerate(samples, start=1):
        model = fit_regressor(train_df, params)
        pred = model.predict(val_df[base.FEATURES])
        row = {
            "regressor": label,
            "iteration": i,
            "validation_rmse": rmse(val_df[base.TARGET], pred),
            "validation_mae": float(mean_absolute_error(val_df[base.TARGET], pred)),
            "validation_r2": float(r2_score(val_df[base.TARGET], pred)),
            "params": json.dumps(params, sort_keys=True),
        }
        rows.append(row)
        if row["validation_rmse"] < best_rmse:
            best_rmse = row["validation_rmse"]
            best_params = params
        print(f"{label} regressor random search {i:02d}/{n_iter}: RMSE={row['validation_rmse']:.4f}")

    assert best_params is not None
    return best_params, pd.DataFrame(rows).sort_values("validation_rmse")


def route_predictions(
    df: pd.DataFrame,
    classifier: Pipeline,
    normal_regressor: Pipeline,
    high_regressor: Pipeline,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    high_prob = classifier.predict_proba(df[base.FEATURES])[:, 1]
    predicted_high = high_prob >= threshold
    normal_pred = normal_regressor.predict(df[base.FEATURES])
    high_pred = high_regressor.predict(df[base.FEATURES])
    routed_pred = normal_pred.copy()
    routed_pred[predicted_high] = high_pred[predicted_high]
    return routed_pred, predicted_high, high_prob, high_pred


def sharpen_probability(prob: np.ndarray, gamma: float) -> np.ndarray:
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    if gamma == 1.0:
        return prob
    positive = prob**gamma
    negative = (1 - prob) ** gamma
    return positive / (positive + negative)


def soft_predictions(
    df: pd.DataFrame,
    classifier: Pipeline,
    normal_regressor: Pipeline,
    high_regressor: Pipeline,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high_prob = classifier.predict_proba(df[base.FEATURES])[:, 1]
    high_weight = sharpen_probability(high_prob, gamma)
    normal_pred = normal_regressor.predict(df[base.FEATURES])
    high_pred = high_regressor.predict(df[base.FEATURES])
    blended_pred = (1 - high_weight) * normal_pred + high_weight * high_pred
    return blended_pred, high_prob, high_weight


def oracle_predictions(
    df: pd.DataFrame,
    normal_regressor: Pipeline,
    high_regressor: Pipeline,
) -> tuple[np.ndarray, np.ndarray]:
    actual_high = high_eui_flag(df).to_numpy().astype(bool)
    normal_pred = normal_regressor.predict(df[base.FEATURES])
    high_pred = high_regressor.predict(df[base.FEATURES])
    routed_pred = normal_pred.copy()
    routed_pred[actual_high] = high_pred[actual_high]
    return routed_pred, actual_high


def evaluate_soft_gammas(
    val_df: pd.DataFrame,
    classifier: Pipeline,
    normal_regressor: Pipeline,
    high_regressor: Pipeline,
) -> pd.DataFrame:
    y_class = high_eui_flag(val_df).to_numpy()
    high_prob = classifier.predict_proba(val_df[base.FEATURES])[:, 1]
    normal_pred = normal_regressor.predict(val_df[base.FEATURES])
    high_pred = high_regressor.predict(val_df[base.FEATURES])
    actual_high_mask = y_class.astype(bool)
    rows = []
    for gamma in SOFT_GAMMA_GRID:
        high_weight = sharpen_probability(high_prob, gamma)
        blended_pred = (1 - high_weight) * normal_pred + high_weight * high_pred
        rows.append(
            {
                "gamma": gamma,
                "mean_high_weight": float(high_weight.mean()),
                "mean_high_weight_actual_high": float(high_weight[actual_high_mask].mean()),
                "mean_high_weight_actual_normal": float(high_weight[~actual_high_mask].mean()),
                "overall_rmse": rmse(val_df[base.TARGET], blended_pred),
                "overall_mae": float(mean_absolute_error(val_df[base.TARGET], blended_pred)),
                "high150_rmse": rmse(val_df.loc[actual_high_mask, base.TARGET], blended_pred[actual_high_mask]),
                "high150_mae": float(mean_absolute_error(val_df.loc[actual_high_mask, base.TARGET], blended_pred[actual_high_mask])),
            }
        )
    return pd.DataFrame(rows)


def evaluate_thresholds(
    val_df: pd.DataFrame,
    classifier: Pipeline,
    normal_regressor: Pipeline,
    high_regressor: Pipeline,
) -> pd.DataFrame:
    y_class = high_eui_flag(val_df).to_numpy()
    high_prob = classifier.predict_proba(val_df[base.FEATURES])[:, 1]
    normal_pred = normal_regressor.predict(val_df[base.FEATURES])
    high_pred = high_regressor.predict(val_df[base.FEATURES])
    actual_high_mask = y_class.astype(bool)
    rows = []
    for threshold in THRESHOLD_GRID:
        predicted_high = high_prob >= threshold
        routed_pred = normal_pred.copy()
        routed_pred[predicted_high] = high_pred[predicted_high]
        class_row = classification_metrics(y_class, high_prob, threshold)
        rows.append(
            {
                **class_row,
                "overall_rmse": rmse(val_df[base.TARGET], routed_pred),
                "overall_mae": float(mean_absolute_error(val_df[base.TARGET], routed_pred)),
                "high150_rmse": rmse(val_df.loc[actual_high_mask, base.TARGET], routed_pred[actual_high_mask]),
                "high150_mae": float(mean_absolute_error(val_df.loc[actual_high_mask, base.TARGET], routed_pred[actual_high_mask])),
            }
        )
    return pd.DataFrame(rows)


def prediction_frame(
    df: pd.DataFrame,
    variant: str,
    prediction: np.ndarray,
    predicted_high: np.ndarray | None = None,
    high_probability: np.ndarray | None = None,
    soft_high_weight: np.ndarray | None = None,
) -> pd.DataFrame:
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
    out["prediction"] = prediction
    out["abs_error"] = (out["prediction"] - out[base.TARGET]).abs()
    out["squared_error"] = (out["prediction"] - out[base.TARGET]) ** 2
    out["bias_pred_minus_actual"] = out["prediction"] - out[base.TARGET]
    out["actual_high_eui_150"] = high_eui_flag(df).to_numpy()
    if predicted_high is not None:
        out["predicted_high_eui_150"] = predicted_high.astype(int)
    if high_probability is not None:
        out["high_eui_probability_150"] = high_probability
    if soft_high_weight is not None:
        out["soft_high_weight_150"] = soft_high_weight
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
        for subgroup, mask_fn in subgroup_defs.items():
            selected = group[mask_fn(group)]
            if selected.empty:
                continue
            rows.append(
                {
                    "variant": variant,
                    "subgroup": subgroup,
                    "n": len(selected),
                    "actual_mean_eui": float(selected[base.TARGET].mean()),
                    "predicted_mean_eui": float(selected["prediction"].mean()),
                    "mae": float(selected["abs_error"].mean()),
                    "rmse": rmse(selected[base.TARGET], selected["prediction"]),
                    "bias_mean_pred_minus_actual": float(selected["bias_pred_minus_actual"].mean()),
                }
            )
    return pd.DataFrame(rows)


def add_delta_vs_reference(subgroup_df: pd.DataFrame, reference: str) -> pd.DataFrame:
    ref = subgroup_df[subgroup_df["variant"].eq(reference)]
    rows = []
    for variant in subgroup_df["variant"].unique():
        if variant == reference:
            continue
        merged = subgroup_df[subgroup_df["variant"].eq(variant)].merge(ref, on="subgroup", suffixes=("", "_reference"))
        merged["reference_variant"] = reference
        merged["mae_change"] = merged["mae"] - merged["mae_reference"]
        merged["mae_change_pct"] = merged["mae_change"] / merged["mae_reference"] * 100
        merged["rmse_change"] = merged["rmse"] - merged["rmse_reference"]
        merged["rmse_change_pct"] = merged["rmse_change"] / merged["rmse_reference"] * 100
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def load_prior_comparison_predictions(test_df: pd.DataFrame) -> pd.DataFrame:
    path = OUTPUT_DIR / "high_eui_weather_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    prior = pd.read_csv(path)
    keep = [
        "weather_xgb_reference",
        "weather_high_features_xgb",
        "weather_high_features_weighted_xgb",
    ]
    prior = prior[prior["variant"].isin(keep)].copy()
    prior["actual_high_eui_150"] = prior[base.TARGET].ge(HIGH_EUI_THRESHOLD).astype(int)
    return prior


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
    variant_order = [
        "weather_xgb_reference",
        "weather_high_features_xgb",
        "weather_high_features_weighted_xgb",
        "two_stage_soft_probability",
        "two_stage_soft_overall_gamma",
        "two_stage_soft_high_gamma",
        "two_stage_f1_threshold",
        "two_stage_high_rmse_threshold",
        "two_stage_oracle_route",
    ]
    pivot = summary.pivot(index="subgroup", columns="variant", values=metric).reindex(ordered_groups)
    pivot = pivot[[col for col in variant_order if col in pivot.columns]]
    ax = pivot.plot(
        kind="bar",
        figsize=(16, 7),
        color=["#C44E52", "#2F6F73", "#7A5195", "#72B7B2", "#4C78A8", "#F58518", "#9467BD", "#B279A2", "#54A24B"],
    )
    ax.set_xlabel("Subgroup")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Two-Stage High-EUI Experiment: Subgroup {metric.upper()}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Variant")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"two_stage_high_eui_subgroup_{metric}.png", dpi=180)
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


def write_report(
    metrics: pd.DataFrame,
    subgroup: pd.DataFrame,
    delta: pd.DataFrame,
    threshold_tuning: pd.DataFrame,
    soft_tuning: pd.DataFrame,
    classifier_tuning: pd.DataFrame,
    selected_thresholds: dict,
    selected_gammas: dict,
) -> None:
    focus_groups = [
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
    focus_variants = [
        "weather_xgb_reference",
        "weather_high_features_weighted_xgb",
        "two_stage_soft_probability",
        "two_stage_soft_overall_gamma",
        "two_stage_soft_high_gamma",
        "two_stage_f1_threshold",
        "two_stage_high_rmse_threshold",
        "two_stage_oracle_route",
    ]
    focus = subgroup[subgroup["subgroup"].isin(focus_groups) & subgroup["variant"].isin(focus_variants)]
    two_stage_delta = delta[delta["variant"].str.startswith("two_stage") & delta["subgroup"].isin(focus_groups)]
    report = f"""# Two-Stage High-EUI Model Experiment

This experiment tests whether a two-stage model can reduce errors for high-EUI buildings.

## Setup

- Feature base: `base + weather + high-EUI historical features`
- High-EUI threshold: Site EUI >= {HIGH_EUI_THRESHOLD}
- Train: 2018-2021
- Validation: 2022, used for classifier/regressor tuning and threshold selection
- Test: 2023, used only for final evaluation

## Two-Stage Logic

1. Stage 1 classifier predicts whether a building is high-EUI.
2. Hard routing sends each record to either a normal-EUI regressor or a high-EUI regressor.
3. Soft routing blends both regressors: `prediction = (1 - p_high) * normal_prediction + p_high * high_prediction`.
4. `two_stage_oracle_route` is not deployable; it uses the true high-EUI label on test data only as an upper-bound check.

## Selected Thresholds

| variant | validation threshold |
|---|---:|
| two_stage_f1_threshold | {selected_thresholds["f1"]:.2f} |
| two_stage_high_rmse_threshold | {selected_thresholds["high_rmse"]:.2f} |

## Selected Soft-Routing Gamma

| variant | validation gamma |
|---|---:|
| two_stage_soft_probability | 1.00 |
| two_stage_soft_overall_gamma | {selected_gammas["overall"]:.2f} |
| two_stage_soft_high_gamma | {selected_gammas["high_rmse"]:.2f} |

## Overall Test Metrics

{md_table(metrics, ["variant", "rmse", "mae", "r2", "bias_mean_pred_minus_actual"], digits=4)}

## Subgroup Error

{md_table(focus.sort_values(["subgroup", "variant"]), ["variant", "subgroup", "n", "mae", "rmse", "bias_mean_pred_minus_actual"], digits=3)}

## Change vs Weather Reference

{md_table(two_stage_delta.sort_values(["subgroup", "variant"]), ["variant", "subgroup", "n", "mae_reference", "mae", "mae_change", "rmse_reference", "rmse", "rmse_change"], digits=3)}

## Best Classifier Tuning Rows

{md_table(classifier_tuning.head(5), ["iteration", "validation_roc_auc", "validation_average_precision", "best_threshold_by_f1", "validation_precision_at_best_f1", "validation_recall_at_best_f1", "validation_f1_at_best_threshold"], digits=4)}

## Threshold Tuning

{md_table(threshold_tuning.sort_values("high150_rmse").head(8), ["threshold", "precision", "recall", "f1", "predicted_high_rate", "overall_rmse", "overall_mae", "high150_rmse", "high150_mae"], digits=4)}

## Soft Routing Gamma Tuning

{md_table(soft_tuning.sort_values("overall_rmse"), ["gamma", "mean_high_weight", "mean_high_weight_actual_high", "mean_high_weight_actual_normal", "overall_rmse", "overall_mae", "high150_rmse", "high150_mae"], digits=4)}

## Visualizations

![Subgroup MAE](../outputs/plots/two_stage_high_eui_subgroup_mae.png)

![Subgroup RMSE](../outputs/plots/two_stage_high_eui_subgroup_rmse.png)
"""
    (REPORT_DIR / "two_stage_high_eui_experiment.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    weather_df = high_exp.load_weather_base()
    high_df = high_exp.add_high_eui_historical_features(weather_df)

    train_df = high_df[high_df["data_year"].between(2018, 2021)].copy()
    val_df = high_df[high_df["data_year"].eq(2022)].copy()
    trainval_df = high_df[high_df["data_year"].between(2018, 2022)].copy()
    test_df = high_df[high_df["data_year"].eq(2023)].copy()

    original = high_exp.set_feature_globals(high_exp.HIGH_EUI_FEATURES)
    try:
        classifier_params, classifier_tuning = tune_classifier(train_df, val_df)

        train_normal = train_df[train_df[base.TARGET] < HIGH_EUI_THRESHOLD].copy()
        val_normal = val_df[val_df[base.TARGET] < HIGH_EUI_THRESHOLD].copy()
        train_high = train_df[train_df[base.TARGET] >= HIGH_EUI_THRESHOLD].copy()
        val_high = val_df[val_df[base.TARGET] >= HIGH_EUI_THRESHOLD].copy()
        normal_params, normal_tuning = tune_regressor("normal_eui", train_normal, val_normal)
        high_params, high_tuning = tune_regressor("high_eui", train_high, val_high)

        validation_classifier = fit_classifier(train_df, classifier_params)
        validation_normal_regressor = fit_regressor(train_normal, normal_params)
        validation_high_regressor = fit_regressor(train_high, high_params)
        threshold_tuning = evaluate_thresholds(
            val_df,
            validation_classifier,
            validation_normal_regressor,
            validation_high_regressor,
        )
        soft_tuning = evaluate_soft_gammas(
            val_df,
            validation_classifier,
            validation_normal_regressor,
            validation_high_regressor,
        )
        f1_threshold = float(threshold_tuning.sort_values(["f1", "recall"], ascending=False).iloc[0]["threshold"])
        high_rmse_threshold = float(threshold_tuning.sort_values(["high150_rmse", "overall_rmse"]).iloc[0]["threshold"])
        soft_overall_gamma = float(soft_tuning.sort_values(["overall_rmse", "high150_rmse"]).iloc[0]["gamma"])
        soft_high_gamma = float(soft_tuning.sort_values(["high150_rmse", "overall_rmse"]).iloc[0]["gamma"])

        trainval_normal = trainval_df[trainval_df[base.TARGET] < HIGH_EUI_THRESHOLD].copy()
        trainval_high = trainval_df[trainval_df[base.TARGET] >= HIGH_EUI_THRESHOLD].copy()
        final_classifier = fit_classifier(trainval_df, classifier_params)
        final_normal_regressor = fit_regressor(trainval_normal, normal_params)
        final_high_regressor = fit_regressor(trainval_high, high_params)

        f1_pred, f1_route, f1_prob, _ = route_predictions(
            test_df,
            final_classifier,
            final_normal_regressor,
            final_high_regressor,
            f1_threshold,
        )
        high_rmse_pred, high_rmse_route, high_rmse_prob, _ = route_predictions(
            test_df,
            final_classifier,
            final_normal_regressor,
            final_high_regressor,
            high_rmse_threshold,
        )
        soft_probability_pred, soft_probability_prob, soft_probability_weight = soft_predictions(
            test_df,
            final_classifier,
            final_normal_regressor,
            final_high_regressor,
            gamma=1.0,
        )
        soft_overall_pred, soft_overall_prob, soft_overall_weight = soft_predictions(
            test_df,
            final_classifier,
            final_normal_regressor,
            final_high_regressor,
            gamma=soft_overall_gamma,
        )
        soft_high_pred, soft_high_prob, soft_high_weight = soft_predictions(
            test_df,
            final_classifier,
            final_normal_regressor,
            final_high_regressor,
            gamma=soft_high_gamma,
        )
        oracle_pred, oracle_route = oracle_predictions(test_df, final_normal_regressor, final_high_regressor)
    finally:
        high_exp.restore_feature_globals(original)

    two_stage_predictions = pd.concat(
        [
            prediction_frame(
                test_df,
                "two_stage_soft_probability",
                soft_probability_pred,
                high_probability=soft_probability_prob,
                soft_high_weight=soft_probability_weight,
            ),
            prediction_frame(
                test_df,
                "two_stage_soft_overall_gamma",
                soft_overall_pred,
                high_probability=soft_overall_prob,
                soft_high_weight=soft_overall_weight,
            ),
            prediction_frame(
                test_df,
                "two_stage_soft_high_gamma",
                soft_high_pred,
                high_probability=soft_high_prob,
                soft_high_weight=soft_high_weight,
            ),
            prediction_frame(test_df, "two_stage_f1_threshold", f1_pred, f1_route, f1_prob),
            prediction_frame(test_df, "two_stage_high_rmse_threshold", high_rmse_pred, high_rmse_route, high_rmse_prob),
            prediction_frame(test_df, "two_stage_oracle_route", oracle_pred, oracle_route),
        ],
        ignore_index=True,
    )
    prior_predictions = load_prior_comparison_predictions(test_df)
    comparison_predictions = pd.concat([prior_predictions, two_stage_predictions], ignore_index=True)

    metrics = summarize_metrics(comparison_predictions)
    subgroup = summarize_subgroups(comparison_predictions)
    delta = add_delta_vs_reference(subgroup, "weather_xgb_reference")
    regressor_tuning = pd.concat([normal_tuning, high_tuning], ignore_index=True)

    two_stage_predictions.to_csv(OUTPUT_DIR / "two_stage_high_eui_predictions.csv", index=False, encoding="utf-8-sig")
    comparison_predictions.to_csv(OUTPUT_DIR / "two_stage_high_eui_comparison_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUTPUT_DIR / "two_stage_high_eui_metrics.csv", index=False, encoding="utf-8-sig")
    subgroup.to_csv(OUTPUT_DIR / "two_stage_high_eui_subgroup_error.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(OUTPUT_DIR / "two_stage_high_eui_subgroup_delta.csv", index=False, encoding="utf-8-sig")
    classifier_tuning.to_csv(OUTPUT_DIR / "two_stage_high_eui_classifier_tuning.csv", index=False, encoding="utf-8-sig")
    regressor_tuning.to_csv(OUTPUT_DIR / "two_stage_high_eui_regressor_tuning.csv", index=False, encoding="utf-8-sig")
    threshold_tuning.to_csv(OUTPUT_DIR / "two_stage_high_eui_threshold_tuning.csv", index=False, encoding="utf-8-sig")
    soft_tuning.to_csv(OUTPUT_DIR / "two_stage_high_eui_soft_tuning.csv", index=False, encoding="utf-8-sig")

    plot_subgroup(subgroup, "mae")
    plot_subgroup(subgroup, "rmse")
    write_report(
        metrics,
        subgroup,
        delta,
        threshold_tuning,
        soft_tuning,
        classifier_tuning,
        {"f1": f1_threshold, "high_rmse": high_rmse_threshold},
        {"overall": soft_overall_gamma, "high_rmse": soft_high_gamma},
    )

    print("Done two-stage high-EUI experiment.")
    print(metrics.to_string(index=False))
    print(threshold_tuning.sort_values("high150_rmse").head(5).to_string(index=False))
    print(soft_tuning.sort_values("overall_rmse").to_string(index=False))


if __name__ == "__main__":
    main()
