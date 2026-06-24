from __future__ import annotations

import json
import math
import random
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

try:
    from document_generation.main_report_generator import write_report as write_main_report
except ModuleNotFoundError:
    write_main_report = None


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = ROOT / "reports"

SOCRATA_ENDPOINT = "https://data.cityofchicago.org/resource/xq83-jr8c.json"
DATASET_PAGE = (
    "https://data.cityofchicago.org/Environment-Sustainable-Development/"
    "Chicago-Energy-Benchmarking/xq83-jr8c/about_data"
)

TARGET = "site_eui_kbtu_sq_ft"
GROSS_FLOOR = "gross_floor_area_buildings_sq_ft"

NUMERIC_FEATURES = [
    "data_year",
    "years_since_2018",
    GROSS_FLOOR,
    "log_floor_area",
    "year_built",
    "building_age",
    "of_buildings",
    "latitude",
    "longitude",
    "prev_year_site_eui",
    "prior_building_mean_eui",
    "prior_community_mean_eui",
    "prior_property_type_mean_eui",
    "prior_comm_property_mean_eui",
    "prior_community_high_eui_ratio",
    "community_year_record_count",
    "community_year_building_count",
    "community_year_avg_floor_area",
    "property_year_record_count",
]

CATEGORICAL_FEATURES = [
    "community_area_clean",
    "primary_property_type_clean",
    "zip_code_clean",
    "regulation_stage",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LEAKAGE_FIELDS_EXCLUDED = [
    "energy_star_score",
    "electricity_use_kbtu",
    "natural_gas_use_kbtu",
    "district_steam_use_kbtu",
    "district_chilled_water_use_kbtu",
    "all_other_fuel_use_kbtu",
    "source_eui_kbtu_sq_ft",
    "weather_normalized_site_eui_kbtu_sq_ft",
    "weather_normalized_source_eui_kbtu_sq_ft",
    "total_ghg_emissions_metric_tons_co2e",
    "ghg_intensity_kg_co2e_sq_ft",
    "water_use_kgal",
]


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, PLOT_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def download_data() -> pd.DataFrame:
    params = {
        "$limit": "50000",
        "$where": "data_year between 2018 and 2023",
        "$order": "data_year,id",
    }
    url = SOCRATA_ENDPOINT + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as response:
        rows = json.load(response)
    df = pd.DataFrame(rows)
    raw_path = RAW_DIR / "chicago_energy_benchmarking_2018_2023_raw.csv"
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    return df


def to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan


def clean_category(series: pd.Series, unknown: str = "UNKNOWN") -> pd.Series:
    clean = series.astype("string").str.strip()
    clean = clean.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return clean.fillna(unknown)


def add_historical_mean(
    df: pd.DataFrame,
    key_cols: list[str],
    out_col: str,
    value_col: str = TARGET,
) -> pd.DataFrame:
    yearly = (
        df.groupby(key_cols + ["data_year"], dropna=False)[value_col]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values(key_cols + ["data_year"])
    )
    grouped = yearly.groupby(key_cols, dropna=False)
    yearly["prev_sum"] = grouped["sum"].cumsum() - yearly["sum"]
    yearly["prev_count"] = grouped["count"].cumsum() - yearly["count"]
    yearly[out_col] = yearly["prev_sum"] / yearly["prev_count"].replace(0, np.nan)
    return df.merge(yearly[key_cols + ["data_year", out_col]], on=key_cols + ["data_year"], how="left")


def add_historical_ratio(
    df: pd.DataFrame,
    key_cols: list[str],
    out_col: str,
    flag_col: str,
) -> pd.DataFrame:
    yearly = (
        df.groupby(key_cols + ["data_year"], dropna=False)[flag_col]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values(key_cols + ["data_year"])
    )
    grouped = yearly.groupby(key_cols, dropna=False)
    yearly["prev_sum"] = grouped["sum"].cumsum() - yearly["sum"]
    yearly["prev_count"] = grouped["count"].cumsum() - yearly["count"]
    yearly[out_col] = yearly["prev_sum"] / yearly["prev_count"].replace(0, np.nan)
    return df.merge(yearly[key_cols + ["data_year", out_col]], on=key_cols + ["data_year"], how="left")


def preprocess_and_engineer(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    summary: dict[str, object] = {"raw_rows_2018_2023": int(len(df))}

    numeric_cols = [
        "data_year",
        "id",
        TARGET,
        "source_eui_kbtu_sq_ft",
        GROSS_FLOOR,
        "year_built",
        "of_buildings",
        "latitude",
        "longitude",
        "energy_star_score",
        "electricity_use_kbtu",
        "natural_gas_use_kbtu",
        "total_ghg_emissions_metric_tons_co2e",
        "ghg_intensity_kg_co2e_sq_ft",
    ]
    to_numeric(df, numeric_cols)

    df = df[df["data_year"].between(2018, 2023)].copy()
    summary["rows_after_year_filter"] = int(len(df))

    df["reporting_status"] = clean_category(df.get("reporting_status", pd.Series(index=df.index)))
    df = df[df["reporting_status"].isin(["Submitted", "Submitted Data"])].copy()
    summary["rows_after_reporting_status"] = int(len(df))

    df["community_area_clean"] = clean_category(df.get("community_area", pd.Series(index=df.index))).str.upper()
    df["primary_property_type_clean"] = clean_category(
        df.get("primary_property_type", pd.Series(index=df.index))
    )
    df["zip_code_clean"] = clean_category(df.get("zip_code", pd.Series(index=df.index)))

    core_cols = [
        TARGET,
        GROSS_FLOOR,
        "community_area_clean",
        "primary_property_type_clean",
        "latitude",
        "longitude",
    ]
    for col in ["community_area_clean", "primary_property_type_clean"]:
        df.loc[df[col].eq("UNKNOWN"), col] = pd.NA
    df = df.dropna(subset=core_cols).copy()
    summary["rows_after_core_non_null"] = int(len(df))

    before_outlier = len(df)
    df = df[df[TARGET].between(1, 500)].copy()
    summary["rows_removed_by_outlier_rule_1_to_500"] = int(before_outlier - len(df))
    summary["rows_after_outlier_filter"] = int(len(df))
    summary["distinct_buildings_after_filter"] = int(df["id"].nunique())

    df["primary_property_type_clean"] = df["primary_property_type_clean"].astype(str).str.strip()
    df["community_area_clean"] = df["community_area_clean"].astype(str).str.strip().str.upper()
    df["zip_code_clean"] = df["zip_code_clean"].astype(str).str.strip()

    df["years_since_2018"] = df["data_year"] - 2018
    df["log_floor_area"] = np.log1p(df[GROSS_FLOOR].clip(lower=0))
    df["building_age"] = df["data_year"] - df["year_built"]
    df.loc[(df["building_age"] < 0) | (df["building_age"] > 250), "building_age"] = np.nan
    df["of_buildings"] = df["of_buildings"].clip(lower=1)
    df["regulation_stage"] = np.select(
        [
            df["data_year"].isin([2018, 2019]),
            df["data_year"].isin([2020, 2021]),
            df["data_year"].isin([2022, 2023]),
        ],
        ["pre_covid_mature_policy", "covid_period", "post_covid_recent"],
        default="unknown",
    )

    df = df.sort_values(["id", "data_year"]).copy()
    df["prev_observed_site_eui"] = df.groupby("id")[TARGET].shift(1)
    df["prev_observed_year"] = df.groupby("id")["data_year"].shift(1)
    df["prev_year_site_eui"] = np.where(
        df["data_year"] - df["prev_observed_year"] == 1,
        df["prev_observed_site_eui"],
        np.nan,
    )
    df["prior_building_mean_eui"] = df.groupby("id", group_keys=False)[TARGET].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    df = add_historical_mean(df, ["community_area_clean"], "prior_community_mean_eui")
    df = add_historical_mean(df, ["primary_property_type_clean"], "prior_property_type_mean_eui")
    df = add_historical_mean(
        df,
        ["community_area_clean", "primary_property_type_clean"],
        "prior_comm_property_mean_eui",
    )
    df["high_eui_flag"] = (df[TARGET] >= 100).astype(int)
    df = add_historical_ratio(
        df,
        ["community_area_clean"],
        "prior_community_high_eui_ratio",
        "high_eui_flag",
    )

    community_year = (
        df.groupby(["community_area_clean", "data_year"], dropna=False)
        .agg(
            community_year_record_count=("id", "size"),
            community_year_building_count=("id", "nunique"),
            community_year_avg_floor_area=(GROSS_FLOOR, "mean"),
        )
        .reset_index()
    )
    property_year = (
        df.groupby(["primary_property_type_clean", "data_year"], dropna=False)
        .agg(property_year_record_count=("id", "size"))
        .reset_index()
    )
    df = df.merge(community_year, on=["community_area_clean", "data_year"], how="left")
    df = df.merge(property_year, on=["primary_property_type_clean", "data_year"], how="left")

    keep_cols = [
        "id",
        "property_name",
        "address",
        "reporting_status",
        "community_area",
        "primary_property_type",
        TARGET,
    ] + FEATURES
    keep_cols = [col for col in keep_cols if col in df.columns]
    model_df = df[keep_cols].copy()

    processed_path = PROCESSED_DIR / "chicago_energy_modeling_table_2018_2023.csv"
    model_df.to_csv(processed_path, index=False, encoding="utf-8-sig")

    summary["rows_by_year_after_filter"] = {
        str(int(k)): int(v) for k, v in model_df["data_year"].value_counts().sort_index().items()
    }
    summary["target_summary_after_filter"] = {
        "mean": float(model_df[TARGET].mean()),
        "median": float(model_df[TARGET].median()),
        "std": float(model_df[TARGET].std()),
        "min": float(model_df[TARGET].min()),
        "max": float(model_df[TARGET].max()),
    }
    return model_df, summary


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def metrics_dict(split: str, model: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "split": split,
        "model": model,
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fallback_group_mean_predict(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    keys: list[str],
) -> np.ndarray:
    global_mean = train_df[TARGET].mean()
    group_mean = train_df.groupby(keys, dropna=False)[TARGET].mean().rename("group_mean").reset_index()
    prop_mean = (
        train_df.groupby(["primary_property_type_clean"], dropna=False)[TARGET]
        .mean()
        .rename("property_mean")
        .reset_index()
    )
    comm_mean = (
        train_df.groupby(["community_area_clean"], dropna=False)[TARGET]
        .mean()
        .rename("community_mean")
        .reset_index()
    )
    pred_cols = list(dict.fromkeys(keys + ["primary_property_type_clean", "community_area_clean"]))
    pred_df = eval_df[pred_cols].copy()
    pred_df = pred_df.merge(group_mean, on=keys, how="left")
    pred_df = pred_df.merge(prop_mean, on=["primary_property_type_clean"], how="left")
    pred_df = pred_df.merge(comm_mean, on=["community_area_clean"], how="left")
    return (
        pred_df["group_mean"]
        .fillna(pred_df["property_mean"])
        .fillna(pred_df["community_mean"])
        .fillna(global_mean)
        .to_numpy()
    )


def fit_ridge(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[np.ndarray, Pipeline]:
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]
    )
    pipeline.fit(train_df[FEATURES], train_df[TARGET])
    pred = pipeline.predict(eval_df[FEATURES])
    return pred, pipeline


def tune_random_search(
    name: str,
    estimator_factory,
    param_distributions: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_iter: int,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    best_params: dict | None = None
    best_rmse = math.inf

    samples = list(
        ParameterSampler(
            param_distributions,
            n_iter=n_iter,
            random_state=RANDOM_STATE,
        )
    )
    for i, params in enumerate(samples, start=1):
        model = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", estimator_factory(**params)),
            ]
        )
        model.fit(train_df[FEATURES], train_df[TARGET])
        pred = model.predict(val_df[FEATURES])
        row = {
            "model": name,
            "iteration": i,
            "validation_rmse": rmse(val_df[TARGET], pred),
            "validation_mae": float(mean_absolute_error(val_df[TARGET], pred)),
            "validation_r2": float(r2_score(val_df[TARGET], pred)),
            "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
        }
        rows.append(row)
        if row["validation_rmse"] < best_rmse:
            best_rmse = row["validation_rmse"]
            best_params = params
        print(f"{name} random search {i:02d}/{n_iter}: RMSE={row['validation_rmse']:.4f}")

    assert best_params is not None
    return best_params, pd.DataFrame(rows).sort_values("validation_rmse")


def fit_pipeline(name: str, params: dict, train_df: pd.DataFrame) -> Pipeline:
    if name == "Random Forest":
        estimator = RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )
    elif name == "XGBoost":
        estimator = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            eval_metric="rmse",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            **params,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )
    pipeline.fit(train_df[FEATURES], train_df[TARGET])
    return pipeline


def evaluate_by_group(
    df: pd.DataFrame,
    group_col: str,
    min_n: int = 15,
) -> pd.DataFrame:
    grouped = []
    for key, group in df.groupby(group_col, dropna=False):
        if len(group) < min_n:
            continue
        grouped.append(
            {
                group_col: key,
                "n": int(len(group)),
                "actual_mean_eui": float(group[TARGET].mean()),
                "predicted_mean_eui": float(group["prediction"].mean()),
                "rmse": rmse(group[TARGET], group["prediction"]),
                "mae": float(mean_absolute_error(group[TARGET], group["prediction"])),
                "bias_mean_pred_minus_actual": float(group["prediction"].mean() - group[TARGET].mean()),
            }
        )
    return pd.DataFrame(grouped).sort_values("mae", ascending=False)


def save_bar_plot(df: pd.DataFrame, label_col: str, value_col: str, title: str, path: Path, top_n: int = 15) -> None:
    plot_df = df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df[label_col].astype(str), plot_df[value_col], color="#2F6F73")
    plt.xlabel(value_col.upper())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_actual_vs_predicted(test_pred: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(test_pred[TARGET], test_pred["prediction"], s=14, alpha=0.45, color="#375A7F")
    low = min(test_pred[TARGET].min(), test_pred["prediction"].min())
    high = max(test_pred[TARGET].max(), test_pred["prediction"].max())
    plt.plot([low, high], [low, high], color="#C44E52", linewidth=2)
    plt.xlabel("Actual Site EUI")
    plt.ylabel("Predicted Site EUI")
    plt.title("Actual vs Predicted Site EUI, Test Year 2023")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_residual_plots(test_pred: pd.DataFrame, path: Path) -> None:
    residual = test_pred["prediction"] - test_pred[TARGET]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(test_pred["prediction"], residual, s=14, alpha=0.45, color="#3B6EA8")
    axes[0].axhline(0, color="#C44E52", linewidth=2)
    axes[0].set_xlabel("Predicted Site EUI")
    axes[0].set_ylabel("Residual: predicted - actual")
    axes[0].set_title("Residual vs Predicted")
    axes[1].hist(residual, bins=45, color="#6A994E", edgecolor="white")
    axes[1].axvline(0, color="#C44E52", linewidth=2)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def transformed_feature_to_original(feature_name: str) -> str:
    if "__" in feature_name:
        feature_name = feature_name.split("__", 1)[1]
    for col in CATEGORICAL_FEATURES:
        if feature_name.startswith(col + "_"):
            return col
    return feature_name


def feature_importance_table(model: Pipeline) -> pd.DataFrame:
    names = model.named_steps["preprocess"].get_feature_names_out()
    estimator = model.named_steps["model"]
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame()
    raw = pd.DataFrame({"transformed_feature": names, "importance": importances})
    raw["feature"] = raw["transformed_feature"].map(transformed_feature_to_original)
    grouped = raw.groupby("feature", as_index=False)["importance"].sum()
    grouped["importance"] = grouped["importance"] / grouped["importance"].sum()
    return grouped.sort_values("importance", ascending=False)


def save_feature_importance_plot(fi: pd.DataFrame, path: Path, top_n: int = 20) -> None:
    plot_df = fi.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["feature"], plot_df["importance"], color="#7A5195")
    plt.xlabel("Aggregated importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()



def main() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    ensure_dirs()

    print("Downloading Chicago Energy Benchmarking data...")
    raw = download_data()
    print("Preprocessing and feature engineering...")
    df, summary = preprocess_and_engineer(raw)

    train_df = df[df["data_year"].between(2018, 2021)].copy()
    val_df = df[df["data_year"].eq(2022)].copy()
    test_df = df[df["data_year"].eq(2023)].copy()
    trainval_df = df[df["data_year"].between(2018, 2022)].copy()

    y_val = val_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    validation_rows = []
    test_rows = []

    print("Evaluating baselines...")
    val_global_pred = np.full(len(val_df), train_df[TARGET].mean())
    test_global_pred = np.full(len(test_df), trainval_df[TARGET].mean())
    validation_rows.append(metrics_dict("validation", "Global Mean Baseline", y_val, val_global_pred))
    test_rows.append(metrics_dict("test", "Global Mean Baseline", y_test, test_global_pred))

    group_keys = ["community_area_clean", "primary_property_type_clean"]
    val_group_pred = fallback_group_mean_predict(train_df, val_df, group_keys)
    test_group_pred = fallback_group_mean_predict(trainval_df, test_df, group_keys)
    validation_rows.append(metrics_dict("validation", "Group Mean Baseline", y_val, val_group_pred))
    test_rows.append(metrics_dict("test", "Group Mean Baseline", y_test, test_group_pred))

    val_ridge_pred, _ = fit_ridge(train_df, val_df)
    test_ridge_pred, _ = fit_ridge(trainval_df, test_df)
    validation_rows.append(metrics_dict("validation", "Ridge Regression", y_val, val_ridge_pred))
    test_rows.append(metrics_dict("test", "Ridge Regression", y_test, test_ridge_pred))

    print("Running Random Forest random search...")
    rf_param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [8, 12, 16, 24, None],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_features": ["sqrt", 0.5, 0.8],
    }
    rf_factory = lambda **params: RandomForestRegressor(  # noqa: E731
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )
    best_rf_params, rf_tuning = tune_random_search(
        "Random Forest",
        rf_factory,
        rf_param_dist,
        train_df,
        val_df,
        n_iter=20,
    )

    print("Running XGBoost random search...")
    xgb_param_dist = {
        "n_estimators": [200, 400, 600],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_lambda": [1, 5, 10],
        "min_child_weight": [1, 3, 5],
    }
    xgb_factory = lambda **params: XGBRegressor(  # noqa: E731
        objective="reg:squarederror",
        tree_method="hist",
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **params,
    )
    best_xgb_params, xgb_tuning = tune_random_search(
        "XGBoost",
        xgb_factory,
        xgb_param_dist,
        train_df,
        val_df,
        n_iter=25,
    )

    tuning_results = pd.concat([rf_tuning, xgb_tuning], ignore_index=True)
    tuning_results.to_csv(OUTPUT_DIR / "model_tuning_results.csv", index=False, encoding="utf-8-sig")

    best_params = {"Random Forest": best_rf_params, "XGBoost": best_xgb_params}
    validation_best = {
        "Random Forest": float(rf_tuning.iloc[0]["validation_rmse"]),
        "XGBoost": float(xgb_tuning.iloc[0]["validation_rmse"]),
    }
    best_model_name = min(validation_best, key=validation_best.get)

    print("Fitting final tree models and evaluating test set...")
    final_models: dict[str, Pipeline] = {}
    for name, params in best_params.items():
        val_model = fit_pipeline(name, params, train_df)
        val_pred = val_model.predict(val_df[FEATURES])
        validation_rows.append(metrics_dict("validation", name, y_val, val_pred))

        final_model = fit_pipeline(name, params, trainval_df)
        final_models[name] = final_model
        test_pred = final_model.predict(test_df[FEATURES])
        test_rows.append(metrics_dict("test", name, y_test, test_pred))

    validation_metrics = pd.DataFrame(validation_rows)
    test_metrics = pd.DataFrame(test_rows)
    validation_metrics.to_csv(OUTPUT_DIR / "metrics_validation.csv", index=False, encoding="utf-8-sig")
    test_metrics.to_csv(OUTPUT_DIR / "metrics_test.csv", index=False, encoding="utf-8-sig")

    best_model = final_models[best_model_name]
    best_test_pred = best_model.predict(test_df[FEATURES])
    test_pred_df = test_df.copy()
    test_pred_df["prediction"] = best_test_pred
    test_pred_df["residual_pred_minus_actual"] = test_pred_df["prediction"] - test_pred_df[TARGET]
    test_pred_df.to_csv(OUTPUT_DIR / "test_predictions_2023.csv", index=False, encoding="utf-8-sig")

    community_error = evaluate_by_group(test_pred_df, "community_area_clean", min_n=15)
    building_error = evaluate_by_group(test_pred_df, "primary_property_type_clean", min_n=15)
    community_error.to_csv(OUTPUT_DIR / "community_error_test.csv", index=False, encoding="utf-8-sig")
    building_error.to_csv(OUTPUT_DIR / "building_type_error_test.csv", index=False, encoding="utf-8-sig")

    save_actual_vs_predicted(test_pred_df, PLOT_DIR / "actual_vs_predicted_test.png")
    save_residual_plots(test_pred_df, PLOT_DIR / "residual_analysis_test.png")
    save_bar_plot(
        community_error,
        "community_area_clean",
        "mae",
        "Community Area Error, Test Year 2023",
        PLOT_DIR / "community_error_mae_top15.png",
    )
    save_bar_plot(
        building_error,
        "primary_property_type_clean",
        "mae",
        "Building Type Error, Test Year 2023",
        PLOT_DIR / "building_type_error_mae_top15.png",
    )

    feature_importance = feature_importance_table(best_model)
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    save_feature_importance_plot(feature_importance, PLOT_DIR / "feature_importance_top20.png")

    summary["split_rows"] = {
        "train_2018_2021": int(len(train_df)),
        "validation_2022": int(len(val_df)),
        "test_2023": int(len(test_df)),
    }
    (OUTPUT_DIR / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if write_main_report is not None:
        write_main_report(
            summary=summary,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            best_params=best_params,
            best_model_name=best_model_name,
            community_error=community_error,
            building_error=building_error,
            feature_importance=feature_importance,
            tuning_results=tuning_results.sort_values(["model", "validation_rmse"]),
        )

    print("Done.")
    print(f"Best model by validation RMSE: {best_model_name}")
    print(test_metrics.sort_values("rmse").to_string(index=False))


if __name__ == "__main__":
    main()
