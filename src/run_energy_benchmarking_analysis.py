from __future__ import annotations

import json
import math
import random
import textwrap
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


def markdown_table(df: pd.DataFrame, columns: list[str], float_digits: int = 4) -> str:
    out = df[columns].copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].map(lambda x: f"{x:.{float_digits}f}" if pd.notna(x) else "")
    out = out.fillna("")
    headers = [str(col) for col in out.columns]
    rows = [[str(value) for value in row] for row in out.to_numpy()]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    return "\n".join([fmt_row(headers), separator, *[fmt_row(row) for row in rows]])


def write_report(
    summary: dict,
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    best_params: dict,
    best_model_name: str,
    community_error: pd.DataFrame,
    building_error: pd.DataFrame,
    feature_importance: pd.DataFrame,
    tuning_results: pd.DataFrame,
) -> None:
    report_path = REPORT_DIR / "chicago_energy_benchmarking_report.md"

    selected_test = test_metrics.loc[test_metrics["model"].eq(best_model_name)].iloc[0]
    best_test_overall = test_metrics.sort_values("rmse").iloc[0]
    baseline_test = test_metrics.loc[test_metrics["model"].eq("Group Mean Baseline")].iloc[0]
    selected_rmse_improvement = (baseline_test["rmse"] - selected_test["rmse"]) / baseline_test["rmse"] * 100
    selected_mae_improvement = (baseline_test["mae"] - selected_test["mae"]) / baseline_test["mae"] * 100
    best_test_rmse_improvement = (baseline_test["rmse"] - best_test_overall["rmse"]) / baseline_test["rmse"] * 100
    best_test_mae_improvement = (baseline_test["mae"] - best_test_overall["mae"]) / baseline_test["mae"] * 100

    top_communities_actual = (
        community_error.sort_values("actual_mean_eui", ascending=False)
        .head(5)[["community_area_clean", "n", "actual_mean_eui", "mae", "bias_mean_pred_minus_actual"]]
    )
    top_types_actual = (
        building_error.sort_values("actual_mean_eui", ascending=False)
        .head(5)[["primary_property_type_clean", "n", "actual_mean_eui", "mae", "bias_mean_pred_minus_actual"]]
    )

    year_table = pd.DataFrame(
        {
            "year": list(summary["rows_by_year_after_filter"].keys()),
            "records": list(summary["rows_by_year_after_filter"].values()),
        }
    )

    report = f"""# Predicting Building Energy Use Intensity in Chicago via Spatio-Temporal Data Mining

## 1. 研究目標

本專案使用 Chicago Energy Benchmarking 資料，建立一個 regression model 來預測建築物年度 `Site EUI (kBtu/sq ft)`。分析單位是 **building-year record**，也就是一棟建築在某一年的能源表現。研究問題是：芝加哥不同 community area、不同 building type、不同年份的建築能源使用強度是否存在可學習的時空模式。

此任務屬於 spatio-temporal predictive analytics。空間面向來自 `community_area`、`zip_code`、`latitude`、`longitude`；時間面向來自 `data_year`、前一年 EUI、歷史 rolling/aggregate EUI，以及政策成熟階段。

資料來源：[{DATASET_PAGE}]({DATASET_PAGE})

## 2. 資料範圍與前處理

主實驗選取 **2018-2023**。2014-2017 是芝加哥 benchmarking 政策逐步導入期，樣本數較不穩定，因此沒有納入主模型訓練。2018-2023 原始下載筆數為 **{summary["raw_rows_2018_2023"]:,}**，完成篩選後剩下 **{summary["rows_after_outlier_filter"]:,}** 筆、**{summary["distinct_buildings_after_filter"]:,}** 棟建築。

資料篩選與清理規則：

1. 只保留 `data_year` 介於 2018-2023 的紀錄。
2. 只保留 `reporting_status` 為 `Submitted` 或 `Submitted Data` 的紀錄。
3. 移除 `Site EUI`、`gross_floor_area_buildings_sq_ft`、`primary_property_type`、`community_area`、`latitude`、`longitude` 缺失的紀錄。
4. 統一 `community_area` 大小寫，避免 `LOOP` 與 `Loop` 被視為不同區域。
5. Outlier 處理：只保留 `1 <= Site EUI <= 500`。此規則移除 **{summary["rows_removed_by_outlier_rule_1_to_500"]:,}** 筆極端值。
6. Leakage 處理：不把當年度能源消耗或能源表現結果類欄位放入模型，包括 `{", ".join(LEAKAGE_FIELDS_EXCLUDED)}`。這些欄位與目標 `Site EUI` 同時或事後產生，若納入模型會造成不合理的高估表現。

篩選後各年份筆數：

{markdown_table(year_table, ["year", "records"], float_digits=0)}

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

Model selected by validation RMSE: **{best_model_name}**

Best parameters:

```json
{json.dumps(best_params[best_model_name], indent=2, ensure_ascii=False)}
```

Random search 最佳結果摘要：

{markdown_table(tuning_results.groupby("model").head(3), ["model", "iteration", "validation_rmse", "validation_mae", "validation_r2", "params"], float_digits=4)}

## 5. 模型評估結果

Validation results:

{markdown_table(validation_metrics.sort_values("rmse"), ["model", "rmse", "mae", "r2"], float_digits=4)}

Test results:

{markdown_table(test_metrics.sort_values("rmse"), ["model", "rmse", "mae", "r2"], float_digits=4)}

依 validation RMSE 選出的模型 **{best_model_name}** 在 test set 相對於 Group Mean Baseline：

- RMSE 改善約 **{selected_rmse_improvement:.2f}%**
- MAE 改善約 **{selected_mae_improvement:.2f}%**

Test set 中 RMSE 最低的模型是 **{best_test_overall["model"]}**，相對於 Group Mean Baseline：

- RMSE 改善約 **{best_test_rmse_improvement:.2f}%**
- MAE 改善約 **{best_test_mae_improvement:.2f}%**

需要注意的是，模型選擇應以 validation set 為準，不能因為 test set 結果再回頭選模型或調參。因此本報告同時呈現 validation-selected model 與 test-best model，並把 test set 視為最終泛化能力檢查。

本次輸出的評估圖表：

- `outputs/plots/actual_vs_predicted_test.png`：實際值 vs 預測值。
- `outputs/plots/residual_analysis_test.png`：residual vs prediction 與 residual distribution。
- `outputs/plots/community_error_mae_top15.png`：test set 中 community area 的 MAE 比較。
- `outputs/plots/building_type_error_mae_top15.png`：test set 中 building type 的 MAE 比較。
- `outputs/plots/feature_importance_top20.png`：最佳 tree model 的 feature importance。

## 6. Community Area 誤差分析

以下表格列出 test set 中平均 EUI 較高的 community area。`bias_mean_pred_minus_actual` 若為負值，代表模型平均低估該區 EUI。

{markdown_table(top_communities_actual, ["community_area_clean", "n", "actual_mean_eui", "mae", "bias_mean_pred_minus_actual"], float_digits=3)}

MAE 最高的 community area：

{markdown_table(community_error.head(10), ["community_area_clean", "n", "actual_mean_eui", "predicted_mean_eui", "rmse", "mae", "bias_mean_pred_minus_actual"], float_digits=3)}

## 7. Building Type 誤差分析

以下表格列出 test set 中平均 EUI 較高的 building type。

{markdown_table(top_types_actual, ["primary_property_type_clean", "n", "actual_mean_eui", "mae", "bias_mean_pred_minus_actual"], float_digits=3)}

MAE 最高的 building type：

{markdown_table(building_error.head(10), ["primary_property_type_clean", "n", "actual_mean_eui", "predicted_mean_eui", "rmse", "mae", "bias_mean_pred_minus_actual"], float_digits=3)}

## 8. Feature Importance 分析

依 validation RMSE 選出的 tree model 的前 15 個重要特徵如下：

{markdown_table(feature_importance.head(15), ["feature", "importance"], float_digits=4)}

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

依 validation RMSE 選出的模型為 **{best_model_name}**，在 test set 的結果為 RMSE = **{selected_test["rmse"]:.4f}**、MAE = **{selected_test["mae"]:.4f}**、R² = **{selected_test["r2"]:.4f}**。另外，test set 上 RMSE 最低的是 **{best_test_overall["model"]}**，RMSE = **{best_test_overall["rmse"]:.4f}**、MAE = **{best_test_overall["mae"]:.4f}**、R² = **{best_test_overall["r2"]:.4f}**。此結果可以用來協助城市管理者初步識別高能源使用強度的建築類型與地區，並作為後續節能稽核或政策分區管理的資料基礎。
"""

    report_path.write_text(report, encoding="utf-8")


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

    write_report(
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
