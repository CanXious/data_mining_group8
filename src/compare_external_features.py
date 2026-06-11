from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import run_energy_benchmarking_analysis as base


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = ROOT / "reports"

MODELING_TABLE = PROCESSED_DIR / "chicago_energy_modeling_table_2018_2023.csv"

FOOTPRINTS_ENDPOINT = "https://data.cityofchicago.org/resource/syp8-uezg.json"
PERMITS_ENDPOINT = "https://data.cityofchicago.org/resource/ydr8-5enu.json"
OPEN_METEO_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"

RANDOM_STATE = base.RANDOM_STATE


DIR_MAP = {
    "NORTH": "N",
    "SOUTH": "S",
    "EAST": "E",
    "WEST": "W",
    "N": "N",
    "S": "S",
    "E": "E",
    "W": "W",
}

TYPE_MAP = {
    "STREET": "ST",
    "ST": "ST",
    "AVENUE": "AVE",
    "AV": "AVE",
    "AVE": "AVE",
    "ROAD": "RD",
    "RD": "RD",
    "BOULEVARD": "BLVD",
    "BLVD": "BLVD",
    "DRIVE": "DR",
    "DR": "DR",
    "PLACE": "PL",
    "PL": "PL",
    "COURT": "CT",
    "CT": "CT",
    "PARKWAY": "PKWY",
    "PKWY": "PKWY",
    "TERRACE": "TER",
    "TER": "TER",
    "LANE": "LN",
    "LN": "LN",
    "WAY": "WAY",
}


WEATHER_FEATURES = [
    "weather_temp_mean_c",
    "weather_temp_max_mean_c",
    "weather_temp_min_mean_c",
    "weather_precip_sum_mm",
    "weather_hdd65",
    "weather_cdd65",
    "weather_hot_days_30c",
    "weather_cold_days_minus10c",
]

FOOTPRINT_FEATURES = [
    "footprint_matched",
    "footprint_shape_area",
    "footprint_shape_len",
    "footprint_compactness",
    "footprint_stories",
    "footprint_year_built",
    "footprint_bldg_sqft",
    "gross_to_footprint_area_ratio",
    "footprint_age",
]

PERMIT_FEATURES = [
    "permit_matched",
    "permit_count_3yr",
    "permit_count_all_prior",
    "permit_reported_cost_sum_3yr",
    "permit_total_fee_sum_3yr",
    "permit_mechanical_count_3yr",
    "permit_electrical_count_3yr",
    "permit_renovation_count_3yr",
    "permit_major_cost_count_3yr",
    "years_since_last_permit",
    "has_recent_permit",
]


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, PLOT_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def socrata_download(endpoint: str, select: list[str], where: str, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path, low_memory=False)

    all_rows = []
    limit = 50_000
    offset = 0
    while True:
        params = {
            "$select": ",".join(select),
            "$where": where,
            "$limit": str(limit),
            "$offset": str(offset),
        }
        url = endpoint + "?" + urllib.parse.urlencode(params, safe="(),* ':-")
        with urllib.request.urlopen(url, timeout=120) as response:
            rows = json.load(response)
        all_rows.extend(rows)
        print(f"Downloaded {len(all_rows):,} rows from {endpoint.rsplit('/', 1)[-1]}")
        if len(rows) < limit:
            break
        offset += limit

    df = pd.DataFrame(all_rows)
    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return df


def clean_token(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).upper()
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_street_name_type(tokens: list[str]) -> tuple[str, str]:
    if not tokens:
        return "", ""
    street_type = ""
    if tokens[-1] in TYPE_MAP:
        street_type = TYPE_MAP[tokens[-1]]
        tokens = tokens[:-1]
    street_name = " ".join(tokens)
    return street_name, street_type


def normalize_energy_address(address: object) -> tuple[float, str, str, str, str]:
    text = clean_token(address)
    match = re.search(r"\b(\d+)\b", text)
    if not match:
        return math.nan, "", "", "", ""

    street_number = float(match.group(1))
    rest = text[match.end() :].strip()
    rest_tokens = rest.split()

    # Drop the second number in address ranges such as "411 415 S WELLS ST".
    if rest_tokens and rest_tokens[0].isdigit():
        rest_tokens = rest_tokens[1:]

    direction = ""
    if rest_tokens and rest_tokens[0] in DIR_MAP:
        direction = DIR_MAP[rest_tokens[0]]
        rest_tokens = rest_tokens[1:]

    street_name, street_type = parse_street_name_type(rest_tokens)
    full_key = f"{int(street_number)}|{direction}|{street_name}|{street_type}"
    street_key = f"{direction}|{street_name}|{street_type}"
    loose_key = f"{direction}|{street_name}"
    return street_number, street_key, loose_key, full_key, street_type


def normalize_structured_address(
    number: object,
    direction: object,
    street_name_raw: object,
    street_type_raw: object | None = None,
) -> tuple[float, str, str, str, str]:
    try:
        street_number = float(number)
    except (TypeError, ValueError):
        return math.nan, "", "", "", ""

    direction_clean = DIR_MAP.get(clean_token(direction), "")
    tokens = clean_token(street_name_raw).split()
    if street_type_raw is not None and clean_token(street_type_raw):
        street_name = " ".join(tokens)
        street_type = TYPE_MAP.get(clean_token(street_type_raw), clean_token(street_type_raw))
    else:
        street_name, street_type = parse_street_name_type(tokens)
    full_key = f"{int(street_number)}|{direction_clean}|{street_name}|{street_type}"
    street_key = f"{direction_clean}|{street_name}|{street_type}"
    loose_key = f"{direction_clean}|{street_name}"
    return street_number, street_key, loose_key, full_key, street_type


def add_address_keys(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["address"].map(normalize_energy_address)
    keys = pd.DataFrame(
        parsed.tolist(),
        columns=["address_number", "street_key", "loose_street_key", "address_key", "street_type"],
        index=df.index,
    )
    return pd.concat([df, keys], axis=1)


def download_weather_features() -> pd.DataFrame:
    cache_path = RAW_DIR / "open_meteo_chicago_daily_2018_2023.csv"
    if cache_path.exists():
        daily = pd.read_csv(cache_path)
    else:
        params = {
            "latitude": "41.8781",
            "longitude": "-87.6298",
            "start_date": "2018-01-01",
            "end_date": "2023-12-31",
            "daily": ",".join(
                [
                    "temperature_2m_mean",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                ]
            ),
            "timezone": "America/Chicago",
        }
        url = OPEN_METEO_ENDPOINT + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = json.load(response)["daily"]
        daily = pd.DataFrame(data)
        daily.to_csv(cache_path, index=False, encoding="utf-8-sig")

    daily["time"] = pd.to_datetime(daily["time"])
    daily["data_year"] = daily["time"].dt.year
    base_temp_c = (65 - 32) * 5 / 9
    daily["hdd65"] = (base_temp_c - daily["temperature_2m_mean"]).clip(lower=0)
    daily["cdd65"] = (daily["temperature_2m_mean"] - base_temp_c).clip(lower=0)
    daily["hot_days_30c"] = (daily["temperature_2m_max"] >= 30).astype(int)
    daily["cold_days_minus10c"] = (daily["temperature_2m_min"] <= -10).astype(int)

    annual = (
        daily.groupby("data_year", as_index=False)
        .agg(
            weather_temp_mean_c=("temperature_2m_mean", "mean"),
            weather_temp_max_mean_c=("temperature_2m_max", "mean"),
            weather_temp_min_mean_c=("temperature_2m_min", "mean"),
            weather_precip_sum_mm=("precipitation_sum", "sum"),
            weather_hdd65=("hdd65", "sum"),
            weather_cdd65=("cdd65", "sum"),
            weather_hot_days_30c=("hot_days_30c", "sum"),
            weather_cold_days_minus10c=("cold_days_minus10c", "sum"),
        )
        .sort_values("data_year")
    )
    annual.to_csv(PROCESSED_DIR / "external_weather_features_2018_2023.csv", index=False, encoding="utf-8-sig")
    return annual


def add_weather_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    weather = download_weather_features()
    out = df.merge(weather, on="data_year", how="left")
    return out, {
        "feature_set": "weather",
        "matched_rows": int(out[WEATHER_FEATURES[0]].notna().sum()),
        "matched_row_rate": float(out[WEATHER_FEATURES[0]].notna().mean()),
    }


def download_footprints() -> pd.DataFrame:
    select = [
        "bldg_id",
        "f_add1",
        "t_add1",
        "pre_dir1",
        "st_name1",
        "st_type1",
        "stories",
        "no_stories",
        "year_built",
        "bldg_sq_fo",
        "shape_area",
        "shape_len",
    ]
    where = "f_add1 > 0 and st_name1 is not null"
    return socrata_download(
        FOOTPRINTS_ENDPOINT,
        select,
        where,
        RAW_DIR / "chicago_building_footprints_selected.csv",
    )


def add_footprint_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    footprints = download_footprints()
    for col in ["f_add1", "t_add1", "stories", "no_stories", "year_built", "bldg_sq_fo", "shape_area", "shape_len"]:
        footprints[col] = pd.to_numeric(footprints[col], errors="coerce")

    parsed = footprints.apply(
        lambda row: normalize_structured_address(
            row["f_add1"],
            row.get("pre_dir1"),
            row.get("st_name1"),
            row.get("st_type1"),
        ),
        axis=1,
    )
    key_df = pd.DataFrame(
        parsed.tolist(),
        columns=["fp_address_number", "fp_street_key", "fp_loose_street_key", "fp_address_key", "fp_street_type"],
        index=footprints.index,
    )
    footprints = pd.concat([footprints, key_df], axis=1)
    footprints = footprints.dropna(subset=["f_add1", "t_add1", "shape_area", "shape_len"]).copy()
    footprints["t_add1"] = footprints["t_add1"].fillna(footprints["f_add1"])
    footprints["range_width"] = (footprints["t_add1"] - footprints["f_add1"]).abs()

    by_street: dict[str, pd.DataFrame] = {
        key: group.sort_values(["range_width", "shape_area"], ascending=[True, False])
        for key, group in footprints.groupby("fp_street_key")
    }
    by_loose: dict[str, pd.DataFrame] = {
        key: group.sort_values(["range_width", "shape_area"], ascending=[True, False])
        for key, group in footprints.groupby("fp_loose_street_key")
    }

    def match_one(row: pd.Series) -> dict:
        number = row["address_number"]
        candidates = by_street.get(row["street_key"])
        if candidates is None or candidates.empty:
            candidates = by_loose.get(row["loose_street_key"])
        if candidates is None or candidates.empty or pd.isna(number):
            return {"address_key": row["address_key"], "footprint_matched": 0}

        matched = candidates[(candidates["f_add1"] <= number) & (candidates["t_add1"] >= number)]
        if matched.empty:
            matched = candidates[candidates["f_add1"].eq(number)]
        if matched.empty:
            return {"address_key": row["address_key"], "footprint_matched": 0}

        item = matched.iloc[0]
        stories = item["stories"] if item["stories"] and item["stories"] > 0 else item["no_stories"]
        shape_area = item["shape_area"]
        shape_len = item["shape_len"]
        compactness = 4 * math.pi * shape_area / (shape_len**2) if shape_len and shape_len > 0 else np.nan
        ratio = row[base.GROSS_FLOOR] / shape_area if shape_area and shape_area > 0 else np.nan
        return {
            "address_key": row["address_key"],
            "footprint_matched": 1,
            "footprint_shape_area": shape_area,
            "footprint_shape_len": shape_len,
            "footprint_compactness": compactness,
            "footprint_stories": stories if stories and stories > 0 else np.nan,
            "footprint_year_built": item["year_built"] if item["year_built"] and item["year_built"] > 0 else np.nan,
            "footprint_bldg_sqft": item["bldg_sq_fo"] if item["bldg_sq_fo"] and item["bldg_sq_fo"] > 0 else np.nan,
            "gross_to_footprint_area_ratio": ratio,
        }

    unique_addresses = df.drop_duplicates("address_key").copy()
    matched_features = pd.DataFrame([match_one(row) for _, row in unique_addresses.iterrows()])
    out = df.merge(matched_features, on="address_key", how="left")
    out["footprint_matched"] = out["footprint_matched"].fillna(0)
    out["footprint_age"] = out["data_year"] - out["footprint_year_built"]
    out.loc[(out["footprint_age"] < 0) | (out["footprint_age"] > 250), "footprint_age"] = np.nan
    for col in FOOTPRINT_FEATURES:
        if col not in out.columns:
            out[col] = np.nan

    return out, {
        "feature_set": "footprints",
        "matched_rows": int(out["footprint_matched"].sum()),
        "matched_row_rate": float(out["footprint_matched"].mean()),
        "matched_unique_addresses": int(matched_features["footprint_matched"].sum()),
        "unique_addresses": int(len(matched_features)),
    }


def download_permits() -> pd.DataFrame:
    select = [
        "id",
        "permit_",
        "permit_status",
        "permit_type",
        "issue_date",
        "street_number",
        "street_direction",
        "street_name",
        "work_type",
        "work_description",
        "reported_cost",
        "total_fee",
        "latitude",
        "longitude",
    ]
    where = "issue_date between '2015-01-01T00:00:00' and '2022-12-31T23:59:59'"
    return socrata_download(
        PERMITS_ENDPOINT,
        select,
        where,
        RAW_DIR / "chicago_building_permits_2015_2022_selected.csv",
    )


def add_permit_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    permits = download_permits()
    permits["issue_date"] = pd.to_datetime(permits["issue_date"], errors="coerce")
    permits["issue_year"] = permits["issue_date"].dt.year
    for col in ["reported_cost", "total_fee"]:
        permits[col] = pd.to_numeric(permits[col], errors="coerce").fillna(0)

    parsed = permits.apply(
        lambda row: normalize_structured_address(
            row.get("street_number"),
            row.get("street_direction"),
            row.get("street_name"),
            None,
        ),
        axis=1,
    )
    key_df = pd.DataFrame(
        parsed.tolist(),
        columns=["permit_address_number", "permit_street_key", "permit_loose_street_key", "address_key", "permit_street_type"],
        index=permits.index,
    )
    permits = pd.concat([permits, key_df], axis=1)
    permits = permits[permits["address_key"].ne("nan|||") & permits["issue_year"].notna()].copy()

    text = (
        permits["work_type"].fillna("").astype(str).str.upper()
        + " "
        + permits["work_description"].fillna("").astype(str).str.upper()
        + " "
        + permits["permit_type"].fillna("").astype(str).str.upper()
    )
    permits["mechanical_flag"] = text.str.contains(
        r"MECHANICAL|HVAC|VENTILATION|REFRIGERATION|FURNACE|BOILER|AIR CONDITION|COOLING|HEATING",
        regex=True,
    ).astype(int)
    permits["electrical_flag"] = text.str.contains(r"ELECTRIC|WIRING|POWER|LIGHTING", regex=True).astype(int)
    permits["renovation_flag"] = text.str.contains(r"RENOVATION|ALTERATION|REHAB|REMODEL|REPAIR", regex=True).astype(int)
    permits["major_cost_flag"] = (permits["reported_cost"] >= 100_000).astype(int)

    annual = (
        permits.groupby(["address_key", "issue_year"], as_index=False)
        .agg(
            permit_count=("id", "count"),
            reported_cost_sum=("reported_cost", "sum"),
            total_fee_sum=("total_fee", "sum"),
            mechanical_count=("mechanical_flag", "sum"),
            electrical_count=("electrical_flag", "sum"),
            renovation_count=("renovation_flag", "sum"),
            major_cost_count=("major_cost_flag", "sum"),
        )
        .sort_values(["address_key", "issue_year"])
    )

    feature_frames = []
    for year in range(2018, 2024):
        recent = annual[(annual["issue_year"] >= year - 3) & (annual["issue_year"] <= year - 1)]
        recent_grouped = (
            recent.groupby("address_key", as_index=False)
            .agg(
                permit_count_3yr=("permit_count", "sum"),
                permit_reported_cost_sum_3yr=("reported_cost_sum", "sum"),
                permit_total_fee_sum_3yr=("total_fee_sum", "sum"),
                permit_mechanical_count_3yr=("mechanical_count", "sum"),
                permit_electrical_count_3yr=("electrical_count", "sum"),
                permit_renovation_count_3yr=("renovation_count", "sum"),
                permit_major_cost_count_3yr=("major_cost_count", "sum"),
            )
            .assign(data_year=year)
        )

        prior = annual[annual["issue_year"] <= year - 1]
        prior_count = (
            prior.groupby("address_key", as_index=False)
            .agg(
                permit_count_all_prior=("permit_count", "sum"),
                last_permit_year=("issue_year", "max"),
            )
            .assign(data_year=year)
        )
        features = recent_grouped.merge(prior_count, on=["address_key", "data_year"], how="outer")
        feature_frames.append(features)

    permit_features = pd.concat(feature_frames, ignore_index=True)
    out = df.merge(permit_features, on=["address_key", "data_year"], how="left")
    count_cols = [
        "permit_count_3yr",
        "permit_count_all_prior",
        "permit_reported_cost_sum_3yr",
        "permit_total_fee_sum_3yr",
        "permit_mechanical_count_3yr",
        "permit_electrical_count_3yr",
        "permit_renovation_count_3yr",
        "permit_major_cost_count_3yr",
    ]
    for col in count_cols:
        out[col] = out[col].fillna(0)
    out["years_since_last_permit"] = out["data_year"] - out["last_permit_year"]
    out["has_recent_permit"] = (out["permit_count_3yr"] > 0).astype(int)
    out["permit_matched"] = (out["permit_count_all_prior"] > 0).astype(int)
    for col in PERMIT_FEATURES:
        if col not in out.columns:
            out[col] = np.nan

    return out, {
        "feature_set": "permits",
        "matched_rows": int(out["permit_matched"].sum()),
        "matched_row_rate": float(out["permit_matched"].mean()),
        "recent_permit_rows": int(out["has_recent_permit"].sum()),
        "recent_permit_row_rate": float(out["has_recent_permit"].mean()),
    }


def set_feature_globals(extra_numeric: list[str]) -> tuple[list[str], list[str], list[str]]:
    original = (base.NUMERIC_FEATURES.copy(), base.CATEGORICAL_FEATURES.copy(), base.FEATURES.copy())
    base.NUMERIC_FEATURES = list(dict.fromkeys(base.NUMERIC_FEATURES + extra_numeric))
    base.CATEGORICAL_FEATURES = base.CATEGORICAL_FEATURES.copy()
    base.FEATURES = base.NUMERIC_FEATURES + base.CATEGORICAL_FEATURES
    return original


def restore_feature_globals(original: tuple[list[str], list[str], list[str]]) -> None:
    base.NUMERIC_FEATURES, base.CATEGORICAL_FEATURES, base.FEATURES = original


def evaluate_variant(name: str, df: pd.DataFrame, extra_numeric: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = set_feature_globals(extra_numeric)
    try:
        train_df = df[df["data_year"].between(2018, 2021)].copy()
        val_df = df[df["data_year"].eq(2022)].copy()
        test_df = df[df["data_year"].eq(2023)].copy()
        trainval_df = df[df["data_year"].between(2018, 2022)].copy()

        validation_rows = []
        test_rows = []

        group_keys = ["community_area_clean", "primary_property_type_clean"]
        val_group_pred = base.fallback_group_mean_predict(train_df, val_df, group_keys)
        test_group_pred = base.fallback_group_mean_predict(trainval_df, test_df, group_keys)
        validation_rows.append(
            {"feature_set": name, **base.metrics_dict("validation", "Group Mean Baseline", val_df[base.TARGET], val_group_pred)}
        )
        test_rows.append(
            {"feature_set": name, **base.metrics_dict("test", "Group Mean Baseline", test_df[base.TARGET], test_group_pred)}
        )

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
        rf_factory = lambda **params: RandomForestRegressor(  # noqa: E731
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )
        xgb_factory = lambda **params: XGBRegressor(  # noqa: E731
            objective="reg:squarederror",
            tree_method="hist",
            eval_metric="rmse",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            **params,
        )

        print(f"Evaluating {name}: Random Forest random search")
        rf_params, rf_tuning = base.tune_random_search(
            "Random Forest",
            rf_factory,
            rf_param_dist,
            train_df,
            val_df,
            n_iter=20,
        )
        print(f"Evaluating {name}: XGBoost random search")
        xgb_params, xgb_tuning = base.tune_random_search(
            "XGBoost",
            xgb_factory,
            xgb_param_dist,
            train_df,
            val_df,
            n_iter=25,
        )
        tuning = pd.concat([rf_tuning, xgb_tuning], ignore_index=True)
        tuning["feature_set"] = name

        for model_name, params in {"Random Forest": rf_params, "XGBoost": xgb_params}.items():
            val_model = base.fit_pipeline(model_name, params, train_df)
            val_pred = val_model.predict(val_df[base.FEATURES])
            validation_rows.append(
                {"feature_set": name, **base.metrics_dict("validation", model_name, val_df[base.TARGET], val_pred)}
            )

            final_model = base.fit_pipeline(model_name, params, trainval_df)
            test_pred = final_model.predict(test_df[base.FEATURES])
            test_rows.append(
                {"feature_set": name, **base.metrics_dict("test", model_name, test_df[base.TARGET], test_pred)}
            )

        metrics = pd.concat([pd.DataFrame(validation_rows), pd.DataFrame(test_rows)], ignore_index=True)
        return metrics, tuning
    finally:
        restore_feature_globals(original)


def plot_comparison(test_metrics: pd.DataFrame) -> None:
    model_order = ["Group Mean Baseline", "Random Forest", "XGBoost"]
    feature_order = ["base", "weather", "footprints", "permits"]
    pivot = (
        test_metrics[test_metrics["model"].isin(model_order)]
        .pivot(index="feature_set", columns="model", values="rmse")
        .reindex(feature_order)
    )
    ax = pivot.plot(kind="bar", figsize=(10, 6), color=["#7F7F7F", "#2F6F73", "#7A5195"])
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Test RMSE")
    ax.set_title("External Feature Ablation: Test RMSE")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "external_feature_test_rmse_comparison.png", dpi=180)
    plt.close()


def plot_comparison_with_all(test_metrics: pd.DataFrame) -> None:
    model_order = ["Group Mean Baseline", "Random Forest", "XGBoost"]
    feature_order = ["base", "weather", "footprints", "permits", "all_external"]
    pivot = (
        test_metrics[test_metrics["model"].isin(model_order)]
        .pivot(index="feature_set", columns="model", values="rmse")
        .reindex(feature_order)
    )
    ax = pivot.plot(kind="bar", figsize=(11, 6), color=["#7F7F7F", "#2F6F73", "#7A5195"])
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Test RMSE")
    ax.set_title("External Feature Ablation: Test RMSE Including All External Features")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "external_feature_test_rmse_comparison_with_all.png", dpi=180)
    plt.close()


def write_report(
    metrics: pd.DataFrame,
    tuning: pd.DataFrame,
    join_summary: pd.DataFrame,
) -> None:
    test = metrics[metrics["split"].eq("test")].copy()
    best_by_set = test.sort_values(["feature_set", "rmse"]).groupby("feature_set", as_index=False).first()
    base_best_rmse = best_by_set.loc[best_by_set["feature_set"].eq("base"), "rmse"].iloc[0]
    best_by_set["rmse_change_vs_base_best"] = best_by_set["rmse"] - base_best_rmse
    best_by_set["rmse_change_pct_vs_base_best"] = best_by_set["rmse_change_vs_base_best"] / base_best_rmse * 100

    def md_table(df: pd.DataFrame, cols: list[str], digits: int = 4) -> str:
        out = df[cols].copy()
        for col in out.select_dtypes(include=[np.number]).columns:
            out[col] = out[col].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
        rows = [cols] + out.astype(str).values.tolist()
        widths = [max(len(row[i]) for row in rows) for i in range(len(cols))]
        lines = []
        lines.append("| " + " | ".join(rows[0][i].ljust(widths[i]) for i in range(len(cols))) + " |")
        lines.append("| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |")
        for row in rows[1:]:
            lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) + " |")
        return "\n".join(lines)

    report = f"""# External Feature Ablation Comparison

本補充實驗分別加入三種外部資料，觀察是否能提升 Chicago Energy Benchmarking 的 Site EUI regression model。

比較方式：

1. `base`：原始報告中的資料前處理與 feature engineering。
2. `weather`：加入年度天氣特徵，包括 HDD/CDD、平均溫度、降水、極端高溫/低溫天數。
3. `footprints`：加入 Chicago Building Footprints 的建築幾何與樓層特徵。
4. `permits`：加入 Building Permits 的前三年 permit history。

所有版本都使用同一個切分：

- Train：2018-2021
- Validation：2022，用於 random search hyperparameter tuning
- Test：2023，只做最後評估

## Join Coverage

{md_table(join_summary, list(join_summary.columns), digits=4)}

## Test Metrics

{md_table(test.sort_values(["feature_set", "rmse"]), ["feature_set", "model", "rmse", "mae", "r2"], digits=4)}

## Best Model By Feature Set

{md_table(best_by_set, ["feature_set", "model", "rmse", "mae", "r2", "rmse_change_vs_base_best", "rmse_change_pct_vs_base_best"], digits=4)}

## 圖表

![External Feature RMSE Comparison](../outputs/plots/external_feature_test_rmse_comparison.png)

## Interpretation

- 若 `rmse_change_vs_base_best` 為負值，代表該外部資料相對於 base 最佳模型有提升。
- 若為正值，代表加入該外部資料後 test RMSE 沒有改善，可能是資料 join coverage 不足、外部資料與 EUI 關係較弱，或新增特徵在 validation/test 年份間不穩定。
- Weather features 是年度層級，所有建築同一年拿到同一組天氣值，因此主要補充時間變異，不能補足建築間差異。
- Footprint features 屬於建築形狀與規模資訊，若地址 match coverage 高，理論上最有機會補強建築物理特性。
- Permit features 使用 reporting year 前三年的 permit history，避免使用未來資訊；若沒有提升，可能代表 permit 類型太粗、地址匹配不足，或 permit 對 EUI 的影響有時間延遲。
"""
    (REPORT_DIR / "external_feature_comparison.md").write_text(report, encoding="utf-8")


def write_report_with_all(
    metrics: pd.DataFrame,
    tuning: pd.DataFrame,
    join_summary: pd.DataFrame,
) -> None:
    test = metrics[metrics["split"].eq("test")].copy()
    best_by_set = test.sort_values(["feature_set", "rmse"]).groupby("feature_set", as_index=False).first()
    base_best_rmse = best_by_set.loc[best_by_set["feature_set"].eq("base"), "rmse"].iloc[0]
    weather_best_rmse = best_by_set.loc[best_by_set["feature_set"].eq("weather"), "rmse"].iloc[0]
    best_by_set["rmse_change_vs_base_best"] = best_by_set["rmse"] - base_best_rmse
    best_by_set["rmse_change_pct_vs_base_best"] = best_by_set["rmse_change_vs_base_best"] / base_best_rmse * 100
    best_by_set["rmse_change_vs_weather_best"] = best_by_set["rmse"] - weather_best_rmse
    best_by_set["rmse_change_pct_vs_weather_best"] = (
        best_by_set["rmse_change_vs_weather_best"] / weather_best_rmse * 100
    )

    def md_table(df: pd.DataFrame, cols: list[str], digits: int = 4) -> str:
        out = df[cols].copy()
        for col in out.select_dtypes(include=[np.number]).columns:
            out[col] = out[col].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
        rows = [cols] + out.fillna("").astype(str).values.tolist()
        widths = [max(len(row[i]) for row in rows) for i in range(len(cols))]
        lines = []
        lines.append("| " + " | ".join(rows[0][i].ljust(widths[i]) for i in range(len(cols))) + " |")
        lines.append("| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |")
        for row in rows[1:]:
            lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) + " |")
        return "\n".join(lines)

    all_best = best_by_set.loc[best_by_set["feature_set"].eq("all_external")].iloc[0]
    weather_best = best_by_set.loc[best_by_set["feature_set"].eq("weather")].iloc[0]

    report = f"""# External Feature Ablation Comparison With All External Features

本補充實驗在原本三個單獨外部資料比較之外，額外加入一組 `all_external`：

> Base + Weather + Building Footprints + Building Permits

所有版本都使用同一個切分與 tuning 設定：

- Train：2018-2021
- Validation：2022，用於 random search hyperparameter tuning
- Test：2023，只做最後評估
- Random Forest：20 組 random search
- XGBoost：25 組 random search

## Join Coverage

{md_table(join_summary, list(join_summary.columns), digits=4)}

## Test Metrics

{md_table(test.sort_values(["feature_set", "rmse"]), ["feature_set", "model", "rmse", "mae", "r2"], digits=4)}

## Best Model By Feature Set

{md_table(best_by_set, ["feature_set", "model", "rmse", "mae", "r2", "rmse_change_vs_base_best", "rmse_change_pct_vs_base_best", "rmse_change_vs_weather_best", "rmse_change_pct_vs_weather_best"], digits=4)}

## 圖表

![External Feature RMSE Comparison With All](../outputs/plots/external_feature_test_rmse_comparison_with_all.png)

## Interpretation

Weather-only 的最佳模型是 **{weather_best["model"]}**，test RMSE = **{weather_best["rmse"]:.4f}**。

All external 的最佳模型是 **{all_best["model"]}**，test RMSE = **{all_best["rmse"]:.4f}**。

相對於 weather-only，all external 的 RMSE 變化為 **{all_best["rmse_change_vs_weather_best"]:.4f}**，百分比為 **{all_best["rmse_change_pct_vs_weather_best"]:.2f}%**。

若 all external 沒有優於 weather-only，代表 footprints 與 permits 在目前的 address matching、資料粒度與特徵設計下，沒有在 weather features 已存在時提供穩定額外訊息。若 all external 優於 weather-only，則表示建築幾何或 permit history 對 weather-adjusted model 還有額外補充價值。
"""
    (REPORT_DIR / "external_feature_comparison_with_all.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(MODELING_TABLE, low_memory=False)
    df = add_address_keys(df)

    feature_sets: dict[str, tuple[pd.DataFrame, list[str], dict]] = {
        "base": (df.copy(), [], {"feature_set": "base", "matched_rows": len(df), "matched_row_rate": 1.0}),
    }

    weather_df, weather_summary = add_weather_features(df.copy())
    feature_sets["weather"] = (weather_df, WEATHER_FEATURES, weather_summary)

    footprint_df, footprint_summary = add_footprint_features(df.copy())
    feature_sets["footprints"] = (footprint_df, FOOTPRINT_FEATURES, footprint_summary)

    permit_df, permit_summary = add_permit_features(df.copy())
    feature_sets["permits"] = (permit_df, PERMIT_FEATURES, permit_summary)

    join_summary = pd.DataFrame([feature_sets[name][2] for name in feature_sets])
    join_summary.to_csv(OUTPUT_DIR / "external_feature_join_summary.csv", index=False, encoding="utf-8-sig")

    all_metrics = []
    all_tuning = []
    for name, (variant_df, extra_numeric, _) in feature_sets.items():
        metrics, tuning = evaluate_variant(name, variant_df, extra_numeric)
        all_metrics.append(metrics)
        all_tuning.append(tuning)
        pd.concat(all_metrics, ignore_index=True).to_csv(
            OUTPUT_DIR / "external_feature_comparison_metrics_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    tuning_df = pd.concat(all_tuning, ignore_index=True)
    metrics_df.to_csv(OUTPUT_DIR / "external_feature_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    tuning_df.to_csv(OUTPUT_DIR / "external_feature_tuning_results.csv", index=False, encoding="utf-8-sig")
    plot_comparison(metrics_df[metrics_df["split"].eq("test")])
    write_report(metrics_df, tuning_df, join_summary)

    print("Done external feature comparison.")
    print(metrics_df[metrics_df["split"].eq("test")].sort_values(["feature_set", "rmse"]).to_string(index=False))


if __name__ == "__main__":
    main()
