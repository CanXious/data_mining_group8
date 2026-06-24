from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import compare_external_features as ext


SOCIO_ENDPOINT = "https://data.cityofchicago.org/resource/kn9c-c2s2.json"

SOCIO_OCCUPANCY_FEATURES = [
    "socio_occupancy_context_matched",
    "socio_housing_crowded_pct",
    "socio_poverty_pct",
    "socio_unemployment_pct",
    "socio_no_high_school_diploma_pct",
    "socio_dependency_pct",
    "socio_per_capita_income",
    "socio_hardship_index",
]


def clean_area_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).upper().strip()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    aliases = {
        "OHARE": "O HARE",
        "LAKEVIEW": "LAKE VIEW",
        "MONTCLARE": "MONTCLAIRE",
        "WASHINGTON HEIGHTS": "WASHINGTON HEIGHT",
    }
    return aliases.get(text, text)


def fetch_json(url: str) -> list[dict]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 data-mining-course-project"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def census_direct_available() -> tuple[bool, str]:
    if not os.environ.get("CENSUS_API_KEY"):
        return False, "CENSUS_API_KEY is not set; Census API currently requires a key."

    params = {
        "get": "NAME,DP03_0009PE,DP03_0062E,DP03_0088E,DP03_0128PE,DP02_0001E,DP04_0001E,DP04_0002E",
        "for": "tract:*",
        "in": "state:17 county:031",
        "key": os.environ["CENSUS_API_KEY"],
    }
    url = "https://api.census.gov/data/2023/acs/acs5/profile?" + urllib.parse.urlencode(params)
    try:
        rows = fetch_json(url)
    except Exception as exc:  # noqa: BLE001
        return False, f"Census ACS API request failed: {type(exc).__name__}: {exc}"
    if len(rows) <= 1:
        return False, "Census ACS API returned no tract rows."
    return True, "Census ACS API is reachable, but tract-level spatial joining is not available in the current modeling table."


def download_selected_socioeconomic() -> pd.DataFrame:
    cache_path = ext.RAW_DIR / "chicago_selected_socioeconomic_indicators_2008_2012.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path, low_memory=False)

    params = {"$limit": "5000"}
    url = SOCIO_ENDPOINT + "?" + urllib.parse.urlencode(params)
    rows = fetch_json(url)
    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return df


def build_socioeconomic_occupancy_features() -> tuple[pd.DataFrame, dict]:
    census_ok, census_note = census_direct_available()

    raw = download_selected_socioeconomic()
    df = raw.copy()
    df["community_area_key"] = df["community_area_name"].map(clean_area_key)
    df = df[df["community_area_key"].ne("CHICAGO")].copy()

    rename = {
        "percent_of_housing_crowded": "socio_housing_crowded_pct",
        "percent_households_below_poverty": "socio_poverty_pct",
        "percent_aged_16_unemployed": "socio_unemployment_pct",
        "percent_aged_25_without_high_school_diploma": "socio_no_high_school_diploma_pct",
        "percent_aged_under_18_or_over_64": "socio_dependency_pct",
        "per_capita_income_": "socio_per_capita_income",
        "hardship_index": "socio_hardship_index",
    }
    df = df.rename(columns=rename)
    for col in rename.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = ["community_area_key"] + list(rename.values())
    features = df[keep_cols].drop_duplicates("community_area_key").copy()
    features.to_csv(
        ext.PROCESSED_DIR / "external_socioeconomic_occupancy_features_community_area.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "feature_set": "socio_occupancy",
        "requested_source": "U.S. Census ACS 5-Year",
        "actual_source": "Chicago Selected Socioeconomic Indicators 2008-2012",
        "fallback_used": True,
        "fallback_reason": census_note if not census_ok else "Direct ACS tract join not available in current modeling table.",
        "source_rows": int(len(raw)),
        "community_area_rows": int(len(features)),
    }
    return features, summary


def add_socioeconomic_occupancy_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    socio, summary = build_socioeconomic_occupancy_features()
    out = df.copy()
    out["community_area_key"] = out["community_area_clean"].map(clean_area_key)
    out = out.merge(socio, on="community_area_key", how="left")
    out["socio_occupancy_context_matched"] = out["socio_housing_crowded_pct"].notna().astype(int)

    for col in SOCIO_OCCUPANCY_FEATURES:
        if col not in out.columns:
            out[col] = np.nan

    summary = {
        **summary,
        "matched_rows": int(out["socio_occupancy_context_matched"].sum()),
        "matched_row_rate": float(out["socio_occupancy_context_matched"].mean()),
        "matched_community_areas": int(out.loc[out["socio_occupancy_context_matched"].eq(1), "community_area_key"].nunique()),
        "model_rows": int(len(out)),
    }
    return out, summary


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


def plot_socio_comparison(test_metrics: pd.DataFrame) -> None:
    model_order = ["Group Mean Baseline", "Random Forest", "XGBoost"]
    feature_order = [
        "base",
        "weather",
        "socio_occupancy",
        "weather_socio_occupancy",
        "all_external",
        "all_external_plus_socio_occupancy",
    ]
    pivot = (
        test_metrics[test_metrics["model"].isin(model_order)]
        .pivot(index="feature_set", columns="model", values="rmse")
        .reindex(feature_order)
    )
    ax = pivot.plot(kind="bar", figsize=(12, 6), color=["#7F7F7F", "#2F6F73", "#7A5195"])
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Socioeconomic and Occupancy Feature Comparison")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(ext.PLOT_DIR / "socioeconomic_occupancy_test_rmse_comparison.png", dpi=180)
    plt.close()


def write_report(
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
    best_by_set["rmse_change_pct_vs_weather_best"] = best_by_set["rmse_change_vs_weather_best"] / weather_best_rmse * 100

    feature_formula = pd.DataFrame(
        [
            ["socio_housing_crowded_pct", "Percent occupied housing units with more than one person per room"],
            ["socio_poverty_pct", "Percent households below the federal poverty level"],
            ["socio_unemployment_pct", "Percent labor force aged 16+ unemployed"],
            ["socio_no_high_school_diploma_pct", "Percent population aged 25+ without high school diploma"],
            ["socio_dependency_pct", "Percent population under 18 or over 64"],
            ["socio_per_capita_income", "Community-area per capita income"],
            ["socio_hardship_index", "Composite hardship index from six selected socioeconomic indicators"],
            ["socio_occupancy_context_matched", "1 if community-area socioeconomic record matched, else 0"],
        ],
        columns=["feature", "definition"],
    )

    report = f"""# Socioeconomic and Occupancy Feature Comparison

This experiment tested whether community-area socioeconomic and occupancy-context features improve the Chicago Site EUI regression model.

The requested first choice was U.S. Census ACS 5-Year. The direct Census API source was not used in this run because no `CENSUS_API_KEY` was available and tract-level ACS joining would require a tract GEOID or point-to-tract spatial join that is not present in the current modeling table. Following the requested fallback rule, this run used the City of Chicago Selected Socioeconomic Indicators dataset.

## Added Features

{md_table(feature_formula, ["feature", "definition"], digits=4)}

## Data Source and Join Coverage

{md_table(join_summary, list(join_summary.columns), digits=4)}

## Test Metrics

{md_table(test.sort_values(["feature_set", "rmse"]), ["feature_set", "model", "rmse", "mae", "r2"], digits=4)}

## Best Model by Feature Set

{md_table(best_by_set, ["feature_set", "model", "rmse", "mae", "r2", "rmse_change_vs_base_best", "rmse_change_pct_vs_base_best", "rmse_change_vs_weather_best", "rmse_change_pct_vs_weather_best"], digits=4)}

## Plot

![Socioeconomic Occupancy RMSE Comparison](../outputs/plots/socioeconomic_occupancy_test_rmse_comparison.png)

## Interpretation

- `socio_occupancy` tests socioeconomic and occupancy-context features alone against the base model.
- `weather_socio_occupancy` tests whether socioeconomic context adds value on top of the current best-performing weather feature set.
- `all_external_plus_socio_occupancy` tests whether socioeconomic context improves the full external-feature model.
- The socioeconomic fallback data is community-area level and time-static for 2008-2012, so it should be interpreted as long-term neighborhood context rather than year-specific 2018-2023 socioeconomic conditions.
"""
    (ext.REPORT_DIR / "socioeconomic_occupancy_feature_comparison.md").write_text(report, encoding="utf-8")


def main() -> None:
    ext.ensure_dirs()
    base_df = pd.read_csv(ext.MODELING_TABLE, low_memory=False)
    base_df = ext.add_address_keys(base_df)

    socio_df, socio_summary = add_socioeconomic_occupancy_features(base_df.copy())
    weather_df, weather_summary = ext.add_weather_features(base_df.copy())
    weather_socio_df, weather_socio_summary = add_socioeconomic_occupancy_features(weather_df)

    footprint_df, footprint_summary = ext.add_footprint_features(weather_df)
    all_external_df, permit_summary = ext.add_permit_features(footprint_df)
    all_plus_socio_df, all_plus_socio_summary = add_socioeconomic_occupancy_features(all_external_df)

    variants = [
        ("socio_occupancy", socio_df, SOCIO_OCCUPANCY_FEATURES),
        ("weather_socio_occupancy", weather_socio_df, ext.WEATHER_FEATURES + SOCIO_OCCUPANCY_FEATURES),
        (
            "all_external_plus_socio_occupancy",
            all_plus_socio_df,
            ext.WEATHER_FEATURES + ext.FOOTPRINT_FEATURES + ext.PERMIT_FEATURES + SOCIO_OCCUPANCY_FEATURES,
        ),
    ]

    metrics_frames = []
    tuning_frames = []
    for name, df, features in variants:
        metrics, tuning = ext.evaluate_variant(name, df, features)
        metrics_frames.append(metrics)
        tuning_frames.append(tuning)

    new_metrics = pd.concat(metrics_frames, ignore_index=True)
    new_tuning = pd.concat(tuning_frames, ignore_index=True)
    new_metrics.to_csv(
        ext.OUTPUT_DIR / "socioeconomic_occupancy_feature_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    new_tuning.to_csv(
        ext.OUTPUT_DIR / "socioeconomic_occupancy_feature_tuning_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    previous_metrics_path = ext.OUTPUT_DIR / "external_feature_comparison_metrics_with_all.csv"
    previous_metrics = pd.read_csv(previous_metrics_path)
    combined_metrics = pd.concat(
        [
            previous_metrics[
                ~previous_metrics["feature_set"].isin(
                    ["socio_occupancy", "weather_socio_occupancy", "all_external_plus_socio_occupancy"]
                )
            ],
            new_metrics,
        ],
        ignore_index=True,
    )
    combined_metrics.to_csv(
        ext.OUTPUT_DIR / "external_feature_comparison_metrics_with_socioeconomic_occupancy.csv",
        index=False,
        encoding="utf-8-sig",
    )

    previous_tuning_path = ext.OUTPUT_DIR / "external_feature_tuning_results_with_all.csv"
    if previous_tuning_path.exists():
        previous_tuning = pd.read_csv(previous_tuning_path)
        combined_tuning = pd.concat(
            [
                previous_tuning[
                    ~previous_tuning["feature_set"].isin(
                        ["socio_occupancy", "weather_socio_occupancy", "all_external_plus_socio_occupancy"]
                    )
                ],
                new_tuning,
            ],
            ignore_index=True,
        )
    else:
        combined_tuning = new_tuning
    combined_tuning.to_csv(
        ext.OUTPUT_DIR / "external_feature_tuning_results_with_socioeconomic_occupancy.csv",
        index=False,
        encoding="utf-8-sig",
    )

    join_summary = pd.DataFrame(
        [
            socio_summary,
            {
                **weather_socio_summary,
                "feature_set": "weather_socio_occupancy",
                "weather_matched_rows": weather_summary.get("matched_rows"),
                "weather_matched_row_rate": weather_summary.get("matched_row_rate"),
            },
            {
                **all_plus_socio_summary,
                "feature_set": "all_external_plus_socio_occupancy",
                "weather_matched_rows": weather_summary.get("matched_rows"),
                "weather_matched_row_rate": weather_summary.get("matched_row_rate"),
                "footprint_matched_rows": footprint_summary.get("matched_rows"),
                "footprint_matched_row_rate": footprint_summary.get("matched_row_rate"),
                "permit_matched_rows": permit_summary.get("matched_rows"),
                "permit_matched_row_rate": permit_summary.get("matched_row_rate"),
                "recent_permit_rows": permit_summary.get("recent_permit_rows"),
                "recent_permit_row_rate": permit_summary.get("recent_permit_row_rate"),
            },
        ]
    )
    join_summary.to_csv(
        ext.OUTPUT_DIR / "socioeconomic_occupancy_join_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_socio_comparison(combined_metrics[combined_metrics["split"].eq("test")])
    write_report(combined_metrics, combined_tuning, join_summary)

    print("Done socioeconomic/occupancy comparison.")
    print(
        combined_metrics[combined_metrics["split"].eq("test")]
        .sort_values(["feature_set", "rmse"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
