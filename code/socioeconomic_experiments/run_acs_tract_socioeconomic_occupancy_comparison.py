from __future__ import annotations

import json
import os
import struct
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import compare_external_features as ext


ACS_API_BASE = "https://api.census.gov/data/{year}/acs/acs5/profile"
TIGER_TRACT_ZIP = "https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_17_tract.zip"
STATE_FIPS = "17"
COOK_COUNTY_FIPS = "031"

ACS_VARIABLES = {
    "DP03_0062E": "acs_median_household_income",
    "DP03_0088E": "acs_per_capita_income",
    "DP03_0009PE": "acs_unemployment_rate",
    "DP03_0128PE": "acs_poverty_rate",
    "DP02_0067PE": "acs_high_school_or_higher_pct",
    "DP02_0068PE": "acs_bachelor_or_higher_pct",
    "DP02_0001E": "acs_total_households",
    "DP02_0016E": "acs_avg_household_size",
    "DP04_0001E": "acs_total_housing_units",
    "DP04_0002E": "acs_occupied_housing_units",
    "DP04_0003PE": "acs_vacancy_rate",
    "DP04_0046PE": "acs_owner_occupied_pct",
    "DP04_0047PE": "acs_renter_occupied_pct",
    "DP04_0078PE": "acs_occupants_per_room_1_01_to_1_50_pct",
    "DP04_0079PE": "acs_occupants_per_room_1_51_plus_pct",
}

ACS_SOCIO_OCCUPANCY_FEATURES = [
    "acs_tract_context_matched",
    "acs_median_household_income",
    "acs_per_capita_income",
    "acs_unemployment_rate",
    "acs_poverty_rate",
    "acs_no_high_school_diploma_pct",
    "acs_bachelor_or_higher_pct",
    "acs_total_households",
    "acs_avg_household_size",
    "acs_total_housing_units",
    "acs_occupied_housing_units",
    "acs_vacancy_rate",
    "acs_owner_occupied_pct",
    "acs_renter_occupied_pct",
    "acs_crowded_housing_pct",
]


def require_census_key() -> str:
    key = os.environ.get("CENSUS_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "CENSUS_API_KEY is not set. Set it in the shell before running this script; "
            "do not commit the key into source files."
        )
    return key


def fetch_json(url: str, retries: int = 3) -> object:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 data-mining-course-project"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read()
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(attempt * 2)
    assert last_error is not None
    raise last_error


def download_file(url: str, path: Path) -> None:
    if path.exists():
        return
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 data-mining-course-project"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
    path.write_bytes(data)


def download_acs_profile_year(year: int, api_key: str) -> pd.DataFrame:
    cache_path = ext.RAW_DIR / f"acs5_profile_cook_county_tracts_{year}.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path, low_memory=False, dtype={"state": str, "county": str, "tract": str, "tract_geoid": str})
        df["tract_geoid"] = df["tract_geoid"].astype(str).str.zfill(11)
        return df

    variables = ["NAME"] + list(ACS_VARIABLES.keys())
    params = {
        "get": ",".join(variables),
        "for": "tract:*",
        "in": f"state:{STATE_FIPS} county:{COOK_COUNTY_FIPS}",
        "key": api_key,
    }
    url = ACS_API_BASE.format(year=year) + "?" + urllib.parse.urlencode(params, safe=",:* ")
    data = fetch_json(url)
    if not isinstance(data, list) or len(data) <= 1:
        raise RuntimeError(f"ACS API returned no rows for {year}")

    header = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=header)
    df["data_year"] = year
    df["tract_geoid"] = df["state"].astype(str) + df["county"].astype(str) + df["tract"].astype(str)
    df["tract_geoid"] = df["tract_geoid"].astype(str).str.zfill(11)
    df = df.rename(columns=ACS_VARIABLES)

    for col in ACS_VARIABLES.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col].isin([-666666666, -999999999, -888888888, -222222222, -333333333, -555555555]), col] = np.nan

    df["acs_no_high_school_diploma_pct"] = 100 - df["acs_high_school_or_higher_pct"]
    df["acs_crowded_housing_pct"] = (
        df["acs_occupants_per_room_1_01_to_1_50_pct"].fillna(0)
        + df["acs_occupants_per_room_1_51_plus_pct"].fillna(0)
    )

    keep_cols = ["data_year", "tract_geoid", "NAME"] + [
        col for col in ACS_SOCIO_OCCUPANCY_FEATURES if col != "acs_tract_context_matched"
    ]
    df = df[keep_cols].copy()
    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return df


def download_acs_features() -> pd.DataFrame:
    api_key = require_census_key()
    frames = [download_acs_profile_year(year, api_key) for year in range(2018, 2024)]
    features = pd.concat(frames, ignore_index=True)
    features["tract_geoid"] = features["tract_geoid"].astype(str).str.zfill(11)
    features.to_csv(
        ext.PROCESSED_DIR / "external_acs_tract_socioeconomic_occupancy_features_2018_2023.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return features


def read_dbf_records(dbf_bytes: bytes) -> list[dict[str, str]]:
    record_count = struct.unpack("<I", dbf_bytes[4:8])[0]
    header_len = struct.unpack("<H", dbf_bytes[8:10])[0]
    record_len = struct.unpack("<H", dbf_bytes[10:12])[0]

    fields = []
    offset = 32
    field_start = 1
    while offset < header_len and dbf_bytes[offset] != 0x0D:
        descriptor = dbf_bytes[offset : offset + 32]
        name = descriptor[:11].split(b"\x00", 1)[0].decode("latin1").strip()
        length = descriptor[16]
        fields.append((name, field_start, length))
        field_start += length
        offset += 32

    records: list[dict[str, str]] = []
    pos = header_len
    for _ in range(record_count):
        raw = dbf_bytes[pos : pos + record_len]
        pos += record_len
        if not raw or raw[0:1] == b"*":
            continue
        record = {}
        for name, start, length in fields:
            record[name] = raw[start : start + length].decode("latin1", errors="ignore").strip()
        records.append(record)
    return records


def read_shp_polygons(shp_bytes: bytes, records: list[dict[str, str]]) -> list[dict]:
    polygons = []
    pos = 100
    record_index = 0
    while pos + 8 <= len(shp_bytes) and record_index < len(records):
        content_len_words = struct.unpack(">i", shp_bytes[pos + 4 : pos + 8])[0]
        content_start = pos + 8
        content_end = content_start + content_len_words * 2
        content = shp_bytes[content_start:content_end]
        pos = content_end
        record = records[record_index]
        record_index += 1

        if len(content) < 44:
            continue
        shape_type = struct.unpack("<i", content[:4])[0]
        if shape_type != 5:
            continue

        xmin, ymin, xmax, ymax = struct.unpack("<4d", content[4:36])
        num_parts, num_points = struct.unpack("<2i", content[36:44])
        parts_start = 44
        points_start = parts_start + num_parts * 4
        if points_start + num_points * 16 > len(content):
            continue

        parts = list(struct.unpack(f"<{num_parts}i", content[parts_start:points_start]))
        points = [
            struct.unpack("<2d", content[points_start + i * 16 : points_start + (i + 1) * 16])
            for i in range(num_points)
        ]
        rings = []
        for i, start in enumerate(parts):
            end = parts[i + 1] if i + 1 < len(parts) else num_points
            rings.append(points[start:end])

        if record.get("COUNTYFP") != COOK_COUNTY_FIPS:
            continue
        polygons.append(
            {
                "geoid": record.get("GEOID"),
                "bbox": (xmin, ymin, xmax, ymax),
                "rings": rings,
            }
        )
    return polygons


def load_tract_polygons(year: int) -> list[dict]:
    cache_path = ext.RAW_DIR / f"tl_{year}_17_tract.zip"
    download_file(TIGER_TRACT_ZIP.format(year=year), cache_path)
    with zipfile.ZipFile(cache_path) as archive:
        shp_name = next(name for name in archive.namelist() if name.endswith(".shp"))
        dbf_name = next(name for name in archive.namelist() if name.endswith(".dbf"))
        dbf_records = read_dbf_records(archive.read(dbf_name))
        polygons = read_shp_polygons(archive.read(shp_name), dbf_records)
    if not polygons:
        raise RuntimeError(f"No Cook County tract polygons loaded for {year}")
    return polygons


def point_in_ring(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    if len(ring) < 3:
        return False
    x1, y1 = ring[-1]
    for x2, y2 in ring:
        if (y1 > y) != (y2 > y):
            x_intersect = (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-300) + x1
            if x < x_intersect:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def polygon_contains_point(poly: dict, x: float, y: float) -> bool:
    xmin, ymin, xmax, ymax = poly["bbox"]
    if x < xmin or x > xmax or y < ymin or y > ymax:
        return False
    inside = False
    for ring in poly["rings"]:
        if point_in_ring(x, y, ring):
            inside = not inside
    return inside


def get_tract_geoid_for_point(longitude: float, latitude: float, polygons: list[dict]) -> str | None:
    for poly in polygons:
        if polygon_contains_point(poly, longitude, latitude):
            return poly["geoid"]
    return None


def build_tract_lookup(df: pd.DataFrame) -> pd.DataFrame:
    cache_path = ext.PROCESSED_DIR / "external_acs_tract_lookup_2018_2023.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, low_memory=False, dtype={"tract_geoid": str})
    else:
        cached = pd.DataFrame(columns=["data_year", "latitude_key", "longitude_key", "tract_geoid"])

    cached_keys = set(zip(cached["data_year"], cached["latitude_key"], cached["longitude_key"]))

    unique_points = (
        df[["data_year", "latitude", "longitude"]]
        .dropna()
        .assign(
            data_year=lambda d: d["data_year"].astype(int),
            latitude_key=lambda d: d["latitude"].round(6),
            longitude_key=lambda d: d["longitude"].round(6),
        )[["data_year", "latitude_key", "longitude_key"]]
        .drop_duplicates()
        .sort_values(["data_year", "latitude_key", "longitude_key"])
        .reset_index(drop=True)
    )

    rows: list[dict] = []
    missing = unique_points[
        ~unique_points.apply(lambda r: (r["data_year"], r["latitude_key"], r["longitude_key"]) in cached_keys, axis=1)
    ]
    total = len(missing)
    done = 0
    for year, year_points in missing.groupby("data_year"):
        polygons = load_tract_polygons(int(year))
        for row in year_points.itertuples(index=False):
            tract_geoid = get_tract_geoid_for_point(row.longitude_key, row.latitude_key, polygons)
            rows.append(
                {
                    "data_year": int(row.data_year),
                    "latitude_key": float(row.latitude_key),
                    "longitude_key": float(row.longitude_key),
                    "tract_geoid": tract_geoid,
                }
            )
            done += 1
            if done % 500 == 0 or done == total:
                print(f"Assigned tract GEOID for {done:,}/{total:,} missing point-year rows")

    if rows:
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached = cached.drop_duplicates(["data_year", "latitude_key", "longitude_key"], keep="last")
        cached.to_csv(cache_path, index=False, encoding="utf-8-sig")

    return cached


def add_acs_tract_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    out["data_year"] = out["data_year"].astype(int)
    out["latitude_key"] = out["latitude"].round(6)
    out["longitude_key"] = out["longitude"].round(6)

    lookup = build_tract_lookup(out)
    acs = download_acs_features()
    lookup["tract_geoid"] = lookup["tract_geoid"].astype("string")
    acs["tract_geoid"] = acs["tract_geoid"].astype("string")
    out = out.merge(lookup, on=["data_year", "latitude_key", "longitude_key"], how="left")
    out["tract_geoid"] = out["tract_geoid"].astype("string")
    out = out.merge(acs, on=["data_year", "tract_geoid"], how="left")
    out["acs_tract_context_matched"] = out["acs_median_household_income"].notna().astype(int)

    for col in ACS_SOCIO_OCCUPANCY_FEATURES:
        if col not in out.columns:
            out[col] = np.nan

    summary = {
        "feature_set": "acs_socio_occupancy",
        "source": "U.S. Census ACS 5-Year Data Profiles, tract-level",
        "geography_lookup": "Census TIGER/Line tract polygons, local point-in-polygon lookup",
        "model_rows": int(len(out)),
        "tract_lookup_rows": int(len(lookup)),
        "unique_tracts_matched": int(out.loc[out["acs_tract_context_matched"].eq(1), "tract_geoid"].nunique()),
        "matched_rows": int(out["acs_tract_context_matched"].sum()),
        "matched_row_rate": float(out["acs_tract_context_matched"].mean()),
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


def plot_acs_comparison(test_metrics: pd.DataFrame) -> None:
    model_order = ["Group Mean Baseline", "Random Forest", "XGBoost"]
    feature_order = [
        "base",
        "weather",
        "acs_socio_occupancy",
        "weather_acs_socio_occupancy",
        "all_external",
        "all_external_plus_acs_socio_occupancy",
    ]
    pivot = (
        test_metrics[test_metrics["model"].isin(model_order)]
        .pivot(index="feature_set", columns="model", values="rmse")
        .reindex(feature_order)
    )
    ax = pivot.plot(kind="bar", figsize=(12, 6), color=["#7F7F7F", "#2F6F73", "#7A5195"])
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Test RMSE")
    ax.set_title("ACS Tract-Level Socioeconomic and Occupancy Feature Comparison")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(ext.PLOT_DIR / "acs_tract_socioeconomic_occupancy_test_rmse_comparison.png", dpi=180)
    plt.close()


def write_report(metrics: pd.DataFrame, join_summary: pd.DataFrame) -> None:
    test = metrics[metrics["split"].eq("test")].copy()
    best_by_set = test.sort_values(["feature_set", "rmse"]).groupby("feature_set", as_index=False).first()
    base_best_rmse = best_by_set.loc[best_by_set["feature_set"].eq("base"), "rmse"].iloc[0]
    weather_best_rmse = best_by_set.loc[best_by_set["feature_set"].eq("weather"), "rmse"].iloc[0]
    best_by_set["rmse_change_vs_base_best"] = best_by_set["rmse"] - base_best_rmse
    best_by_set["rmse_change_pct_vs_base_best"] = best_by_set["rmse_change_vs_base_best"] / base_best_rmse * 100
    best_by_set["rmse_change_vs_weather_best"] = best_by_set["rmse"] - weather_best_rmse
    best_by_set["rmse_change_pct_vs_weather_best"] = best_by_set["rmse_change_vs_weather_best"] / weather_best_rmse * 100

    feature_defs = pd.DataFrame(
        [
            ["acs_median_household_income", "ACS tract median household income"],
            ["acs_per_capita_income", "ACS tract per capita income"],
            ["acs_unemployment_rate", "ACS tract unemployment rate"],
            ["acs_poverty_rate", "ACS tract poverty rate for all people"],
            ["acs_no_high_school_diploma_pct", "100 - percent high school graduate or higher"],
            ["acs_bachelor_or_higher_pct", "Percent population 25+ with bachelor's degree or higher"],
            ["acs_total_households", "Total households"],
            ["acs_avg_household_size", "Average household size"],
            ["acs_total_housing_units", "Total housing units"],
            ["acs_occupied_housing_units", "Occupied housing units"],
            ["acs_vacancy_rate", "Percent vacant housing units"],
            ["acs_owner_occupied_pct", "Percent owner-occupied units"],
            ["acs_renter_occupied_pct", "Percent renter-occupied units"],
            ["acs_crowded_housing_pct", "Percent units with 1.01+ occupants per room"],
            ["acs_tract_context_matched", "1 if building-year was matched to an ACS tract record, else 0"],
        ],
        columns=["feature", "definition"],
    )

    report = f"""# ACS Tract-Level Socioeconomic and Occupancy Feature Comparison

This experiment replaces the earlier community-level socioeconomic fallback with U.S. Census ACS 5-Year Data Profile features at census tract level.

Building coordinates were assigned to census tracts using Census TIGER/Line tract polygons and a local point-in-polygon lookup. ACS Data Profile variables were downloaded for 2018-2023 and joined by `data_year + tract_geoid`.

## Added Features

{md_table(feature_defs, ["feature", "definition"])}

## Join Coverage

{md_table(join_summary, list(join_summary.columns), digits=4)}

## Test Metrics

{md_table(test.sort_values(["feature_set", "rmse"]), ["feature_set", "model", "rmse", "mae", "r2"], digits=4)}

## Best Model by Feature Set

{md_table(best_by_set, ["feature_set", "model", "rmse", "mae", "r2", "rmse_change_vs_base_best", "rmse_change_pct_vs_base_best", "rmse_change_vs_weather_best", "rmse_change_pct_vs_weather_best"], digits=4)}

## Plot

![ACS Tract Socioeconomic Occupancy RMSE Comparison](../outputs/plots/acs_tract_socioeconomic_occupancy_test_rmse_comparison.png)

## Notes

- ACS 5-Year estimates are rolling multi-year estimates, not true single-year building-level occupancy.
- These features improve spatial granularity compared with community-area socioeconomic indicators, but still represent neighborhood context rather than actual building occupancy.
- The Census API key is read from the `CENSUS_API_KEY` environment variable and is not stored in project files.
"""
    (ext.REPORT_DIR / "acs_tract_socioeconomic_occupancy_feature_comparison.md").write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    ext.ensure_dirs()
    base_df = pd.read_csv(ext.MODELING_TABLE, low_memory=False)
    base_df = ext.add_address_keys(base_df)

    acs_df, acs_summary = add_acs_tract_features(base_df.copy())
    weather_df, weather_summary = ext.add_weather_features(base_df.copy())
    weather_acs_df, weather_acs_summary = add_acs_tract_features(weather_df)

    footprint_df, footprint_summary = ext.add_footprint_features(weather_df)
    all_external_df, permit_summary = ext.add_permit_features(footprint_df)
    all_plus_acs_df, all_plus_acs_summary = add_acs_tract_features(all_external_df)

    variants = [
        ("acs_socio_occupancy", acs_df, ACS_SOCIO_OCCUPANCY_FEATURES),
        ("weather_acs_socio_occupancy", weather_acs_df, ext.WEATHER_FEATURES + ACS_SOCIO_OCCUPANCY_FEATURES),
        (
            "all_external_plus_acs_socio_occupancy",
            all_plus_acs_df,
            ext.WEATHER_FEATURES + ext.FOOTPRINT_FEATURES + ext.PERMIT_FEATURES + ACS_SOCIO_OCCUPANCY_FEATURES,
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
        ext.OUTPUT_DIR / "acs_tract_socioeconomic_occupancy_feature_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    new_tuning.to_csv(
        ext.OUTPUT_DIR / "acs_tract_socioeconomic_occupancy_feature_tuning_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    previous_metrics = pd.read_csv(ext.OUTPUT_DIR / "external_feature_comparison_metrics_with_all.csv")
    combined_metrics = pd.concat(
        [
            previous_metrics[
                ~previous_metrics["feature_set"].isin(
                    [
                        "acs_socio_occupancy",
                        "weather_acs_socio_occupancy",
                        "all_external_plus_acs_socio_occupancy",
                    ]
                )
            ],
            new_metrics,
        ],
        ignore_index=True,
    )
    combined_metrics.to_csv(
        ext.OUTPUT_DIR / "external_feature_comparison_metrics_with_acs_tract_socioeconomic_occupancy.csv",
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
                        [
                            "acs_socio_occupancy",
                            "weather_acs_socio_occupancy",
                            "all_external_plus_acs_socio_occupancy",
                        ]
                    )
                ],
                new_tuning,
            ],
            ignore_index=True,
        )
    else:
        combined_tuning = new_tuning
    combined_tuning.to_csv(
        ext.OUTPUT_DIR / "external_feature_tuning_results_with_acs_tract_socioeconomic_occupancy.csv",
        index=False,
        encoding="utf-8-sig",
    )

    join_summary = pd.DataFrame(
        [
            acs_summary,
            {
                **weather_acs_summary,
                "feature_set": "weather_acs_socio_occupancy",
                "weather_matched_rows": weather_summary.get("matched_rows"),
                "weather_matched_row_rate": weather_summary.get("matched_row_rate"),
            },
            {
                **all_plus_acs_summary,
                "feature_set": "all_external_plus_acs_socio_occupancy",
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
        ext.OUTPUT_DIR / "acs_tract_socioeconomic_occupancy_join_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_acs_comparison(combined_metrics[combined_metrics["split"].eq("test")])
    write_report(combined_metrics, join_summary)

    print("Done ACS tract-level socioeconomic/occupancy comparison.")
    print(
        combined_metrics[combined_metrics["split"].eq("test")]
        .sort_values(["feature_set", "rmse"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
