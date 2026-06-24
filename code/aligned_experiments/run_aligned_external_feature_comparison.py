from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import compare_external_features as ext


FEATURE_ORDER = [
    "aligned_base",
    "aligned_weather",
    "aligned_footprints",
    "aligned_permits",
    "aligned_weather_footprints",
    "aligned_all_external",
]
DISPLAY_NAMES = {
    "aligned_base": "Base",
    "aligned_weather": "Weather",
    "aligned_footprints": "Footprints",
    "aligned_permits": "Permits",
    "aligned_weather_footprints": "Weather + Footprints",
    "aligned_all_external": "All External",
}


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


def plot_aligned_comparison(test_metrics: pd.DataFrame) -> None:
    model_order = ["Group Mean Baseline", "Random Forest", "XGBoost"]
    pivot = (
        test_metrics[test_metrics["model"].isin(model_order)]
        .pivot(index="feature_set", columns="model", values="rmse")
        .reindex(FEATURE_ORDER)
        .rename(index=DISPLAY_NAMES)
    )
    ax = pivot.plot(kind="bar", figsize=(11, 6), color=["#7F7F7F", "#2F6F73", "#7A5195"])
    ax.set_xlabel("Feature set on aligned sample")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Aligned External Feature Comparison: Test RMSE")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(ext.PLOT_DIR / "aligned_external_feature_test_rmse_comparison.png", dpi=180)
    plt.close()


def write_report(metrics: pd.DataFrame, tuning: pd.DataFrame, sample_summary: pd.DataFrame) -> None:
    test = metrics[metrics["split"].eq("test")].copy()
    best_by_set = (
        test.sort_values(["feature_set", "rmse"])
        .groupby("feature_set", as_index=False)
        .first()
        .set_index("feature_set")
        .reindex(FEATURE_ORDER)
        .reset_index()
    )
    base_best_rmse = best_by_set.loc[best_by_set["feature_set"].eq("aligned_base"), "rmse"].iloc[0]
    weather_best_rmse = best_by_set.loc[best_by_set["feature_set"].eq("aligned_weather"), "rmse"].iloc[0]
    best_by_set["rmse_change_vs_aligned_base"] = best_by_set["rmse"] - base_best_rmse
    best_by_set["rmse_change_pct_vs_aligned_base"] = (
        best_by_set["rmse_change_vs_aligned_base"] / base_best_rmse * 100
    )
    best_by_set["rmse_change_vs_aligned_weather"] = best_by_set["rmse"] - weather_best_rmse
    best_by_set["rmse_change_pct_vs_aligned_weather"] = (
        best_by_set["rmse_change_vs_aligned_weather"] / weather_best_rmse * 100
    )
    best_by_set["display_feature_set"] = best_by_set["feature_set"].map(DISPLAY_NAMES)

    test_table = test.copy()
    test_table["display_feature_set"] = test_table["feature_set"].map(DISPLAY_NAMES)
    test_table["feature_order"] = test_table["feature_set"].map({name: i for i, name in enumerate(FEATURE_ORDER)})
    test_table = test_table.sort_values(["feature_order", "rmse"])

    aligned_rows = int(sample_summary.loc[sample_summary["subset"].eq("aligned_sample"), "rows"].iloc[0])
    full_rows = int(sample_summary.loc[sample_summary["subset"].eq("full_modeling_table"), "rows"].iloc[0])
    retained_rate = aligned_rows / full_rows * 100

    report = f"""# Aligned External Feature Comparison

This experiment isolates whether building footprint and permit features have predictive value when the missing-coverage problem is controlled.

The original external-feature ablation keeps all Chicago Energy Benchmarking rows and joins external sources with a left join. That measures practical performance under real coverage. This aligned version instead uses one fixed sample for every feature set:

```text
footprint_matched == 1
permit_matched == 1
```

Weather coverage is complete, so it does not reduce the sample.

## Sample Definition

{md_table(sample_summary, list(sample_summary.columns), digits=4)}

The aligned sample keeps {aligned_rows:,} of {full_rows:,} building-year rows ({retained_rate:.2f}%).

## Models

Each feature set is re-tuned on the aligned sample using the same train/validation/test split and random-search spaces as the main ablation:

- Train: 2018-2021
- Validation: 2022
- Test: 2023
- Models: Group Mean Baseline, Random Forest, XGBoost

## Test Metrics

{md_table(test_table, ["display_feature_set", "model", "rmse", "mae", "r2"], digits=4)}

## Best Model By Feature Set

{md_table(best_by_set, ["display_feature_set", "model", "rmse", "mae", "r2", "rmse_change_vs_aligned_base", "rmse_change_pct_vs_aligned_base", "rmse_change_vs_aligned_weather", "rmse_change_pct_vs_aligned_weather"], digits=4)}

## Plot

![Aligned External Feature RMSE Comparison](../outputs/plots/aligned_external_feature_test_rmse_comparison.png)

## Interpretation Guide

Use this result together with the full-sample ablation:

- If footprints or permits improve here but not in the full sample, their signal is likely limited by external-data coverage.
- If they still do not improve here, their feature content is likely weak for this target under the current matching and feature-engineering method.
- If all external does not beat weather-only here, then footprints and permits are not adding useful information beyond weather even after controlling for sample alignment.
"""
    (ext.REPORT_DIR / "aligned_external_feature_comparison.md").write_text(report, encoding="utf-8")


def main() -> None:
    ext.ensure_dirs()
    df = pd.read_csv(ext.MODELING_TABLE, low_memory=False)
    df = ext.add_address_keys(df)

    weather_df, weather_summary = ext.add_weather_features(df.copy())
    footprint_df, footprint_summary = ext.add_footprint_features(weather_df)
    all_df, permit_summary = ext.add_permit_features(footprint_df)

    aligned_df = all_df[(all_df["footprint_matched"].eq(1)) & (all_df["permit_matched"].eq(1))].copy()
    aligned_df.to_csv(
        ext.PROCESSED_DIR / "aligned_external_modeling_table_2018_2023.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sample_summary = pd.DataFrame(
        [
            {
                "subset": "full_modeling_table",
                "rows": len(all_df),
                "row_rate_vs_full": 1.0,
                "train_2018_2021": int(all_df["data_year"].between(2018, 2021).sum()),
                "validation_2022": int(all_df["data_year"].eq(2022).sum()),
                "test_2023": int(all_df["data_year"].eq(2023).sum()),
                "weather_matched_rows": weather_summary.get("matched_rows"),
                "footprint_matched_rows": footprint_summary.get("matched_rows"),
                "permit_matched_rows": permit_summary.get("matched_rows"),
            },
            {
                "subset": "aligned_sample",
                "rows": len(aligned_df),
                "row_rate_vs_full": len(aligned_df) / len(all_df),
                "train_2018_2021": int(aligned_df["data_year"].between(2018, 2021).sum()),
                "validation_2022": int(aligned_df["data_year"].eq(2022).sum()),
                "test_2023": int(aligned_df["data_year"].eq(2023).sum()),
                "weather_matched_rows": int(aligned_df[ext.WEATHER_FEATURES[0]].notna().sum()),
                "footprint_matched_rows": int(aligned_df["footprint_matched"].sum()),
                "permit_matched_rows": int(aligned_df["permit_matched"].sum()),
            },
        ]
    )
    sample_summary.to_csv(
        ext.OUTPUT_DIR / "aligned_external_feature_sample_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    variants: dict[str, list[str]] = {
        "aligned_base": [],
        "aligned_weather": ext.WEATHER_FEATURES,
        "aligned_footprints": ext.FOOTPRINT_FEATURES,
        "aligned_permits": ext.PERMIT_FEATURES,
        "aligned_weather_footprints": ext.WEATHER_FEATURES + ext.FOOTPRINT_FEATURES,
        "aligned_all_external": ext.WEATHER_FEATURES + ext.FOOTPRINT_FEATURES + ext.PERMIT_FEATURES,
    }

    metrics_path = ext.OUTPUT_DIR / "aligned_external_feature_metrics.csv"
    tuning_path = ext.OUTPUT_DIR / "aligned_external_feature_tuning_results.csv"
    existing_metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    existing_tuning = pd.read_csv(tuning_path) if tuning_path.exists() else pd.DataFrame()
    all_metrics = []
    all_tuning = []
    for name, extra_numeric in variants.items():
        if not existing_metrics.empty and name in set(existing_metrics["feature_set"]):
            metrics = existing_metrics[existing_metrics["feature_set"].eq(name)].copy()
            tuning = existing_tuning[existing_tuning["feature_set"].eq(name)].copy()
            print(f"Using existing aligned results for {name}")
        else:
            metrics, tuning = ext.evaluate_variant(name, aligned_df.copy(), extra_numeric)
        all_metrics.append(metrics)
        all_tuning.append(tuning)
        pd.concat(all_metrics, ignore_index=True).to_csv(
            ext.OUTPUT_DIR / "aligned_external_feature_metrics_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    tuning_df = pd.concat(all_tuning, ignore_index=True)
    metrics_df.to_csv(ext.OUTPUT_DIR / "aligned_external_feature_metrics.csv", index=False, encoding="utf-8-sig")
    tuning_df.to_csv(ext.OUTPUT_DIR / "aligned_external_feature_tuning_results.csv", index=False, encoding="utf-8-sig")
    plot_aligned_comparison(metrics_df[metrics_df["split"].eq("test")])
    write_report(metrics_df, tuning_df, sample_summary)

    print("Done aligned external feature comparison.")
    print(
        metrics_df[metrics_df["split"].eq("test")]
        .sort_values(["feature_set", "rmse"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
