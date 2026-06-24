from __future__ import annotations

import pandas as pd

import compare_external_features as ext


def main() -> None:
    ext.ensure_dirs()
    df = pd.read_csv(ext.MODELING_TABLE, low_memory=False)
    df = ext.add_address_keys(df)

    weather_df, weather_summary = ext.add_weather_features(df.copy())
    footprint_df, footprint_summary = ext.add_footprint_features(weather_df)
    all_df, permit_summary = ext.add_permit_features(footprint_df)

    all_summary = {
        "feature_set": "all_external",
        "matched_rows": len(all_df),
        "matched_row_rate": 1.0,
        "weather_matched_rows": weather_summary.get("matched_rows"),
        "weather_matched_row_rate": weather_summary.get("matched_row_rate"),
        "footprint_matched_rows": footprint_summary.get("matched_rows"),
        "footprint_matched_row_rate": footprint_summary.get("matched_row_rate"),
        "permit_matched_rows": permit_summary.get("matched_rows"),
        "permit_matched_row_rate": permit_summary.get("matched_row_rate"),
        "recent_permit_rows": permit_summary.get("recent_permit_rows"),
        "recent_permit_row_rate": permit_summary.get("recent_permit_row_rate"),
    }

    extra_numeric = ext.WEATHER_FEATURES + ext.FOOTPRINT_FEATURES + ext.PERMIT_FEATURES
    metrics, tuning = ext.evaluate_variant("all_external", all_df, extra_numeric)

    metrics.to_csv(ext.OUTPUT_DIR / "all_external_feature_metrics.csv", index=False, encoding="utf-8-sig")
    tuning.to_csv(ext.OUTPUT_DIR / "all_external_feature_tuning_results.csv", index=False, encoding="utf-8-sig")

    old_metrics = pd.read_csv(ext.OUTPUT_DIR / "external_feature_comparison_metrics.csv")
    combined_metrics = pd.concat(
        [old_metrics[old_metrics["feature_set"].ne("all_external")], metrics],
        ignore_index=True,
    )
    combined_metrics.to_csv(
        ext.OUTPUT_DIR / "external_feature_comparison_metrics_with_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    old_tuning = pd.read_csv(ext.OUTPUT_DIR / "external_feature_tuning_results.csv")
    combined_tuning = pd.concat(
        [old_tuning[old_tuning["feature_set"].ne("all_external")], tuning],
        ignore_index=True,
    )
    combined_tuning.to_csv(
        ext.OUTPUT_DIR / "external_feature_tuning_results_with_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    old_summary = pd.read_csv(ext.OUTPUT_DIR / "external_feature_join_summary.csv")
    all_summary_df = pd.DataFrame([all_summary])
    combined_summary = pd.concat(
        [old_summary[old_summary["feature_set"].ne("all_external")], all_summary_df],
        ignore_index=True,
    )
    combined_summary.to_csv(
        ext.OUTPUT_DIR / "external_feature_join_summary_with_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ext.plot_comparison_with_all(combined_metrics[combined_metrics["split"].eq("test")])
    ext.write_report_with_all(combined_metrics, combined_tuning, combined_summary)

    print("Done all external comparison.")
    print(
        combined_metrics[combined_metrics["split"].eq("test")]
        .sort_values(["feature_set", "rmse"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
