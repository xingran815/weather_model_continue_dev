"""Tests for data validation: schema and value checks."""

from __future__ import annotations

import pandas as pd


class TestDataValidation:
    """Tests for data quality and schema validation."""

    def test_raw_data_has_required_columns(self, sample_raw_dataframe: pd.DataFrame) -> None:
        """Test that raw data has all expected columns."""
        required_columns = [
            "Date",
            "Location",
            "MinTemp",
            "MaxTemp",
            "Rainfall",
            "RainToday",
            "RainTomorrow",
        ]

        for col in required_columns:
            assert col in sample_raw_dataframe.columns

    def test_raw_data_column_types(self, sample_raw_dataframe: pd.DataFrame) -> None:
        """Test that raw data columns have expected types."""
        df = sample_raw_dataframe

        # Numeric columns
        numeric_cols = ["MinTemp", "MaxTemp", "Rainfall", "Humidity9am", "Pressure9am"]
        for col in numeric_cols:
            if col in df.columns:
                assert pd.api.types.is_numeric_dtype(df[col])

        # Object/string columns
        object_cols = ["Location", "WindGustDir", "RainToday", "RainTomorrow"]
        for col in object_cols:
            if col in df.columns:
                assert df[col].dtype == object or df[col].dtype == "string"

    def test_processed_data_no_nulls(self, sample_processed_dataframe: pd.DataFrame) -> None:
        """Test that processed data has no null values."""
        assert not sample_processed_dataframe.isnull().any().any()

    def test_processed_data_all_numeric(self, sample_processed_dataframe: pd.DataFrame) -> None:
        """Test that all processed data columns are numeric."""
        for col in sample_processed_dataframe.columns:
            assert pd.api.types.is_numeric_dtype(sample_processed_dataframe[col]) or pd.api.types.is_bool_dtype(
                sample_processed_dataframe[col]
            )

    def test_processed_data_target_is_binary(self, sample_processed_dataframe: pd.DataFrame) -> None:
        """Test that RainTomorrow in processed data is binary."""
        unique_values = sample_processed_dataframe["RainTomorrow"].unique()
        assert len(unique_values) <= 2
        assert all(val in [0, 1, True, False] for val in unique_values)

    def test_processed_data_has_features(self, sample_processed_dataframe: pd.DataFrame) -> None:
        """Test that processed data has sufficient features for modeling."""
        # Should have more than just target columns
        feature_cols = [col for col in sample_processed_dataframe.columns if col not in ["RainTomorrow", "RainToday"]]
        assert len(feature_cols) > 10  # Should have multiple features

    def test_temperature_ranges_valid(self, sample_raw_dataframe: pd.DataFrame) -> None:
        """Test that temperature values are in reasonable ranges."""
        df = sample_raw_dataframe

        if "MinTemp" in df.columns:
            # Remove NaNs for this check
            min_temps = df["MinTemp"].dropna()
            assert min_temps.min() >= -20  # Reasonable minimum
            assert min_temps.max() <= 50  # Reasonable maximum

        if "MaxTemp" in df.columns:
            max_temps = df["MaxTemp"].dropna()
            assert max_temps.min() >= -10
            assert max_temps.max() <= 55

    def test_humidity_ranges_valid(self, sample_raw_dataframe: pd.DataFrame) -> None:
        """Test that humidity values are between 0 and 100."""
        df = sample_raw_dataframe

        for col in ["Humidity9am", "Humidity3pm"]:
            if col in df.columns:
                humidity = df[col].dropna()
                assert humidity.min() >= 0
                assert humidity.max() <= 100

    def test_categorical_values_valid(self, sample_raw_dataframe: pd.DataFrame) -> None:
        """Test that categorical columns have expected values."""
        df = sample_raw_dataframe

        # RainToday/RainTomorrow should be Yes/No (ignoring NaN)
        for col in ["RainToday", "RainTomorrow"]:
            if col in df.columns:
                values = df[col].dropna().unique()
                assert all(val in ["Yes", "No", "NA"] for val in values)

        # Wind directions should be valid compass directions
        valid_dirs = [
            "N",
            "NE",
            "E",
            "SE",
            "S",
            "SW",
            "W",
            "NW",
            "NNE",
            "ENE",
            "ESE",
            "SSE",
            "SSW",
            "WSW",
            "WNW",
            "NNW",
        ]
        for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
            if col in df.columns:
                values = df[col].dropna().unique()
                assert all(val in valid_dirs for val in values)
