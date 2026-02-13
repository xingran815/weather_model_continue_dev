"""Tests for data preprocessing module: normalization and data cleaning."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocessing import preprocessing, vector_normalize


class TestVectorNormalize:
    """Tests for vector_normalize function."""

    def test_vector_normalize_basic(self) -> None:
        """Test basic L2 normalization and edge cases."""
        df = pd.DataFrame({
            "a": [3.0, 4.0, 0.0, np.nan],
            "b": [4.0, 3.0, 0.0, 4.0],
        })
        result = vector_normalize(df)

        # Check shape and no NaN in output
        assert result.shape == df.shape
        assert not result.isnull().any().any()

        # Check L2 norm is 1 for non-zero rows
        norms = np.linalg.norm(result.values[:2], axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0])

        # Zero row stays zero
        assert result.iloc[2, 0] == 0.0
        assert result.iloc[2, 1] == 0.0


class TestPreprocessing:
    """Tests for full preprocessing pipeline."""

    def test_preprocessing_creates_valid_output(self, sample_raw_csv: str, tmp_path: Any) -> None:
        """Test that preprocessing creates valid output file with expected schema."""
        model_args = {
            "raw_data_file": sample_raw_csv,
            "processed_data_file": None,
            "date": "20240101_1200",
            "sample_percent": 0.2,
            "duration": 10,
        }

        original_dir = os.getcwd()
        try:
            test_dir = tmp_path / "test_preprocessing"
            test_dir.mkdir()
            os.chdir(test_dir)

            preprocessing(model_args)

            # Check file was created
            assert model_args["processed_data_file"] is not None
            assert os.path.exists(model_args["processed_data_file"])

            # Load and validate output
            df = pd.read_csv(model_args["processed_data_file"])

            # Date column should be removed
            assert "Date" not in df.columns

            # No missing values
            assert not df.isnull().any().any()

            # Target is boolean/int
            assert df["RainTomorrow"].dtype in [bool, int, np.int64]
        finally:
            os.chdir(original_dir)

    def test_preprocessing_one_hot_encodes_categoricals(self, sample_raw_csv: str, tmp_path: Any) -> None:
        """Test that categorical variables are one-hot encoded."""
        model_args = {
            "raw_data_file": sample_raw_csv,
            "processed_data_file": None,
            "date": "20240101_1200",
            "sample_percent": 0.2,
            "duration": 10,
        }

        original_dir = os.getcwd()
        try:
            test_dir = tmp_path / "test_preprocessing"
            test_dir.mkdir()
            os.chdir(test_dir)

            preprocessing(model_args)
            df = pd.read_csv(model_args["processed_data_file"])

            # Check for encoded columns
            location_cols = [col for col in df.columns if col.startswith("Location_")]
            assert len(location_cols) > 0

            # Original categorical columns should not exist
            assert "Location" not in df.columns
        finally:
            os.chdir(original_dir)

    def test_preprocessing_handles_missing_values(self, tmp_path: Any) -> None:
        """Test that missing values and 'NA' strings are handled correctly."""
        df_with_issues = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=15, freq="D"),
            "Location": ["Sydney"] * 15,
            "MinTemp": [15.0, np.nan, 18.0] + [16.0] * 12,
            "MaxTemp": [25.0] * 15,
            "Rainfall": [2.0] * 15,
            "Evaporation": [5.0] * 15,
            "Sunshine": [8.0] * 15,
            "WindGustDir": ["N"] * 15,
            "WindGustSpeed": [30.0] * 15,
            "WindDir9am": ["N"] * 15,
            "WindDir3pm": ["S"] * 15,
            "WindSpeed9am": [15.0] * 15,
            "WindSpeed3pm": [20.0] * 15,
            "Humidity9am": [60.0] * 15,
            "Humidity3pm": [55.0] * 15,
            "Pressure9am": [1015.0] * 15,
            "Pressure3pm": [1013.0] * 15,
            "Cloud9am": [4] * 15,
            "Cloud3pm": [5] * 15,
            "Temp9am": [18.0] * 15,
            "Temp3pm": [23.0] * 15,
            "RainToday": ["No"] * 15,
            "RainTomorrow": ["Yes", "No", "NA"] + ["No"] * 12,
        })

        csv_path = tmp_path / "data_with_issues.csv"
        df_with_issues.to_csv(csv_path, index=False)

        model_args = {
            "raw_data_file": str(csv_path),
            "processed_data_file": None,
            "date": "20240101_1200",
            "sample_percent": 0.2,
            "duration": 10,
        }

        original_dir = os.getcwd()
        try:
            test_dir = tmp_path / "test_preprocessing"
            test_dir.mkdir()
            os.chdir(test_dir)

            preprocessing(model_args)
            df = pd.read_csv(model_args["processed_data_file"])

            # No NaN in output (filled or dropped)
            assert not df.isnull().any().any()

            # 'NA' in target should be dropped
            assert len(df) < len(df_with_issues)
        finally:
            os.chdir(original_dir)
