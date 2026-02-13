"""Tests for predict_model module: prediction and evaluation."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.models.predict_model import predict


class TestPredict:
    """Tests for predict function."""

    @patch("joblib.load")
    @patch("mlflow.sklearn.load_model")
    def test_predict_with_ground_truth(
        self,
        mock_load_model: MagicMock,
        mock_joblib: MagicMock,
        sample_processed_csv: str,
        tmp_path: Any,
    ) -> None:
        """Test prediction with ground truth labels (evaluation mode)."""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 1] + [0] * 45)
        mock_load_model.return_value = mock_model

        # Load sample data to get feature names
        df = pd.read_csv(sample_processed_csv)
        feature_cols = [col for col in df.columns if col not in ["RainTomorrow", "RainToday"]]
        mock_joblib.return_value = feature_cols

        output_path = tmp_path / "predictions.csv"

        predict(input_path=sample_processed_csv, output_path=str(output_path))

        # Verify prediction file was created
        assert output_path.exists()

        # Verify predictions were added
        df_pred = pd.read_csv(output_path)
        assert "RainTomorrow_pred" in df_pred.columns
        assert len(df_pred) == len(df)

    @patch("joblib.load")
    @patch("mlflow.sklearn.load_model")
    def test_predict_without_ground_truth(
        self,
        mock_load_model: MagicMock,
        mock_joblib: MagicMock,
        tmp_path: Any,
    ) -> None:
        """Test prediction without ground truth labels."""
        # Create CSV without RainTomorrow column
        df = pd.DataFrame({
            "feature_1": np.random.uniform(-1, 1, 30),
            "feature_2": np.random.uniform(-1, 1, 30),
            "RainToday": np.random.choice([True, False], 30),
        })
        input_csv = tmp_path / "input_no_target.csv"
        df.to_csv(input_csv, index=False)

        # Setup mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1] * 15)
        mock_load_model.return_value = mock_model

        feature_cols = ["feature_1", "feature_2"]
        mock_joblib.return_value = feature_cols

        output_path = tmp_path / "predictions.csv"

        predict(input_path=str(input_csv), output_path=str(output_path))

        # Verify prediction file was created
        assert output_path.exists()

        df_pred = pd.read_csv(output_path)
        assert "RainTomorrow_pred" in df_pred.columns

    @patch("joblib.load")
    @patch("mlflow.sklearn.load_model")
    def test_predict_with_callback(
        self,
        mock_load_model: MagicMock,
        mock_joblib: MagicMock,
        sample_processed_csv: str,
        tmp_path: Any,
    ) -> None:
        """Test that prediction callback is invoked with progress updates."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0] * 50)
        mock_load_model.return_value = mock_model

        df = pd.read_csv(sample_processed_csv)
        feature_cols = [col for col in df.columns if col not in ["RainTomorrow", "RainToday"]]
        mock_joblib.return_value = feature_cols

        # Track callback calls
        callback_calls = []

        def test_callback(progress: int, message: str) -> None:
            callback_calls.append((progress, message))

        output_path = tmp_path / "predictions.csv"

        predict(
            input_path=sample_processed_csv,
            output_path=str(output_path),
            callback=test_callback,
        )

        # Verify callback was called with progress
        assert len(callback_calls) > 0
        assert any(progress == 100 for progress, _ in callback_calls)

    @patch("joblib.load")
    @patch("mlflow.sklearn.load_model")
    def test_predict_handles_missing_features(
        self,
        mock_load_model: MagicMock,
        mock_joblib: MagicMock,
        tmp_path: Any,
    ) -> None:
        """Test prediction when input data is missing some expected features."""
        # Create CSV with only subset of features
        df = pd.DataFrame({
            "feature_1": np.random.uniform(-1, 1, 20),
            "RainTomorrow": np.random.choice([True, False], 20),
        })
        input_csv = tmp_path / "input_missing_features.csv"
        df.to_csv(input_csv, index=False)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0] * 20)
        mock_load_model.return_value = mock_model

        # Model expects more features than input has
        feature_cols = ["feature_1", "feature_2", "feature_3"]
        mock_joblib.return_value = feature_cols

        output_path = tmp_path / "predictions.csv"

        # Should handle missing features by filling with 0
        predict(input_path=str(input_csv), output_path=str(output_path))

        assert output_path.exists()
