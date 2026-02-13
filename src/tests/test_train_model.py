"""Tests for train_model module: model training and MLflow integration."""
from __future__ import annotations

import pytest

from src.models.train_model import training
from src.models.training_args import TrainingArgs


class TestTraining:
    """Tests for training function."""

    def test_training_raises_error_without_processed_data(self) -> None:
        """Test that training raises ValueError if processed_data_file is None."""
        training_args = TrainingArgs(
            raw_data_file="dummy_raw.csv",
            processed_data_file=None,
            date="20240101_1200",
        )

        with pytest.raises(ValueError, match="processed_data_file must be set"):
            training(training_args)

    def test_training_args_accepts_dict(self, sample_processed_csv: str) -> None:
        """Test that TrainingArgs can be created from dict."""
        args_dict = {
            "raw_data_file": "raw.csv",
            "processed_data_file": sample_processed_csv,
            "date": "20240101",
            "sample_percent": 0.3,
            "duration": 5,
        }

        training_args = TrainingArgs(**args_dict)

        assert training_args.sample_percent == 0.3
        assert training_args.duration == 5
        assert training_args.processed_data_file == sample_processed_csv
