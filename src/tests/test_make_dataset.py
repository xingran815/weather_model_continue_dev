"""Tests for make_dataset module: database sampling and CSV export."""
from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.make_dataset import make_dataset


class TestMakeDataset:
    """Tests for make_dataset function."""

    @patch("src.data.make_dataset.pd.read_sql")
    @patch("src.data.make_dataset.create_engine")
    def test_make_dataset_returns_correct_structure(
        self,
        mock_create_engine: MagicMock,
        mock_read_sql: MagicMock,
        sample_raw_dataframe: pd.DataFrame,
        tmp_path: Any,
    ) -> None:
        """Test that make_dataset returns dict with correct keys and creates CSV file."""
        # Setup mocks
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_result = MagicMock()
        mock_result.scalar.return_value = 10000
        mock_result.fetchone.return_value = (2000,)
        mock_conn.execute.return_value = mock_result
        mock_read_sql.return_value = sample_raw_dataframe

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = make_dataset(sample_percent=0.2, duration=10)

            # Check return structure
            assert all(key in result for key in ["raw_data_file", "processed_data_file", "date", "sample_percent", "duration"])
            assert result["processed_data_file"] is None
            assert result["sample_percent"] == 0.2
            assert result["duration"] == 10

            # Check file was created
            assert os.path.exists(result["raw_data_file"])
            assert "weather_subset_" in result["raw_data_file"]

            # Verify CSV can be read
            df = pd.read_csv(result["raw_data_file"])
            assert len(df) > 0
        finally:
            os.chdir(original_dir)
