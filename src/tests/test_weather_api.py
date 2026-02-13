"""Tests for weather_api module: FastAPI endpoints and state management."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.models.weather_api import api, store


@pytest.fixture
def client() -> TestClient:
    """Create FastAPI test client."""
    return TestClient(api)


@pytest.fixture(autouse=True)
def reset_store() -> None:
    """Reset store state before each test."""
    store._model_args = None
    store.set_training_status("inactive", 0, "")
    store.set_predict_status("inactive", 0, "")


class TestWeatherAPI:
    """Tests for FastAPI endpoints."""

    def test_root_endpoint(self, client: TestClient) -> None:
        """Test root endpoint returns welcome message."""
        response = client.get("/")
        assert response.status_code == 200
        assert "greeting" in response.json()

    @patch("src.data.make_dataset.pd.read_sql")
    @patch("src.data.make_dataset.create_engine")
    def test_make_dataset_endpoint(
        self,
        mock_create_engine: MagicMock,
        mock_read_sql: MagicMock,
        client: TestClient,
        sample_raw_dataframe: pd.DataFrame,
        tmp_path: Any,
    ) -> None:
        """Test POST /make_dataset creates dataset and returns metadata."""
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

            response = client.post("/make_dataset", json={"sample_percent": 0.2, "duration": 10})

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "sub-dataset is created."
            assert data["sample_percent"] == 0.2
            assert data["duration"] == 10
            assert "raw_data_file" in data
        finally:
            os.chdir(original_dir)

    def test_preprocessing_without_dataset(self, client: TestClient) -> None:
        """Test POST /preprocessing fails if no dataset exists."""
        response = client.post("/preprocessing")

        assert response.status_code == 400
        assert "No dataset available" in response.json()["detail"]

    @patch("src.data.make_dataset.pd.read_sql")
    @patch("src.data.make_dataset.create_engine")
    def test_preprocessing_endpoint(
        self,
        mock_create_engine: MagicMock,
        mock_read_sql: MagicMock,
        client: TestClient,
        sample_raw_dataframe: pd.DataFrame,
        sample_raw_csv: str,
        tmp_path: Any,
    ) -> None:
        """Test POST /preprocessing after dataset creation."""
        # First create dataset
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

            client.post("/make_dataset", json={"sample_percent": 0.2, "duration": 10})

            # Now test preprocessing
            response = client.post("/preprocessing")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "data is preprocessed."
            assert data["processed_data_file"] is not None
        finally:
            os.chdir(original_dir)

    def test_training_without_preprocessing(self, client: TestClient) -> None:
        """Test POST /training fails if preprocessing not done."""
        response = client.post("/training")

        assert response.status_code == 400
        assert "No dataset available" in response.json()["detail"]

    def test_predict_without_training(self, client: TestClient, sample_raw_csv: str, tmp_path: Any) -> None:
        """Test POST /predict fails if training not completed."""
        # Manually set model_args without completing training
        store.set_model_args(
            {
                "raw_data_file": sample_raw_csv,
                "processed_data_file": sample_raw_csv,
                "date": "20240101",
                "sample_percent": 0.2,
                "duration": 10,
            }
        )

        response = client.post("/predict")

        assert response.status_code == 503
        assert "Training is not finished" in response.json()["detail"]

    def test_training_status_endpoint(self, client: TestClient) -> None:
        """Test GET /training-status returns current status."""
        response = client.get("/training-status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "progress" in data
        assert "message" in data

    def test_predict_status_endpoint(self, client: TestClient) -> None:
        """Test GET /predict-status returns current status."""
        response = client.get("/predict-status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "progress" in data

    @patch("subprocess.run")
    def test_data_versioning_endpoint(self, mock_subprocess: MagicMock, client: TestClient) -> None:
        """Test POST /data-versioning calls DVC."""
        mock_subprocess.return_value = MagicMock()

        response = client.post("/data-versioning", json={"file_path": "data/raw/test.csv"})

        assert response.status_code == 200
        assert "data versioning is completed" in response.json()["status"]
        assert mock_subprocess.called

    def test_concurrent_training_prevented(self, client: TestClient, sample_raw_csv: str) -> None:
        """Test that concurrent training requests are blocked."""
        # Set up model args and mark training as running
        store.set_model_args(
            {
                "raw_data_file": sample_raw_csv,
                "processed_data_file": sample_raw_csv,
                "date": "20240101",
                "sample_percent": 0.2,
                "duration": 10,
            }
        )
        store.set_training_status("running", 50, "Training in progress")

        response = client.post("/training")

        assert response.status_code == 503
        assert "Training is in progress" in response.json()["detail"]
