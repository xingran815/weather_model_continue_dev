"""Shared fixtures for pytest tests across the weather prediction project."""
from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sample_raw_dataframe() -> pd.DataFrame:
    """
    Create a sample raw weather DataFrame with realistic structure and missing values.

    Returns:
        DataFrame with columns matching the raw weather dataset structure.
    """
    np.random.seed(42)
    n_rows = 100

    locations = ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"]
    wind_dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n_rows, freq="D"),
        "Location": np.random.choice(locations, n_rows),
        "MinTemp": np.random.uniform(5, 20, n_rows),
        "MaxTemp": np.random.uniform(15, 35, n_rows),
        "Rainfall": np.random.uniform(0, 50, n_rows),
        "Evaporation": np.random.uniform(0, 15, n_rows),
        "Sunshine": np.random.uniform(0, 12, n_rows),
        "WindGustDir": np.random.choice(wind_dirs, n_rows),
        "WindGustSpeed": np.random.uniform(10, 80, n_rows),
        "WindDir9am": np.random.choice(wind_dirs, n_rows),
        "WindDir3pm": np.random.choice(wind_dirs, n_rows),
        "WindSpeed9am": np.random.uniform(5, 40, n_rows),
        "WindSpeed3pm": np.random.uniform(5, 40, n_rows),
        "Humidity9am": np.random.uniform(30, 100, n_rows),
        "Humidity3pm": np.random.uniform(20, 90, n_rows),
        "Pressure9am": np.random.uniform(1000, 1030, n_rows),
        "Pressure3pm": np.random.uniform(1000, 1030, n_rows),
        "Cloud9am": np.random.randint(0, 9, n_rows),
        "Cloud3pm": np.random.randint(0, 9, n_rows),
        "Temp9am": np.random.uniform(10, 25, n_rows),
        "Temp3pm": np.random.uniform(15, 30, n_rows),
        "RainToday": np.random.choice(["Yes", "No"], n_rows),
        "RainTomorrow": np.random.choice(["Yes", "No"], n_rows),
    })

    # Introduce some missing values to simulate real data
    for col in ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine"]:
        mask = np.random.random(n_rows) < 0.05  # 5% missing
        df.loc[mask, col] = np.nan

    return df


@pytest.fixture
def sample_processed_dataframe() -> pd.DataFrame:
    """
    Create a sample preprocessed DataFrame ready for model training.

    Returns:
        DataFrame with normalized numerical features and encoded categorical features.
    """
    np.random.seed(42)
    n_rows = 100
    n_features = 50  # After one-hot encoding of locations and wind directions

    # Create normalized numerical features
    data = np.random.uniform(-1, 1, (n_rows, n_features))
    columns = [f"feature_{i}" for i in range(n_features)]

    df = pd.DataFrame(data, columns=columns)

    # Add encoded categorical features
    df["Location_Sydney"] = np.random.choice([0.0, 1.0], n_rows)
    df["Location_Melbourne"] = np.random.choice([0.0, 1.0], n_rows)
    df["WindDir9am_N"] = np.random.choice([0.0, 1.0], n_rows)
    df["WindDir9am_S"] = np.random.choice([0.0, 1.0], n_rows)

    # Add target variables
    df["RainToday"] = np.random.choice([True, False], n_rows)
    df["RainTomorrow"] = np.random.choice([True, False], n_rows)

    return df


@pytest.fixture
def sample_raw_csv(tmp_path: Any) -> str:
    """
    Create a temporary CSV file with sample raw weather data.

    Args:
        tmp_path: pytest fixture providing temporary directory.

    Returns:
        Path to the temporary CSV file.
    """
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=50, freq="D"),
        "Location": ["Sydney"] * 50,
        "MinTemp": np.random.uniform(10, 20, 50),
        "MaxTemp": np.random.uniform(20, 30, 50),
        "Rainfall": np.random.uniform(0, 10, 50),
        "Evaporation": np.random.uniform(2, 8, 50),
        "Sunshine": np.random.uniform(5, 10, 50),
        "WindGustDir": ["N"] * 50,
        "WindGustSpeed": np.random.uniform(20, 50, 50),
        "WindDir9am": ["N"] * 50,
        "WindDir3pm": ["S"] * 50,
        "WindSpeed9am": np.random.uniform(10, 30, 50),
        "WindSpeed3pm": np.random.uniform(10, 30, 50),
        "Humidity9am": np.random.uniform(50, 80, 50),
        "Humidity3pm": np.random.uniform(40, 70, 50),
        "Pressure9am": np.random.uniform(1010, 1020, 50),
        "Pressure3pm": np.random.uniform(1010, 1020, 50),
        "Cloud9am": np.random.randint(2, 7, 50),
        "Cloud3pm": np.random.randint(2, 7, 50),
        "Temp9am": np.random.uniform(15, 22, 50),
        "Temp3pm": np.random.uniform(18, 28, 50),
        "RainToday": np.random.choice(["Yes", "No"], 50),
        "RainTomorrow": np.random.choice(["Yes", "No"], 50),
    })

    csv_path = tmp_path / "sample_weather.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_processed_csv(tmp_path: Any) -> str:
    """
    Create a temporary CSV file with sample preprocessed data ready for training.

    Args:
        tmp_path: pytest fixture providing temporary directory.

    Returns:
        Path to the temporary preprocessed CSV file.
    """
    np.random.seed(42)
    n_rows = 50

    # Create normalized features
    df = pd.DataFrame({
        "MinTemp": np.random.uniform(-1, 1, n_rows),
        "MaxTemp": np.random.uniform(-1, 1, n_rows),
        "Rainfall": np.random.uniform(-1, 1, n_rows),
        "Humidity9am": np.random.uniform(-1, 1, n_rows),
        "Pressure9am": np.random.uniform(-1, 1, n_rows),
        "Location_Sydney": [1.0] * 25 + [0.0] * 25,
        "Location_Melbourne": [0.0] * 25 + [1.0] * 25,
        "WindDir9am_N": np.random.choice([0.0, 1.0], n_rows),
        "RainToday": np.random.choice([True, False], n_rows),
        "RainTomorrow": np.random.choice([True, False], n_rows),
    })

    csv_path = tmp_path / "sample_preprocessed.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def mock_sqlalchemy_engine() -> Generator[MagicMock, None, None]:
    """
    Create a mock SQLAlchemy engine for testing database operations.

    Yields:
        Mock engine with connect/execute methods configured.
    """
    mock_engine = MagicMock()
    mock_conn = MagicMock()

    # Configure mock connection
    mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

    # Configure execute to return mock results
    mock_result = MagicMock()
    mock_result.scalar.return_value = 1000  # Mock row count
    mock_result.fetchone.return_value = (100,)  # Mock sample size
    mock_conn.execute.return_value = mock_result

    yield mock_engine


@pytest.fixture
def mock_mlflow() -> Generator[MagicMock, None, None]:
    """
    Mock MLflow tracking and model registry operations.

    Yields:
        Patched mlflow module with mocked methods.
    """
    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.get_experiment_by_name") as mock_get_exp, \
         patch("mlflow.create_experiment") as mock_create_exp, \
         patch("mlflow.set_experiment"), \
         patch("mlflow.start_run") as mock_start_run, \
         patch("mlflow.log_params"), \
         patch("mlflow.log_metric"), \
         patch("mlflow.log_artifact"), \
         patch("mlflow.sklearn.log_model") as mock_log_model, \
         patch("mlflow.set_tag"), \
         patch("mlflow.sklearn.load_model") as mock_load_model:

        # Configure experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "test_experiment_id"
        mock_get_exp.return_value = mock_experiment
        mock_create_exp.return_value = "test_experiment_id"

        # Configure run context
        mock_run = MagicMock()
        mock_run.__enter__ = MagicMock(return_value=mock_run)
        mock_run.__exit__ = MagicMock(return_value=False)
        mock_start_run.return_value = mock_run

        # Configure model logging
        mock_model_info = MagicMock()
        mock_model_info.registered_model_version = "1"
        mock_log_model.return_value = mock_model_info

        # Configure model loading
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 1])
        mock_load_model.return_value = mock_model

        yield {
            "get_experiment": mock_get_exp,
            "create_experiment": mock_create_exp,
            "start_run": mock_start_run,
            "log_model": mock_log_model,
            "load_model": mock_load_model,
        }


@pytest.fixture
def mock_mlflow_client() -> Generator[MagicMock, None, None]:
    """
    Mock MLflow tracking client for model registry operations.

    Yields:
        Mock MlflowClient with configured methods.
    """
    with patch("mlflow.tracking.MlflowClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Configure model version retrieval
        mock_version = MagicMock()
        mock_version.run_id = "test_run_id"
        mock_client.get_model_version_by_alias.return_value = mock_version

        # Configure run retrieval
        mock_run = MagicMock()
        mock_run.data.metrics = {"mean_cv_f1": 0.75}
        mock_client.get_run.return_value = mock_run

        yield mock_client


@pytest.fixture
def temp_directories(tmp_path: Any) -> dict[str, str]:
    """
    Create temporary directory structure for testing file operations.

    Args:
        tmp_path: pytest fixture providing temporary directory.

    Returns:
        Dict with paths to temp data/raw, data/processed, and models directories.
    """
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    models_dir = tmp_path / "models"

    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    return {
        "raw": str(raw_dir),
        "processed": str(processed_dir),
        "models": str(models_dir),
        "base": str(tmp_path),
    }


@pytest.fixture
def fastapi_test_client() -> TestClient:
    """
    Create a FastAPI TestClient for testing API endpoints.

    Returns:
        TestClient instance configured for the weather API.
    """
    from src.models.weather_api import api

    return TestClient(api)


@pytest.fixture
def sample_model_args(sample_raw_csv: str, tmp_path: Any) -> dict[str, Any]:
    """
    Create sample model arguments dict for testing pipeline functions.

    Args:
        sample_raw_csv: Path to sample raw CSV fixture.
        tmp_path: pytest fixture providing temporary directory.

    Returns:
        Dict with raw_data_file, processed_data_file, date, sample_percent, duration.
    """
    return {
        "raw_data_file": sample_raw_csv,
        "processed_data_file": None,
        "date": "20240101_1200",
        "sample_percent": 0.2,
        "duration": 10,
    }


@pytest.fixture
def mock_create_engine() -> Generator[MagicMock, None, None]:
    """
    Mock sqlalchemy.create_engine to avoid real database connections in tests.

    Yields:
        Mock create_engine function.
    """
    with patch("sqlalchemy.create_engine") as mock:
        mock_engine = MagicMock()
        mock_conn = MagicMock()

        # Configure connection context manager
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        # Configure execute results
        mock_result = MagicMock()
        mock_result.scalar.return_value = 10000
        mock_result.fetchone.return_value = (2000,)
        mock_conn.execute.return_value = mock_result

        mock.return_value = mock_engine
        yield mock


@pytest.fixture
def mock_pandas_read_sql(sample_raw_dataframe: pd.DataFrame) -> Generator[MagicMock, None, None]:
    """
    Mock pandas.read_sql to return sample data without database connection.

    Args:
        sample_raw_dataframe: Sample DataFrame fixture.

    Yields:
        Mock read_sql function returning the sample DataFrame.
    """
    with patch("pandas.read_sql") as mock:
        mock.return_value = sample_raw_dataframe
        yield mock
