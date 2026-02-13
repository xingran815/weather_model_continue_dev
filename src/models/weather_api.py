#!/usr/bin/env python3
"""FastAPI application for weather forecasting: dataset creation, preprocessing, training, prediction."""
from __future__ import annotations

import logging
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.data.make_dataset import make_dataset
from src.data.preprocessing import preprocessing
from src.logging_config import clear_correlation_id, configure_logging, set_correlation_id
from src.models.predict_model import predict
from src.models.train_model import training
from src.models.training_args import TrainingArgs

responses = {
    200: {"description": "OK"},
    404: {"description": "Item not found"},
    302: {"description": "The item was moved"},
    403: {"description": "Not enough privileges"},
}

api = FastAPI(
    title="API for weather forecasting",
    description="""
    This is a weather forecasting API controlling
    the training and predicting processes.
    """,
    version="0.1.0",
)

configure_logging()
logger = logging.getLogger(__name__)


@api.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Attach correlation ID and emit request lifecycle logs."""
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    set_correlation_id(correlation_id)
    start_time = time.perf_counter()
    logger.info(
        "Request started",
        extra={"method": request.method, "path": request.url.path},
    )
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Request failed",
            extra={"method": request.method, "path": request.url.path},
        )
        raise
    finally:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "Request finished",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_ms": elapsed_ms,
            },
        )
        clear_correlation_id()
    response.headers["X-Correlation-ID"] = correlation_id
    return response


@dataclass
class JobStatus:
    """Status for a long-running job (training or prediction)."""
    status: str = "inactive"  # inactive | running | completed | failed
    progress: int = 0
    message: str = ""


class PipelineStore:
    """Thread-safe store for model_args, training_status, and predict_status."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model_args: dict[str, Any] | None = None
        self._training_status = JobStatus()
        self._predict_status = JobStatus()

    @property
    def model_args(self) -> dict[str, Any] | None:
        with self._lock:
            return self._model_args

    def set_model_args(self, value: dict[str, Any]) -> None:
        with self._lock:
            self._model_args = value

    def get_training_status(self) -> JobStatus:
        with self._lock:
            return JobStatus(
                status=self._training_status.status,
                progress=self._training_status.progress,
                message=self._training_status.message,
            )

    def set_training_status(self, status: str, progress: int = 0, message: str = "") -> None:
        with self._lock:
            self._training_status.status = status
            self._training_status.progress = progress
            self._training_status.message = message

    def get_predict_status(self) -> JobStatus:
        with self._lock:
            return JobStatus(
                status=self._predict_status.status,
                progress=self._predict_status.progress,
                message=self._predict_status.message,
            )

    def set_predict_status(self, status: str, progress: int = 0, message: str = "") -> None:
        with self._lock:
            self._predict_status.status = status
            self._predict_status.progress = progress
            self._predict_status.message = message


store = PipelineStore()


# --- Pydantic request/response models ---


class MakeDatasetRequest(BaseModel):
    """Request body for POST /make_dataset."""
    sample_percent: float = Field(default=0.2, ge=0.0, le=1.0, description="Fraction of rows to sample")
    duration: int = Field(default=10, ge=1, le=20, description="Number of years of data from 2008-01-01")


class MakeDatasetResponse(BaseModel):
    """Response for POST /make_dataset."""
    status: str
    raw_data_file: str
    processed_data_file: str | None
    date: str
    sample_percent: float
    duration: int


class PreprocessingResponse(BaseModel):
    """Response for POST /preprocessing."""
    status: str
    raw_data_file: str
    processed_data_file: str | None
    date: str
    sample_percent: float
    duration: int


class TrainingResponse(BaseModel):
    """Response for POST /training."""
    status: str


class PredictResponse(BaseModel):
    """Response for POST /predict."""
    status: str


class DataVersioningRequest(BaseModel):
    """Request body for POST /data-versioning."""
    file_path: str = Field(..., description="Path to the file to version with DVC")


class DataVersioningResponse(BaseModel):
    """Response for POST /data-versioning."""
    status: str


class JobStatusResponse(BaseModel):
    """Response for GET /training-status and GET /predict-status."""
    status: str
    progress: int
    message: str


def update_training_progress(progress: int, message: str) -> None:
    """Update training status in the store (used as callback)."""
    store.set_training_status(store.get_training_status().status, progress=progress, message=message)


def update_predict_progress(progress: int, message: str) -> None:
    """Update prediction status in the store (used as callback)."""
    store.set_predict_status(store.get_predict_status().status, progress=progress, message=message)


def wrapper_train_model(training_args: TrainingArgs | dict[str, Any]) -> None:
    """Run training in the background and update training status in the store."""
    store.set_training_status("running", progress=0, message="Starting training...")
    try:
        training(training_args, callback=update_training_progress)
        store.set_training_status("completed")
    except Exception as e:
        logger.exception("Training task failed")
        store.set_training_status("failed", message=str(e))
        raise e


def wrapper_predict(predict_args: dict[str, Any]) -> None:
    """Run prediction in the background and update predict status in the store."""
    store.set_predict_status("running", progress=0, message="Starting prediction...")
    try:
        predict(predict_args["processed_data_file"], callback=update_predict_progress)
        store.set_predict_status("completed")
    except Exception as e:
        logger.exception("Prediction task failed")
        store.set_predict_status("failed", message=str(e))
        raise e


# --- Endpoints ---


@api.get("/")
def get_index() -> dict[str, str]:
    """Return a welcome message."""
    return {"greeting": "Welcome to weather forecasting app!"}


@api.post(
    "/make_dataset",
    name="make sub-dataset from the raw data",
    responses=responses,
    response_model=MakeDatasetResponse,
)
def post_make_dataset(body: MakeDatasetRequest) -> MakeDatasetResponse:
    """Create a sub-dataset from the raw MySQL data and return paths and metadata."""
    try:
        result = make_dataset(body.sample_percent, body.duration)
        store.set_model_args(result)
        return MakeDatasetResponse(status="sub-dataset is created.", **result)
    except Exception as e:
        logger.exception("Failed to create sub-dataset")
        raise HTTPException(
            status_code=503, detail=f"Failed to create sub-dataset: {str(e)}"
        ) from e


@api.post(
    "/preprocessing",
    name="preprocess the data",
    responses=responses,
    response_model=PreprocessingResponse,
)
def post_preprocessing() -> PreprocessingResponse:
    """Preprocess the raw dataset and set processed_data_file in stored model_args."""
    model_args = store.model_args
    if model_args is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset available. Call POST /make_dataset first.",
        )
    try:
        preprocessing(model_args)
        store.set_model_args(model_args)
        return PreprocessingResponse(status="data is preprocessed.", **model_args)
    except Exception as e:
        logger.exception("Failed to preprocess data")
        raise HTTPException(
            status_code=503, detail=f"Failed to preprocess data: {str(e)}"
        ) from e


@api.post(
    "/predict",
    name="Predict The Weather",
    responses=responses,
    response_model=PredictResponse,
)
def post_predict(background_tasks: BackgroundTasks) -> PredictResponse:
    """Start prediction in the background using the last preprocessed dataset."""
    training_status = store.get_training_status()
    predict_status = store.get_predict_status()
    model_args = store.model_args
    if training_status.status != "completed":
        raise HTTPException(
            status_code=503,
            detail="Training is not finished, please try to train the model first",
        )
    if predict_status.status == "running":
        raise HTTPException(
            status_code=503,
            detail="Prediction is in progress, please try again later",
        )
    if model_args is None:
        raise HTTPException(
            status_code=400,
            detail="No model args available. Run make_dataset and preprocessing first.",
        )
    try:
        background_tasks.add_task(wrapper_predict, model_args)
        return PredictResponse(status="prediction started.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to start prediction")
        raise HTTPException(status_code=503, detail=str(e)) from e


@api.post(
    "/training",
    name="Train The Model with existing data",
    responses=responses,
    response_model=TrainingResponse,
)
def post_training(background_tasks: BackgroundTasks) -> TrainingResponse:
    """Start model training in the background using the last preprocessed dataset."""
    training_status = store.get_training_status()
    model_args = store.model_args
    if training_status.status == "running":
        raise HTTPException(
            status_code=503,
            detail="Training is in progress, please try again later",
        )
    if model_args is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset available. Call POST /make_dataset and POST /preprocessing first.",
        )
    try:
        background_tasks.add_task(wrapper_train_model, model_args)
        return TrainingResponse(status="training started")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to start training")
        raise HTTPException(status_code=503, detail=str(e)) from e


@api.post(
    "/data-versioning",
    name="Versioning the data",
    responses=responses,
    response_model=DataVersioningResponse,
)
def post_data_versioning(body: DataVersioningRequest) -> DataVersioningResponse:
    """Run DVC add on the given file path for data versioning."""
    try:
        subprocess.run(["dvc", "add", body.file_path], check=True)
        return DataVersioningResponse(status=f"data versioning is completed for {body.file_path}.")
    except Exception as e:
        logger.exception("Failed to version data")
        raise HTTPException(
            status_code=503, detail=f"Failed to version data: {str(e)}"
        ) from e


@api.get(
    "/training-status",
    name="Get Training Status",
    responses=responses,
    response_model=JobStatusResponse,
)
def get_training_status() -> JobStatusResponse:
    """Return current training job status, progress, and message."""
    s = store.get_training_status()
    return JobStatusResponse(status=s.status, progress=s.progress, message=s.message)


@api.get(
    "/predict-status",
    name="Get Predict Status",
    responses=responses,
    response_model=JobStatusResponse,
)
def get_predict_status() -> JobStatusResponse:
    """Return current prediction job status, progress, and message."""
    s = store.get_predict_status()
    return JobStatusResponse(status=s.status, progress=s.progress, message=s.message)
