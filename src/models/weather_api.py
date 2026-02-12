#!/usr/bin/env python3
"""FastAPI application for weather forecasting: dataset creation, preprocessing, training, prediction."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException

from src.data.make_dataset import make_dataset
from src.data.preprocessing import preprocessing
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


@dataclass
class curr_status:
    # status can have four states: inactive, running, completed, failed
    status: str = "inactive"
    progress: int = 0
    message: str = ""


training_status = curr_status()
predict_status = curr_status()
model_args: Optional[dict[str, Any]] = None

def update_training_progress(progress: int, message: str) -> None:
    """Update global training status for progress callbacks."""
    training_status.progress = progress
    training_status.message = message


def update_predict_progress(progress: int, message: str) -> None:
    """Update global prediction status for progress callbacks."""
    predict_status.progress = progress
    predict_status.message = message


def wrapper_train_model(training_args: TrainingArgs | dict[str, Any]) -> None:
    """Run training in the background and update training_status."""
    training_status.status = "running"
    training_status.progress = 0
    training_status.message = "Starting training..."
    try:
        training(training_args, callback=update_training_progress)
        training_status.status = "completed"
    except Exception as e:
        training_status.status = "failed"
        training_status.message = str(e)
        raise e


def wrapper_predict(predict_args: dict[str, Any]) -> None:
    """Run prediction in the background and update predict_status."""
    predict_status.status = "running"
    predict_status.progress = 0
    predict_status.message = "Starting prediction..."
    try:
        predict(predict_args['processed_data_file'], callback=update_predict_progress)
        predict_status.status = "completed"
    except Exception as e:
        predict_status.status = "failed"
        predict_status.message = str(e)
        raise e


@api.get("/")
def get_index() -> dict[str, str]:
    """Return a welcome message."""
    return {"greeting": "Welcome to weather forecasting app!"}

@api.get("/make_dataset", name="make sub-dataset from the raw data", responses=responses)
def get_make_dataset(
    sample_percent: Optional[float] = 0.2, duration: Optional[int] = 10
) -> dict[str, Any]:
    """Create a sub-dataset from the raw MySQL data and return paths and metadata."""
    global model_args
    try:
        model_args = make_dataset(sample_percent, duration)
        return {"status": "sub-dataset is created.", **model_args}
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Failed to create sub-dataset: {str(e)}"
        ) from e


@api.get("/preprocessing", name="preprocess the data", responses=responses)
def get_preprocessing() -> dict[str, Any]:
    """Preprocess the raw dataset and set processed_data_file in model_args."""
    try:
        preprocessing(model_args)
        return {"status": "data is preprocessed.", **model_args}
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Failed to preprocess data: {str(e)}"
        ) from e


@api.get("/predict", name="Predict The Weather", responses=responses)
def get_predict(background_tasks: BackgroundTasks) -> dict[str, str]:
    try:
        if training_status.status != "completed":
            raise HTTPException(
                status_code=503,
                detail='Training is not finished, please try to train the model first')
        elif predict_status.status == "running":
            raise HTTPException(
                status_code=503,
                detail='Prediction is in progress, please try again later')
        else:
            background_tasks.add_task(wrapper_predict, model_args)
            return {"status": "prediction started."}
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


@api.get("/training", name="Train The Model with existing data", responses=responses)
def get_training(background_tasks: BackgroundTasks) -> dict[str, str]:
    try:
        if training_status.status == "running":
            raise HTTPException(
                status_code=503,
                detail='Training is in progress, please try again later')
        elif training_status.status == "inactive" or training_status.status == "completed" or training_status.status == "failed":
            background_tasks.add_task(wrapper_train_model, model_args)
            return {"status": "training started"}
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


@api.get("/data-versioning", name="Versioning the data", responses=responses)
def get_data_versioning(file_path: str) -> dict[str, str]:
    """Run DVC add on the given file path for data versioning."""
    try:
        subprocess.run(["dvc", "add", file_path], check=True)
        return {"status": f"data versioning is completed for {file_path}."}
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Failed to version data: {str(e)}"
        ) from e


@api.get("/training-status", name="Get Training Status", responses=responses)
def get_training_status() -> dict[str, str | int]:
    """Return current training job status, progress, and message."""
    return {
        "status": training_status.status,
        "progress": training_status.progress,
        "message": training_status.message,
    }


@api.get("/predict-status", name="Get Predict Status", responses=responses)
def get_predict_status() -> dict[str, str | int]:
    """Return current prediction job status, progress, and message."""
    return {
        "status": predict_status.status,
        "progress": predict_status.progress,
        "message": predict_status.message,
    }
