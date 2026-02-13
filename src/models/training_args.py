"""Pydantic model for training pipeline arguments."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrainingArgs(BaseModel):
    """
    Arguments for the training and data pipeline (make_dataset, preprocessing, training).

    Attributes:
        raw_data_file: Path to the raw CSV produced by make_dataset.
        processed_data_file: Path to the preprocessed CSV; set by preprocessing, required for training.
        date: Date string used for output filenames (e.g. YYYYMMDD_HHMM).
        sample_percent: Fraction of data sampled in make_dataset (e.g. 0.2 for 20%).
        duration: Number of years of data used in make_dataset.
    """

    raw_data_file: str
    processed_data_file: str | None = Field(default=None, description="Set by preprocessing")
    date: str = ""
    sample_percent: float = 0.2
    duration: int = 10
