#!/bin/bash

# This script runs the pipeline for weather data processing and model training every ten minutes.

python3 /data/make_dataset.py
python3 preprocessing.py
python3 train_model.py
python3 predict_model.py
