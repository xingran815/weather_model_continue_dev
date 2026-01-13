#!/bin/bash

set -e

export PATH=/usr/bin:/bin
export MODEL_URI=${MODEL_URI:-http://model:8000}

LOG="/app/data/cron.log" 
mkdir -p /app/data 

sleep 60

echo "Start cron $(date)" >> "$LOG"

# Trigger the dataset creation, preprocessing, and training via API
echo "/usr/bin/curl -s ${MODEL_URI}/make_dataset" >> "$LOG"
/usr/bin/curl -s ${MODEL_URI}/make_dataset >> "$LOG"

echo "/usr/bin/curl -s ${MODEL_URI}/preprocessing" >> "$LOG"
/usr/bin/curl -s ${MODEL_URI}/preprocessing >> "$LOG"

echo "/usr/bin/curl -s ${MODEL_URI}/training" >> "$LOG"
/usr/bin/curl -s ${MODEL_URI}/training >> "$LOG"

echo "End cron $(date)" >> "$LOG"
