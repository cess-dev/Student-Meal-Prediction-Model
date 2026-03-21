#!/bin/bash
echo "Starting API server..."
exec uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info

echo "Checking for model artefacts..."
if [ ! -f "/app/models/meal_model.pkl" ]; then
    echo "No model found — training now..."
    python train.py
    echo "Training complete."
else
    echo "Model found — skipping training."
fi

echo "API is ready."
wait