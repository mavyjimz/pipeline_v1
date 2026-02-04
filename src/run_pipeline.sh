#!/bin/bash
echo "--- STARTING MLOPS PIPELINE (GHOST-PROOF VERSION) ---"

# Step 1: Cleaning Raw Data
echo "Step 1: Cleaning Data..."
python3 src/clean_data.py

# Step 2: NEW - Feature Engineering (This breaks the Underfitting loop!)
echo "Step 2: Engineering New Features..."
python3 src/feature_engineering.py

# Step 3: Training Model (Now using engineered_sales.csv)
echo "Step 3: Training Model..."
python3 src/train_model.py

# Step 4: Generating Predictions
echo "Step 4: Generating Predictions..."
python3 src/predict_sales.py

echo "--- PIPELINE COMPLETE - CHECK D:/MLOps/.../shared_output ---"