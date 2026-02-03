#!/bin/bash
echo '--- STARTING MLOPS PIPELINE ---'
echo 'Step 1: Cleaning Data...'
python3 src/data_cleaning.py
echo 'Step 2: Training Model...'
python3 src/train_model.py
echo 'Step 3: Generating Predictions...'
python3 src/predict_sales.py
echo '--- PIPELINE COMPLETE ---'
