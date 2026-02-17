#!/bin/bash

echo "======================================"
echo " Wafer Edge AI – Training Pipeline"
echo "======================================"

# Stop script if any command fails
set -e

echo ""
echo "Step 1: Checking Python version"
python --version

echo ""
echo "Step 2: Training the model"
python src/train.py

echo ""
echo "Step 3: Evaluating the model"
python src/evaluate.py

echo ""
echo "Step 4 (Optional): Converting to TFLite for edge deployment"
python src/convert_to_tflite.py

echo ""
echo "======================================"
echo " Pipeline completed successfully ✅"
echo "======================================"
