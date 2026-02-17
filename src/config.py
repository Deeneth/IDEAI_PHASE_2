# src/config.py

import os

IMG_SIZE = 224
BATCH_SIZE = 16  # Increased for more stable gradients
EPOCHS = 1500  # High limit, early stopping will terminate earlier

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "wafer_defect_model.keras")
TFLITE_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "tflite", "wafer_defect_model_int8.tflite")
