# src/convert_to_tflite.py

import tensorflow as tf
import os
from config import MODEL_SAVE_PATH, TFLITE_SAVE_PATH

# Create tflite directory if it doesn't exist
os.makedirs(os.path.dirname(TFLITE_SAVE_PATH), exist_ok=True)

# Load Keras model
model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(TFLITE_SAVE_PATH, "wb") as f:
    f.write(tflite_model)

print("✅ INT8 TFLite model saved.")
