# src/inference_tflite.py

import tensorflow as tf
import numpy as np
import json
import os
from data_loader import load_dataset
from config import TEST_DIR, TFLITE_SAVE_PATH, PROJECT_ROOT

interpreter = tf.lite.Interpreter(model_path=TFLITE_SAVE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test_ds = load_dataset(TEST_DIR, shuffle=False)

all_predictions = []
for image, label in test_ds:
    for i in range(image.shape[0]):
        interpreter.set_tensor(input_details[0]['index'], image[i:i+1])
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        all_predictions.append({
            "prediction_probability": float(prediction[0][0]),
            "actual_label": int(label.numpy()[i])
        })
        
        print(f"Image {len(all_predictions)}: Prediction={prediction[0][0]:.4f}, Actual={label.numpy()[i]}")

# Save predictions
results_dir = os.path.join(PROJECT_ROOT, "results")
os.makedirs(results_dir, exist_ok=True)

tflite_predictions_file = os.path.join(results_dir, "tflite_predictions.json")
with open(tflite_predictions_file, 'w') as f:
    json.dump(all_predictions, f, indent=2)

print(f"\n✅ TFLite predictions saved to: {tflite_predictions_file}")
