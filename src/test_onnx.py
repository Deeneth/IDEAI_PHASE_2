# src/test_onnx.py
"""
Test ONNX model with a sample image
"""

import onnxruntime as ort
import numpy as np
import cv2
import os
from config import PROJECT_ROOT

# Paths
ONNX_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "wafer_defect_model.onnx")

# Auto-detect first available test image
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")  # Changed to train
TEST_IMAGE_PATH = None
for class_name in os.listdir(TEST_DIR):
    class_dir = os.path.join(TEST_DIR, class_name)
    if os.path.isdir(class_dir):
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif'))]
        if images:
            TEST_IMAGE_PATH = os.path.join(class_dir, images[0])
            break

# Class names
CLASS_NAMES = ['bridge', 'cmp_defect', 'cracks', 'ler', 'opens', 'pattern_collapse', 'undefected', 'via_defect']

print("=" * 60)
print("TESTING ONNX MODEL")
print("=" * 60)

# Load ONNX model
print("\n[1/4] Loading ONNX model...")
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
print(f"✓ Model loaded: {ONNX_MODEL_PATH}")

# Load and preprocess image
print("\n[2/4] Loading test image...")
if not os.path.exists(TEST_IMAGE_PATH):
    print(f"⚠️  Image not found: {TEST_IMAGE_PATH}")
    print("Please update TEST_IMAGE_PATH in the script")
    exit(1)

img = cv2.imread(TEST_IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (224, 224))
img_normalized = img_resized.astype(np.float32) / 255.0
img_batch = np.expand_dims(img_normalized, axis=0)

print(f"✓ Image loaded: {TEST_IMAGE_PATH}")
print(f"✓ Image shape: {img_batch.shape}")

# Run inference
print("\n[3/4] Running ONNX inference...")
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

predictions = ort_session.run([output_name], {input_name: img_batch})[0]

# Get results
predicted_class_idx = np.argmax(predictions[0])
predicted_class = CLASS_NAMES[predicted_class_idx]
confidence = predictions[0][predicted_class_idx] * 100

print(f"✓ Inference complete")

# Display results
print("\n[4/4] Results:")
print("=" * 60)
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
print("\nAll Class Probabilities:")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  {class_name:<20} {predictions[0][i]*100:>6.2f}%")

print("\n" + "=" * 60)
print("✅ ONNX MODEL TEST COMPLETE")
print("=" * 60)
