# src/export_to_onnx.py
"""
Export trained Keras model to ONNX format for cross-platform deployment
"""

import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as ort
import numpy as np
from config import MODEL_SAVE_PATH, PROJECT_ROOT
import os

print("=" * 60)
print("ONNX MODEL EXPORT")
print("=" * 60)

# Paths
ONNX_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "wafer_defect_model.onnx")

# Load trained Keras model
print("\n[1/4] Loading trained Keras model...")
model = tf.keras.models.load_model(MODEL_SAVE_PATH)
print(f"✓ Model loaded from: {MODEL_SAVE_PATH}")

# Convert to ONNX
print("\n[2/4] Converting to ONNX format...")
input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='input')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

# Save ONNX model
print(f"\n[3/4] Saving ONNX model...")
onnx.save(onnx_model, ONNX_MODEL_PATH)
print(f"✓ ONNX model saved to: {ONNX_MODEL_PATH}")

# Get model size
model_size_mb = os.path.getsize(ONNX_MODEL_PATH) / (1024 * 1024)
print(f"✓ Model size: {model_size_mb:.2f} MB")

# Verify ONNX model
print("\n[4/4] Verifying ONNX model...")
onnx_model_check = onnx.load(ONNX_MODEL_PATH)
onnx.checker.check_model(onnx_model_check)
print("✓ ONNX model is valid")

# Test inference with ONNX Runtime
print("\n[VALIDATION] Testing ONNX Runtime inference...")
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)

# Create dummy input
dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)

# Run inference
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
ort_outputs = ort_session.run(None, ort_inputs)

print(f"✓ Input shape: {dummy_input.shape}")
print(f"✓ Output shape: {ort_outputs[0].shape}")
print(f"✓ Output sum: {np.sum(ort_outputs[0]):.4f} (should be ~1.0 for softmax)")

# Compare Keras vs ONNX predictions
keras_output = model.predict(dummy_input, verbose=0)
max_diff = np.max(np.abs(keras_output - ort_outputs[0]))
print(f"✓ Max difference (Keras vs ONNX): {max_diff:.6f}")

if max_diff < 1e-5:
    print("\n✅ ONNX export successful! Model is ready for deployment.")
else:
    print(f"\n⚠️  Warning: Difference between Keras and ONNX is {max_diff:.6f}")
    print("   This may indicate conversion issues. Acceptable if < 1e-3")

print("\n" + "=" * 60)
print("EXPORT COMPLETE")
print("=" * 60)
