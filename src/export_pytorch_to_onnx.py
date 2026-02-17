# src/export_pytorch_to_onnx.py (REFERENCE - if using PyTorch)
"""
Export trained PyTorch model to ONNX format
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np

# Load PyTorch model
model = torch.load("models/wafer_defect_model.pth")
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "models/wafer_defect_model.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# Verify
onnx_model = onnx.load("models/wafer_defect_model.onnx")
onnx.checker.check_model(onnx_model)

# Test with ONNX Runtime
ort_session = ort.InferenceSession("models/wafer_defect_model.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
ort_outputs = ort_session.run(None, ort_inputs)

print("✅ PyTorch → ONNX export successful!")
