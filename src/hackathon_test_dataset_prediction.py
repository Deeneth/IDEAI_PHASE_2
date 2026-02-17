# src/hackathon_test_dataset_prediction.py
"""
Phase-2 Hackathon Submission: ONNX Inference with Confidence Threshold
- Model: wafer_defect_model.onnx (UNCHANGED from Phase-1)
- Inference: ONNX Runtime (Edge Simulation)
- Post-processing: Confidence threshold = 0.3
"""

import onnxruntime as ort
import numpy as np
import os
import sys
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from data_loader import load_dataset
from config import TEST_DIR, PROJECT_ROOT

# ============= CONFIGURATION =============
ONNX_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "wafer_defect_model.onnx")
CONFIDENCE_THRESHOLD = 0.5

# Setup logging to file
results_dir = os.path.join(PROJECT_ROOT, "results")
os.makedirs(results_dir, exist_ok=True)
log_file = os.path.join(results_dir, "phase2_prediction_log.txt")

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)

print("=" * 70)
print("PHASE-2 HACKATHON SUBMISSION: ONNX INFERENCE")
print("=" * 70)
print(f"Test File: hackathon_test_dataset_prediction.py")
print(f"Model: wafer_defect_model.onnx")
print(f"Inference Engine: ONNX Runtime")
print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
print("=" * 70)

# Load ONNX model
print("\n[1/5] Loading ONNX model...")
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
print(f"✓ Model loaded: {ONNX_MODEL_PATH}")
print(f"✓ Model status: UNCHANGED from Phase-1")

# Load test dataset and get dynamic class names
print("\n[2/5] Loading test dataset...")
test_ds, test_size, dataset_classes = load_dataset(TEST_DIR, shuffle=False, augment=False)
num_classes = len(dataset_classes)
print(f"✓ Test samples: {test_size}")
print(f"✓ Dataset classes ({num_classes}): {dataset_classes}")

# Run inference
print("\n[3/5] Running ONNX inference with post-processing...")
input_name = ort_session.get_inputs()[0].name

y_true_list = []
y_pred_list = []
low_conf_count = 0

for images, labels in test_ds:
    predictions = ort_session.run(None, {input_name: images.numpy()})[0]
    
    for pred_probs in predictions:
        max_prob = np.max(pred_probs)
        
        if max_prob < CONFIDENCE_THRESHOLD:
            # Map to 'others' class (last index in dataset_classes)
            others_idx = dataset_classes.index('others') if 'others' in dataset_classes else num_classes - 1
            y_pred_list.append(others_idx)
            low_conf_count += 1
        else:
            y_pred_list.append(np.argmax(pred_probs))
    
    y_true_list.append(np.argmax(labels.numpy(), axis=1))

y_true = np.concatenate(y_true_list)
y_pred = np.array(y_pred_list)

print(f"✓ Inference complete")
print(f"✓ Low-confidence predictions → 'others': {low_conf_count}/{test_size} ({low_conf_count/test_size*100:.1f}%)")

# Compute metrics
print("\n[4/5] Computing evaluation metrics...")
correct = np.sum(y_true == y_pred)
accuracy = correct / len(y_true)

# Macro precision and recall (excluding 'others' if present)
others_idx = dataset_classes.index('others') if 'others' in dataset_classes else -1
mask_valid = y_pred != others_idx if others_idx >= 0 else np.ones(len(y_pred), dtype=bool)
valid_labels = [i for i in range(num_classes) if i != others_idx]

if np.sum(mask_valid) > 0 and len(valid_labels) > 0:
    precision_macro = precision_score(y_true[mask_valid], y_pred[mask_valid], average='macro', labels=valid_labels, zero_division=0)
    recall_macro = recall_score(y_true[mask_valid], y_pred[mask_valid], average='macro', labels=valid_labels, zero_division=0)
else:
    precision_macro = 0.0
    recall_macro = 0.0

print(f"✓ Accuracy: {accuracy*100:.2f}%")
print(f"✓ Precision (macro): {precision_macro*100:.2f}%")
print(f"✓ Recall (macro): {recall_macro*100:.2f}%")
print(f"✓ Correct predictions: {correct}/{len(y_true)}")

# Confusion matrix (dynamic size based on dataset classes)
print("\n[5/5] Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

print(f"\nConfusion Matrix ({num_classes}x{num_classes}):")
print("Classes:", dataset_classes)
print(cm)

# Per-class breakdown
print("\nPer-Class Performance:")
for i, cls in enumerate(dataset_classes):
    mask = y_true == i
    if np.sum(mask) > 0:
        cls_correct = np.sum((y_true == i) & (y_pred == i))
        cls_total = np.sum(mask)
        if others_idx >= 0:
            cls_as_others = np.sum((y_true == i) & (y_pred == others_idx))
        else:
            cls_as_others = 0
        cls_acc = cls_correct / cls_total
        print(f"  {cls:<20} Accuracy: {cls_acc*100:>5.1f}%  Correct: {cls_correct:>3}/{cls_total:<3}  As 'others': {cls_as_others:>3}")

print("\n" + "=" * 70)
print("PHASE-2 SUBMISSION COMPLETE")
print("=" * 70)
print(f"Model: wafer_defect_model.onnx (UNCHANGED)")
print(f"Test samples: {test_size}")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision (macro): {precision_macro*100:.2f}%")
print(f"Recall (macro): {recall_macro*100:.2f}%")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
print("=" * 70)

# Save phase2_metrics.txt
metrics_file = os.path.join(results_dir, "phase2_metrics.txt")
with open(metrics_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("PHASE-2 HACKATHON SUBMISSION METRICS\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Model: wafer_defect_model.onnx\n")
    f.write(f"Test File Name: hackathon_test_dataset_prediction.py\n")
    f.write(f"Test Samples: {test_size}\n")
    f.write(f"Inference: ONNX Runtime (Edge Simulation)\n")
    f.write(f"Post-Processing: Confidence Threshold Applied ({CONFIDENCE_THRESHOLD})\n\n")
    f.write(f"Accuracy: {accuracy*100:.2f}%\n")
    f.write(f"Precision (macro): {precision_macro*100:.2f}%\n")
    f.write(f"Recall (macro): {recall_macro*100:.2f}%\n")
    f.write(f"Correct Predictions: {correct}/{len(y_true)}\n")

print(f"\n✓ Metrics saved: {metrics_file}")

# Save phase2_confusion_matrix.txt
cm_file = os.path.join(results_dir, "phase2_confusion_matrix.txt")
with open(cm_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write(f"CONFUSION MATRIX ({num_classes}x{num_classes})\n")
    f.write("=" * 70 + "\n\n")
    f.write("Classes: " + ", ".join(dataset_classes) + "\n\n")
    f.write("Rows = True Labels, Columns = Predicted Labels\n\n")
    
    # Header
    header = "True \\ Pred"
    f.write(f"{header:<20}")
    for cls in dataset_classes:
        f.write(f"{cls[:8]:<10}")
    f.write("\n" + "-" * (20 + num_classes * 10) + "\n")
    
    # Matrix rows
    for i, cls in enumerate(dataset_classes):
        f.write(f"{cls:<20}")
        for j in range(num_classes):
            f.write(f"{cm[i][j]:<10}")
        f.write("\n")

print(f"✓ Confusion matrix saved: {cm_file}")
print(f"✓ Console log saved: {log_file}")

print("\n" + "=" * 70)
print("ALL PHASE-2 ARTIFACTS GENERATED")
print("=" * 70)
print("Generated files:")
print(f"  1. {log_file}")
print(f"  2. {metrics_file}")
print(f"  3. {cm_file}")
print("\nModel status: UNCHANGED (wafer_defect_model.onnx)")
print("Phase-1 files: UNTOUCHED")
print("=" * 70)
