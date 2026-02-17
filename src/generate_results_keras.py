# src/generate_results_keras.py
"""
Generate comprehensive evaluation results using Keras model (no ONNX required)
"""

import tensorflow as tf
import numpy as np
import os
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from data_loader import load_dataset
from config import TEST_DIR, MODEL_SAVE_PATH, PROJECT_ROOT

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("GENERATING HACKATHON SUBMISSION RESULTS")
print("=" * 70)

print("\n[1/6] Loading test dataset...")
test_ds, test_size, class_names = load_dataset(TEST_DIR, shuffle=False, augment=False)
print(f"✓ Test samples: {test_size}")
print(f"✓ Classes: {class_names}")

print("\n[2/6] Loading Keras model...")
model = tf.keras.models.load_model(MODEL_SAVE_PATH)
print(f"✓ Model loaded: {MODEL_SAVE_PATH}")

model_size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
print(f"✓ Model size: {model_size_mb:.2f} MB")

print("\n[3/6] Running inference on test set...")
y_true_list = []
y_pred_list = []
inference_times = []

for images, labels in test_ds:
    start_time = time.time()
    predictions = model.predict(images, verbose=0)
    inference_times.append(time.time() - start_time)
    
    y_true_list.append(np.argmax(labels.numpy(), axis=1))
    y_pred_list.append(np.argmax(predictions, axis=1))

y_true = np.concatenate(y_true_list)
y_pred = np.concatenate(y_pred_list)

avg_inference_time = np.mean(inference_times) * 1000
print(f"✓ Inference complete")
print(f"✓ Average batch inference time: {avg_inference_time:.2f} ms")

print("\n[4/6] Computing evaluation metrics...")
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print(f"✓ Accuracy:  {accuracy*100:.2f}%")
print(f"✓ Precision: {precision*100:.2f}%")
print(f"✓ Recall:    {recall*100:.2f}%")
print(f"✓ F1-Score:  {f1*100:.2f}%")

print("\n[5/6] Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Wafer Defect Classification', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✓ Confusion matrix saved: {cm_path}")
plt.close()

print("\n[6/6] Saving detailed results...")
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
with open(metrics_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("IESA DEEPTECH HACKATHON - MODEL EVALUATION RESULTS\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("MODEL INFORMATION\n")
    f.write("-" * 70 + "\n")
    f.write(f"Model Format:        Keras\n")
    f.write(f"Model Size:          {model_size_mb:.2f} MB\n")
    f.write(f"Architecture:        MobileNetV3Small + Custom Head\n")
    f.write(f"Input Shape:         (224, 224, 3)\n")
    f.write(f"Number of Classes:   {len(class_names)}\n\n")
    
    f.write("DATASET INFORMATION\n")
    f.write("-" * 70 + "\n")
    f.write(f"Test Samples:        {test_size}\n")
    f.write(f"Classes:             {', '.join(class_names)}\n\n")
    
    f.write("PERFORMANCE METRICS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Overall Accuracy:    {accuracy*100:.2f}%\n")
    f.write(f"Weighted Precision:  {precision*100:.2f}%\n")
    f.write(f"Weighted Recall:     {recall*100:.2f}%\n")
    f.write(f"Weighted F1-Score:   {f1*100:.2f}%\n\n")
    
    f.write("INFERENCE PERFORMANCE\n")
    f.write("-" * 70 + "\n")
    f.write(f"Avg Batch Time:      {avg_inference_time:.2f} ms\n")
    f.write(f"Throughput:          ~{1000/avg_inference_time:.1f} batches/sec\n\n")
    
    f.write("PER-CLASS PERFORMANCE\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
    f.write("-" * 70 + "\n")
    for class_name in class_names:
        p = report[class_name]['precision']
        r = report[class_name]['recall']
        f1_class = report[class_name]['f1-score']
        f.write(f"{class_name:<20} {p*100:>6.2f}%      {r*100:>6.2f}%      {f1_class*100:>6.2f}%\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("INTERPRETATION\n")
    f.write("=" * 70 + "\n")
    f.write(f"The model achieves {accuracy*100:.1f}% accuracy on the test set, demonstrating\n")
    f.write("effective defect classification for semiconductor wafer inspection.\n")
    f.write(f"The lightweight model ({model_size_mb:.1f}MB) is optimized for edge deployment with\n")
    f.write(f"fast inference ({avg_inference_time:.1f}ms per batch).\n")

print(f"✓ Metrics saved: {metrics_path}")

results_json = {
    "model_info": {
        "format": "Keras",
        "size_mb": round(model_size_mb, 2),
        "architecture": "MobileNetV3Small",
        "input_shape": [224, 224, 3],
        "num_classes": len(class_names)
    },
    "dataset": {
        "test_samples": int(test_size),
        "classes": class_names
    },
    "metrics": {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4)
    },
    "inference": {
        "avg_batch_time_ms": round(float(avg_inference_time), 2)
    },
    "per_class_metrics": report
}

json_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"✓ JSON results saved: {json_path}")

print("\n" + "=" * 70)
print("✅ ALL RESULTS GENERATED SUCCESSFULLY")
print("=" * 70)
print(f"\nResults saved in: {RESULTS_DIR}/")
print("  - metrics.txt")
print("  - confusion_matrix.png")
print("  - evaluation_results.json")
