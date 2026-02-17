# src/evaluate.py

import tensorflow as tf
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix, classification_report
from data_loader import load_dataset
from config import TEST_DIR, MODEL_SAVE_PATH, PROJECT_ROOT

print("Loading model and test data...")
model = tf.keras.models.load_model(MODEL_SAVE_PATH)
test_ds, test_size, class_names = load_dataset(TEST_DIR, shuffle=False, augment=False)

print(f"Test samples: {test_size}")
print(f"Classes: {class_names}\n")

loss, acc = model.evaluate(test_ds)
print(f"\n🎯 Test Accuracy: {acc*100:.2f}%")
print(f"Test Loss: {loss:.4f}")

y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_true_classes = np.argmax(y_true, axis=1)  # Convert one-hot to class indices
y_pred = model.predict(test_ds)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))

print("\nClassification Report:")
report = classification_report(y_true_classes, y_pred_classes, target_names=class_names, output_dict=True)
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# Save predictions
results_dir = os.path.join(PROJECT_ROOT, "results")
os.makedirs(results_dir, exist_ok=True)

predictions_file = os.path.join(results_dir, "predictions.json")
with open(predictions_file, 'w') as f:
    json.dump({
        "test_accuracy": float(acc),
        "test_loss": float(loss),
        "class_names": class_names,
        "predictions": y_pred.tolist(),
        "predicted_classes": y_pred_classes.tolist(),
        "true_labels": y_true_classes.tolist(),
        "classification_report": report
    }, f, indent=2)

print(f"\n✅ Predictions saved to: {predictions_file}")