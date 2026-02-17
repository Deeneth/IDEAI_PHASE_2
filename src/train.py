# src/train.py

import tensorflow as tf
from data_loader import load_dataset, compute_class_weights
from model import build_model
from config import TRAIN_DIR, VAL_DIR, EPOCHS, MODEL_SAVE_PATH
import os

print("Loading datasets...")
train_ds, train_size, class_names = load_dataset(TRAIN_DIR, shuffle=True, augment=True)
val_ds, val_size, _ = load_dataset(VAL_DIR, shuffle=False, augment=False)

print(f"\nDataset Info:")
print(f"Train samples: {train_size}")
print(f"Val samples: {val_size}")
print(f"Classes: {class_names}")

# Compute class weights for imbalanced data
class_weights = compute_class_weights(TRAIN_DIR)
print(f"\nClass weights: {class_weights}")

print("\nBuilding model...")
model = build_model(num_classes=len(class_names))

# Label smoothing reduces overconfidence and improves generalization
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

model.summary()

# Count trainable parameters
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"\n🔥 Trainable parameters: {trainable_params:,}")

# Callbacks for optimal training
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=25,  # More patience
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,  # More patience before reducing LR
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\nStarting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,  # Handle imbalance
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training complete!")
print(f"Best val accuracy: {max(history.history['val_accuracy']):.4f}")