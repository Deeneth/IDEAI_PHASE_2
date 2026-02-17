import tensorflow as tf

def build_model(num_classes=8):
    """Lightweight model: frozen MobileNetV3Small + minimal head"""
    
    # Frozen feature extractor
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=False  # We normalize in data loader
    )
    base_model.trainable = False  # FROZEN - best for small datasets
    
    # Ultra-lightweight classification head
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Reduces params vs Dense
    x = tf.keras.layers.BatchNormalization()(x)  # Stabilizes training
    x = tf.keras.layers.Dropout(0.4)(x)  # Reduced dropout
    x = tf.keras.layers.Dense(64, activation="relu")(x)  # Small hidden layer
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model
