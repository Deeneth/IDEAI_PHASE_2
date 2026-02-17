# src/data_loader.py

import tensorflow as tf
import os
import numpy as np
from config import IMG_SIZE, BATCH_SIZE

def load_dataset(directory, shuffle=True, augment=False):
    # Get all image files including .jfif
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(directory))
    
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.jfif')):
                    image_paths.append(os.path.join(class_dir, img_file))
                    labels.append(label_idx)
    
    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0,1]
        # Convert to one-hot for CategoricalCrossentropy
        label_onehot = tf.one_hot(label, depth=len(class_names))
        return img, label_onehot
    
    # Strong augmentation for training
    def augment_image(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        # Random rotation via transpose
        if tf.random.uniform([]) > 0.5:
            img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label
    
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        ds = ds.shuffle(len(image_paths), reshuffle_each_iteration=True)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds, len(labels), class_names

def compute_class_weights(directory):
    """Calculate class weights for imbalanced dataset"""
    class_counts = {}
    class_names = sorted(os.listdir(directory))
    
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.jfif'))])
            class_counts[class_name] = count
    
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    class_weights = {}
    
    for idx, (class_name, count) in enumerate(sorted(class_counts.items())):
        class_weights[idx] = total / (num_classes * count)
    
    return class_weights
