"""
============================================================
 Crop Disease Detection System
 By Musinguzi Dickson, Muhairwe Dominic, Kalyegira Emmanuel, Emelda Nakacwa
 File: train_model.py
 Description: This trains a CNN (MobileNetV2 transfer learning)
              on the PlantVillageDataset to classify crop
              diseases from leaf images.
============================================================
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ─── CONFIGURATION ──────────────────────────────────────────
IMG_SIZE    = (224, 224)   # MobileNetV2 expected input size
BATCH_SIZE  = 32
EPOCHS      = 20
DATASET_DIR = "PlantVillageDataset/color"   # Root dir after extracting dataset
MODEL_PATH  = "models/crop_disease_model.h5"
LABELS_PATH = "models/class_labels.json"

# ─── DATA AUGMENTATION ─────────────────────────────────────
# Augmentation prevents overfitting on training images by
# artificially expanding the dataset with random transforms.
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,          # Normalize pixel values to [0, 1]
    rotation_range=30,           # Random rotations up to 30°
    width_shift_range=0.2,       # Horizontal shift
    height_shift_range=0.2,      # Vertical shift
    shear_range=0.2,             # Shear transformation
    zoom_range=0.2,              # Random zoom
    horizontal_flip=True,        # Mirror flip
    fill_mode="nearest",         # Fill empty pixels
    validation_split=0.2         # Reserve 20% for validation
)

# Validation images are ONLY rescaled — no augmentation
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)


def load_data():
    """Load train and validation datasets from directory."""
    print("[INFO] Loading training data...")
    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    print("[INFO] Loading validation data...")
    val_gen = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    # Save class label mapping for use during inference
    os.makedirs("models", exist_ok=True)
    with open(LABELS_PATH, "w") as f:
        # Invert dict: index → class name
        labels = {v: k for k, v in train_gen.class_indices.items()}
        json.dump(labels, f, indent=2)
    print(f"[INFO] Saved {len(labels)} class labels to {LABELS_PATH}")

    return train_gen, val_gen, len(train_gen.class_indices)


def build_model(num_classes):
    """
    Build transfer-learning model using MobileNetV2 as base.

    MobileNetV2 was pre-trained on ImageNet (1.4M images).
    We freeze its convolutional layers and replace the
    classification head with our own for crop disease classes.

    Architecture:
        MobileNetV2 (frozen) → GlobalAveragePooling → Dense(128)
        → Dropout(0.3) → Dense(num_classes, softmax)
    """
    print("[INFO] Building MobileNetV2 transfer learning model...")

    # Load MobileNetV2 pre-trained on ImageNet, remove top classifier
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,          # Remove ImageNet classifier
        weights="imagenet"          # Use pre-trained weights
    )

    # Freeze all base layers — we only train our custom head
    base_model.trainable = False

    # Build custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),   # Reduce spatial dims to 1D
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),               # Regularize to reduce overfitting
        layers.Dense(num_classes, activation="softmax")  # Output probabilities
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


def train(model, train_gen, val_gen):
    """Train the model with early stopping and LR reduction."""
    print("[INFO] Starting training...")

    callbacks = [
        # Save best model based on validation accuracy
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        # Stop training if val_loss doesn't improve for 5 epochs
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce LR when validation loss plateaus
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    return history


def plot_training(history):
    """Plot and save training accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Model Accuracy over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    # Loss plot
    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_curves.png", dpi=150)
    print("[INFO] Training curves saved to results/training_curves.png")
    plt.show()


if __name__ == "__main__":
    train_gen, val_gen, num_classes = load_data()
    model = build_model(num_classes)
    history = train(model, train_gen, val_gen)
    plot_training(history)
    print(f"\n[DONE] Model saved to: {MODEL_PATH}")
    print(f"[DONE] Class labels saved to: {LABELS_PATH}")
