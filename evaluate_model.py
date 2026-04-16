"""
============================================================
 Crop Disease Detection System
 By Musinguzi Dickson, Muhairwe Dominic, Kalyegira Emmanuel, Emelda Nakacwa
 File: evaluate_model.py
 Description: This evaluates the trained model. Generates classification
            report, confusion matrix, and identifies failure
              cases for Milestone 3 (DATA, MODEL EXPLANATION, EVALUATION AND ETHICAL ANALYSIS)
============================================================
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── CONFIGURATION ──────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
DATASET_DIR = "dataset/plantvillage"
MODEL_PATH  = "models/crop_disease_model.h5"
LABELS_PATH = "models/class_labels.json"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_test_data():
    """Load test split of dataset (no augmentation)."""
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )
    test_gen = test_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False  # Keep order for evaluation
    )
    return test_gen


def evaluate(model, test_gen, labels):
    """
    Evaluate model and print full classification metrics.
    Saves confusion matrix and reports for Milestone 3.
    """
    print("[INFO] Running evaluation on validation set...")

    # Get true labels and predictions
    y_true = test_gen.classes
    y_pred_proba = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # ── Core Metrics ──
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Overall Accuracy  : {acc * 100:.2f}%")
    print(f"  Precision (W.Avg) : {prec * 100:.2f}%")
    print(f"  Recall (W.Avg)    : {rec * 100:.2f}%")
    print(f"  F1 Score (W.Avg)  : {f1 * 100:.2f}%")
    print("=" * 60)

    # ── Per-Class Report ──
    class_names = [labels[str(i)] for i in range(len(labels))]
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n[INFO] Per-Class Classification Report:")
    print(report)

    # Save report to file
    with open(f"{RESULTS_DIR}/classification_report.txt", "w") as f:
        f.write("CROP DISEASE DETECTION — EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy  : {acc * 100:.2f}%\n")
        f.write(f"Precision : {prec * 100:.2f}%\n")
        f.write(f"Recall    : {rec * 100:.2f}%\n")
        f.write(f"F1 Score  : {f1 * 100:.2f}%\n\n")
        f.write("Per-Class Report:\n")
        f.write(report)
    print(f"[INFO] Report saved to {RESULTS_DIR}/classification_report.txt")

    return y_true, y_pred, y_pred_proba, class_names


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot and save the confusion matrix heatmap."""
    print("[INFO] Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)

    # For large number of classes, limit to top-N for readability
    n = min(len(class_names), 20)
    cm_subset = cm[:n, :n]
    names_subset = class_names[:n]

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm_subset,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=names_subset,
        yticklabels=names_subset,
        linewidths=0.5
    )
    plt.title("Confusion Matrix — Crop Disease Classifier", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=150)
    print(f"[INFO] Confusion matrix saved to {RESULTS_DIR}/confusion_matrix.png")
    plt.show()


def find_failure_cases(y_true, y_pred, y_pred_proba, class_names, top_n=10):
    """
    Identify top failure cases — misclassified examples with
    highest model confidence. Required for Milestone 3.
    """
    print("\n[INFO] Identifying failure cases...")

    failures = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            confidence = y_pred_proba[i][pred]
            failures.append({
                "index": i,
                "true_class": class_names[true],
                "predicted_class": class_names[pred],
                "confidence": float(confidence)
            })

    # Sort by confidence (high confidence, wrong answer = worst failures)
    failures.sort(key=lambda x: x["confidence"], reverse=True)

    print(f"\n  TOP {top_n} FAILURE CASES (Milestone 3):")
    print("  " + "-" * 70)
    print(f"  {'#':<4} {'True Label':<30} {'Predicted':<30} {'Confidence'}")
    print("  " + "-" * 70)
    for i, f in enumerate(failures[:top_n], 1):
        print(f"  {i:<4} {f['true_class']:<30} {f['predicted_class']:<30} {f['confidence']*100:.1f}%")

    # Save failure cases
    with open(f"{RESULTS_DIR}/failure_cases.json", "w") as fp:
        json.dump(failures[:top_n], fp, indent=2)
    print(f"\n[INFO] Failure cases saved to {RESULTS_DIR}/failure_cases.json")

    # Analysis of why failures occur (for report)
    print("\n[ANALYSIS] Common Causes of Failure:")
    print("  1. Similar visual symptoms between different diseases")
    print("  2. Poor image quality (blur, wrong angle, bad lighting)")
    print("  3. Early-stage disease before clear symptoms appear")
    print("  4. Multiple diseases on the same leaf")
    print("  5. Ugandan crop varieties not in PlantVillageDataset")

    return failures


if __name__ == "__main__":
    # Load trained model and labels
    print("[INFO] Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    # Load test data
    test_gen = load_test_data()

    # Run evaluation
    y_true, y_pred, y_pred_proba, class_names = evaluate(model, test_gen, labels)

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)

    # Find and report failure cases
    find_failure_cases(y_true, y_pred, y_pred_proba, class_names)

    print("\n[DONE] Evaluation complete. All results saved to /results/")
