from __future__ import annotations

import os
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

from classifier import classify_emotions
from config import EMOTIONS, SUPPORTED_EXTENSIONS


def predict_single_label(image_path: str) -> str:
    faces = classify_emotions(image_path)

    if not faces:
        return "no_face"

    best_face = max(faces, key=lambda f: f["top_score"])
    return best_face["top_emotion"]


def collect_dataset_predictions(dataset_dir: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for true_label in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, true_label)

        if not os.path.isdir(class_dir):
            continue

        if true_label not in EMOTIONS:
            continue

        for file_name in os.listdir(class_dir):
            if not file_name.lower().endswith(SUPPORTED_EXTENSIONS):
                continue

            image_path = os.path.join(class_dir, file_name)

            try:
                pred_label = predict_single_label(image_path)
                correct = pred_label == true_label

                rows.append({
                    "file_name": file_name,
                    "image_path": image_path,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "correct": correct
                })

            except Exception as e:
                rows.append({
                    "file_name": file_name,
                    "image_path": image_path,
                    "true_label": true_label,
                    "predicted_label": "error",
                    "correct": False,
                    "error_message": str(e)
                })

    return pd.DataFrame(rows)


def compute_per_emotion_accuracy(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for emotion in EMOTIONS:
        subset = results_df[results_df["true_label"] == emotion]
        total = len(subset)

        if total == 0:
            rows.append({
                "emotion": emotion,
                "num_samples": 0,
                "num_correct": 0,
                "accuracy": 0.0
            })
            continue

        num_correct = int(subset["correct"].sum())
        acc = num_correct / total

        rows.append({
            "emotion": emotion,
            "num_samples": total,
            "num_correct": num_correct,
            "accuracy": acc
        })

    return pd.DataFrame(rows)


def save_confusion_matrix(results_df: pd.DataFrame, save_path: str):
    valid_df = results_df[results_df["predicted_label"].isin(EMOTIONS)].copy()

    if valid_df.empty:
        raise ValueError("No valid predictions available for confusion matrix")

    y_true = valid_df["true_label"]
    y_pred = valid_df["predicted_label"]

    cm = confusion_matrix(y_true, y_pred, labels=EMOTIONS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTIONS)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=45)
    ax.set_title("Emotion Classification - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_batch_analysis(dataset_dir: str = "labeled_dataset", output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)

    results_df = collect_dataset_predictions(dataset_dir)

    if results_df.empty:
        raise ValueError("No images found in dataset")

    results_csv_path = os.path.join(output_dir, "results.csv")
    results_df.to_csv(results_csv_path, index=False)

    valid_df = results_df[results_df["predicted_label"].isin(EMOTIONS)].copy()

    if valid_df.empty:
        raise ValueError("No valid emotion predictions available")

    y_true = valid_df["true_label"]
    y_pred = valid_df["predicted_label"]

    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {overall_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=EMOTIONS, zero_division=0))

    per_emotion_df = compute_per_emotion_accuracy(results_df)
    per_emotion_csv_path = os.path.join(output_dir, "per_emotion_accuracy.csv")
    per_emotion_df.to_csv(per_emotion_csv_path, index=False)

    print("\nPer-emotion Accuracy:")
    print(per_emotion_df)

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    save_confusion_matrix(results_df, cm_path)

    summary_path = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Overall Accuracy: {overall_acc:.4f}\n\n")
        f.write("Per-emotion Accuracy:\n")
        f.write(per_emotion_df.to_string(index=False))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(y_true, y_pred, labels=EMOTIONS, zero_division=0))

    print(f"\nSaved: {results_csv_path}")
    print(f"Saved: {per_emotion_csv_path}")
    print(f"Saved: {cm_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    run_batch_analysis()