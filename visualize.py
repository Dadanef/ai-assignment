from __future__ import annotations

import os
from typing import List, Dict, Any

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from config import EMOTIONS, EMOTION_COLORS


def _load_image_rgb(image_path: str):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_annotated_image(image_path: str, faces_data: List[Dict[str, Any]], save_path: str):
    image = _load_image_rgb(image_path)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title("Emotion Detection - Annotated Faces")

    for face in faces_data:
        box = face["box"]
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        label = face["top_emotion"]
        score = face["top_score"] * 100
        color = EMOTION_COLORS.get(label, "#ffffff")

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2.5,
            edgecolor=color,
            facecolor="none"
        )
        ax.add_patch(rect)

        ax.text(
            x,
            max(0, y - 8),
            f"{label} ({score:.1f}%)",
            color="white",
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.9, edgecolor="none", pad=4)
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_summary_figure(image_path: str, faces_data: List[Dict[str, Any]], save_path: str):
    """
    Creates one summary figure:
    - left: original image with color-coded boxes
    - right: one mini bar chart per face
    """
    image = _load_image_rgb(image_path)
    n_faces = max(1, len(faces_data))

    fig = plt.figure(figsize=(14, 4 + 3 * n_faces))
    gs = GridSpec(n_faces, 2, width_ratios=[1.2, 1], figure=fig)

    # Left big panel
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(image)
    ax_img.axis("off")
    ax_img.set_title("Detected Faces")

    for face in faces_data:
        box = face["box"]
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        label = face["top_emotion"]
        score = face["top_score"] * 100
        color = EMOTION_COLORS.get(label, "#ffffff")

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2.5,
            edgecolor=color,
            facecolor="none"
        )
        ax_img.add_patch(rect)

        ax_img.text(
            x,
            max(0, y - 8),
            f"Face {face['face_id']}: {label} ({score:.1f}%)",
            color="white",
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.9, edgecolor="none", pad=4)
        )

    # Right mini bar charts
    if faces_data:
        for row_idx, face in enumerate(faces_data):
            ax_bar = fig.add_subplot(gs[row_idx, 1])
            scores = [face["emotions"][emotion] for emotion in EMOTIONS]
            colors = [EMOTION_COLORS[e] for e in EMOTIONS]

            ax_bar.bar(EMOTIONS, scores, color=colors)
            ax_bar.set_ylim(0, 1)
            ax_bar.set_ylabel("Confidence")
            ax_bar.set_title(
                f"Face {face['face_id']} - {face['top_emotion']} ({face['top_score']*100:.1f}%)"
            )
            ax_bar.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)