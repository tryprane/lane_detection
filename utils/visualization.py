from pathlib import Path
from typing import Dict, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    overlay = image.copy()
    binary_mask = (mask > 0).astype(np.uint8)
    overlay[binary_mask == 1] = color
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)


def save_prediction_panel(
    save_path: str,
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    title: Optional[str] = None,
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[1].imshow(ground_truth, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(prediction, cmap="gray")
    axes[2].set_title("Prediction")
    axes[3].imshow(overlay_mask_on_image(image, prediction))
    axes[3].set_title("Overlay")

    if title:
        fig.suptitle(title)

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(save_path: str, history: Dict[str, list]) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["val_iou"], label="Val IoU")
    axes[1].plot(epochs, history["val_dice"], label="Val Dice")
    axes[1].plot(epochs, history["val_pixel_accuracy"], label="Val Pixel Accuracy")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
