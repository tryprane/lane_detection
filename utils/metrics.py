from dataclasses import dataclass, field
from typing import Dict

import torch


def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = ((intersection + eps) / (union + eps)).mean().item()
    dice = ((2 * intersection + eps) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps)).mean().item()
    pixel_acc = (preds.eq(targets).float().mean(dim=(1, 2, 3))).mean().item()

    return {"iou": iou, "dice": dice, "pixel_accuracy": pixel_acc}


@dataclass
class MetricTracker:
    totals: Dict[str, float] = field(
        default_factory=lambda: {"loss": 0.0, "iou": 0.0, "dice": 0.0, "pixel_accuracy": 0.0}
    )
    count: int = 0

    def update(self, loss: float, metrics: Dict[str, float], batch_size: int) -> None:
        self.totals["loss"] += loss * batch_size
        for key, value in metrics.items():
            self.totals[key] += value * batch_size
        self.count += batch_size

    def average(self) -> Dict[str, float]:
        if self.count == 0:
            return {key: 0.0 for key in self.totals}
        return {key: value / self.count for key, value in self.totals.items()}
