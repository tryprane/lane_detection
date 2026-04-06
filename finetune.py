import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations import get_training_augmentation, get_validation_augmentation
from dataset import CustomLaneMaskDataset, create_custom_split_records
from models import ENet21
from utils import BCEDiceLoss, MetricTracker, compute_batch_metrics, load_checkpoint, plot_training_history, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a CULane-pretrained lane detector on custom labeled frames.")
    parser.add_argument("--images-dir", type=str, required=True, help="Directory containing custom road images.")
    parser.add_argument("--masks-dir", type=str, required=True, help="Directory containing binary lane masks.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Base pretrained checkpoint to fine-tune from.")
    parser.add_argument("--results-root", type=str, default="./lane_detection/results/indian_finetune")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.125)
    parser.add_argument("--test-ratio", type=float, default=0.125)
    parser.add_argument("--mask-suffix", type=str, default=".png", help="Mask filename suffix, e.g. .png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--use-weather-augmentation", action="store_true")
    parser.add_argument("--freeze-encoder-epochs", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(args: argparse.Namespace) -> Dict[str, DataLoader]:
    splits = create_custom_split_records(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        mask_suffix=args.mask_suffix,
    )

    image_size = (args.image_height, args.image_width)
    datasets = {
        "train": CustomLaneMaskDataset(
            splits["train"],
            transform=get_training_augmentation(image_size, use_weather=args.use_weather_augmentation),
        ),
        "val": CustomLaneMaskDataset(
            splits["val"],
            transform=get_validation_augmentation(image_size),
        ),
        "test": CustomLaneMaskDataset(
            splits["test"],
            transform=get_validation_augmentation(image_size),
        ),
    }

    return {
        split_name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split_name == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split_name, dataset in datasets.items()
    }


def set_encoder_trainable(model: ENet21, trainable: bool) -> None:
    encoder_modules = [model.initial, model.stage1, model.stage2, model.stage3]
    for module in encoder_modules:
        for param in module.parameters():
            param.requires_grad = trainable


def run_epoch(
    model: ENet21,
    loader: DataLoader,
    criterion: BCEDiceLoss,
    device: torch.device,
    optimizer: Optional[Adam] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    tracker = MetricTracker()

    progress = tqdm(loader, leave=False)
    for batch in progress:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        metrics = compute_batch_metrics(logits.detach(), masks, threshold=threshold)
        tracker.update(loss.item(), metrics, images.size(0))
        progress.set_description(f"{'train' if is_train else 'eval'} loss={loss.item():.4f}")

    return tracker.average()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_dataloaders(args)
    criterion = BCEDiceLoss()
    model = ENet21(num_classes=1).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    results_root = Path(args.results_root)
    checkpoint_dir = results_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": [],
        "val_pixel_accuracy": [],
    }

    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        set_encoder_trainable(model, trainable=epoch > args.freeze_encoder_epochs)
        start_time = time.time()

        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer=optimizer, threshold=args.threshold)
        val_metrics = run_epoch(model, loaders["val"], criterion, device, optimizer=None, threshold=args.threshold)
        scheduler.step(val_metrics["iou"])

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_pixel_accuracy"].append(val_metrics["pixel_accuracy"])

        print(
            f"[finetune] Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"time={time.time() - start_time:.1f}s"
        )

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(
                str(checkpoint_dir / "best_model.pth"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best_iou,
                config=vars(args),
            )

        if epoch % args.checkpoint_every == 0:
            save_checkpoint(
                str(checkpoint_dir / f"epoch_{epoch:03d}.pth"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best_iou,
                config=vars(args),
            )

    with (results_root / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    plot_training_history(str(results_root / "training_curves.png"), history)

    best_model = ENet21(num_classes=1).to(device)
    load_checkpoint(str(checkpoint_dir / "best_model.pth"), best_model, map_location=str(device))
    test_metrics = run_epoch(best_model, loaders["test"], criterion, device, optimizer=None, threshold=args.threshold)
    with (results_root / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2)
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
