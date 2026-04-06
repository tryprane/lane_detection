import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations import get_training_augmentation, get_validation_augmentation
from config import ProjectConfig
from dataset import CULaneDataset, get_culane_split_records
from models import ENet21
from utils import BCEDiceLoss, MetricTracker, compute_batch_metrics, load_checkpoint, plot_training_history, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ENet-21 inspired lane detection on CULane.")
    parser.add_argument("--data-root", type=str, required=True, help="Path to CULane dataset root.")
    parser.add_argument("--results-root", type=str, default="./lane_detection/results", help="Directory to save outputs.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--train-list", type=str, default="list/train_gt.txt")
    parser.add_argument("--val-list", type=str, default="list/val_gt.txt")
    parser.add_argument("--test-list", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--experiment", choices=["baseline", "augmented", "both"], default="both")
    parser.add_argument("--disable-scheduler", action="store_true")
    parser.add_argument("--pretrained-checkpoint", type=str, default="", help="Optional checkpoint to initialize weights from.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_model(model: nn.Module, device: torch.device) -> nn.Module:
    model = model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)
    return model


def build_dataloaders(config: ProjectConfig, use_weather: bool) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    splits = get_culane_split_records(
        config.data_root,
        train_list=config.train_list,
        val_list=config.val_list,
        test_list=config.test_list or None,
    )

    train_dataset = CULaneDataset(
        data_root=config.data_root,
        records=splits["train"],
        image_size=config.image_size,
        transform=get_training_augmentation(config.image_size, use_weather=use_weather),
    )
    val_dataset = CULaneDataset(
        data_root=config.data_root,
        records=splits["val"],
        image_size=config.image_size,
        transform=get_validation_augmentation(config.image_size),
    )
    test_dataset = CULaneDataset(
        data_root=config.data_root,
        records=splits["test"],
        image_size=config.image_size,
        transform=get_validation_augmentation(config.image_size),
    )

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    split_sizes = {key: len(value) for key, value in splits.items()}
    return loaders, split_sizes


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


def train_experiment(
    config: ProjectConfig,
    use_weather: bool,
    experiment_name: str,
    use_scheduler: bool,
    pretrained_checkpoint: str = "",
) -> Dict[str, float]:
    device = get_device()
    loaders, split_sizes = build_dataloaders(config, use_weather)
    criterion = BCEDiceLoss()
    model = prepare_model(ENet21(num_classes=1), device)
    if pretrained_checkpoint:
        load_checkpoint(pretrained_checkpoint, model, map_location=str(device))
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3) if use_scheduler else None

    experiment_dir = Path(config.results_root) / experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_iou": [],
        "val_dice": [],
        "val_pixel_accuracy": [],
    }

    best_iou = -1.0
    for epoch in range(1, config.epochs + 1):
        start_time = time.time()
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer=optimizer, threshold=config.threshold)
        val_metrics = run_epoch(model, loaders["val"], criterion, device, optimizer=None, threshold=config.threshold)

        if scheduler is not None:
            scheduler.step(val_metrics["iou"])

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_pixel_accuracy"].append(val_metrics["pixel_accuracy"])

        epoch_time = time.time() - start_time
        print(
            f"[{experiment_name}] Epoch {epoch:03d}/{config.epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        checkpoint_config = {
            **config.as_dict(),
            "experiment_name": experiment_name,
            "use_weather": use_weather,
            "pretrained_checkpoint": pretrained_checkpoint,
        }
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(
                str(checkpoint_dir / "best_model.pth"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best_iou,
                config=checkpoint_config,
            )

        if epoch % config.checkpoint_every == 0:
            save_checkpoint(
                str(checkpoint_dir / f"epoch_{epoch:03d}.pth"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best_iou,
                config=checkpoint_config,
            )

    with (experiment_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    metadata = {
        "split_sizes": split_sizes,
        "experiment_name": experiment_name,
        "use_weather": use_weather,
        "device": str(device),
        "gpu_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "multi_gpu": isinstance(model, nn.DataParallel),
        "best_val_iou": best_iou,
    }
    with (experiment_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    plot_training_history(str(experiment_dir / "training_curves.png"), history)

    best_model = prepare_model(ENet21(num_classes=1), device)
    load_checkpoint(str(checkpoint_dir / "best_model.pth"), best_model, map_location=str(device))
    test_metrics = run_epoch(best_model, loaders["test"], criterion, device, optimizer=None, threshold=config.threshold)
    with (experiment_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2)

    print(
        f"[{experiment_name}] Test | "
        f"loss={test_metrics['loss']:.4f} | "
        f"iou={test_metrics['iou']:.4f} | "
        f"dice={test_metrics['dice']:.4f} | "
        f"pixel_acc={test_metrics['pixel_accuracy']:.4f}"
    )

    return test_metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = ProjectConfig(
        data_root=args.data_root,
        results_root=args.results_root,
        image_size=(args.image_height, args.image_width),
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        checkpoint_every=args.checkpoint_every,
    )
    config.train_list = args.train_list
    config.val_list = args.val_list
    config.test_list = args.test_list

    experiments = []
    if args.experiment in {"baseline", "both"}:
        experiments.append(("baseline", False))
    if args.experiment in {"augmented", "both"}:
        experiments.append(("weather_augmented", True))

    comparison = {}
    for experiment_name, use_weather in experiments:
        comparison[experiment_name] = train_experiment(
            config=config,
            use_weather=use_weather,
            experiment_name=experiment_name,
            use_scheduler=not args.disable_scheduler,
            pretrained_checkpoint=args.pretrained_checkpoint,
        )

    if len(comparison) > 1:
        comparison_path = Path(config.results_root) / "comparison.json"
        with comparison_path.open("w", encoding="utf-8") as file:
            json.dump(comparison, file, indent=2)


if __name__ == "__main__":
    main()
