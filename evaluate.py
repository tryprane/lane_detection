import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations import get_validation_augmentation
from dataset import CULaneDataset, get_culane_split_records
from models import ENet21
from utils import BCEDiceLoss, MetricTracker, compute_batch_metrics, load_checkpoint, save_prediction_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ENet-21 inspired lane detector on CULane.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-list", type=str, default="list/train_gt.txt")
    parser.add_argument("--val-list", type=str, default="list/val_gt.txt")
    parser.add_argument("--test-list", type=str, default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-dir", type=str, default="./lane_detection/results/evaluation")
    parser.add_argument("--num-visualizations", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = get_culane_split_records(
        args.data_root,
        train_list=args.train_list,
        val_list=args.val_list,
        test_list=args.test_list or None,
    )
    dataset = CULaneDataset(
        data_root=args.data_root,
        records=splits[args.split],
        image_size=(args.image_height, args.image_width),
        transform=get_validation_augmentation((args.image_height, args.image_width)),
        return_original=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ENet21(num_classes=1).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()
    criterion = BCEDiceLoss()
    tracker = MetricTracker()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            metrics = compute_batch_metrics(logits, masks, threshold=args.threshold)
            tracker.update(loss.item(), metrics, images.size(0))

            preds = (torch.sigmoid(logits) > args.threshold).float().cpu().numpy()
            gt = masks.cpu().numpy()

            if vis_count < args.num_visualizations:
                originals = batch["original_image"].numpy()
                for image, target_mask, pred_mask, image_path in zip(originals, gt, preds, batch["image_path"]):
                    if vis_count >= args.num_visualizations:
                        break
                    save_prediction_panel(
                        str(save_dir / f"{Path(image_path).stem}_panel.png"),
                        image=image,
                        ground_truth=target_mask.squeeze(),
                        prediction=pred_mask.squeeze(),
                        title=image_path,
                    )
                    vis_count += 1

    metrics_summary = tracker.average()
    print(json.dumps(metrics_summary, indent=2))
    with (save_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_summary, file, indent=2)


if __name__ == "__main__":
    main()
