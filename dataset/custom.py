import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _paired_records(images_dir: Path, masks_dir: Path, mask_suffix: str) -> List[Dict]:
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    records: List[Dict] = []
    for image_path in sorted(path for path in images_dir.iterdir() if path.suffix.lower() in supported):
        mask_name = f"{image_path.stem}{mask_suffix}"
        mask_path = masks_dir / mask_name
        if mask_path.exists():
            records.append({"image_path": str(image_path), "mask_path": str(mask_path)})
    if not records:
        raise FileNotFoundError(
            f"No image/mask pairs found in {images_dir} and {masks_dir} using mask suffix '{mask_suffix}'."
        )
    return records


def create_custom_split_records(
    images_dir: str,
    masks_dir: str,
    train_ratio: float = 0.75,
    val_ratio: float = 0.125,
    test_ratio: float = 0.125,
    seed: int = 42,
    mask_suffix: str = ".png",
) -> Dict[str, List[Dict]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    records = _paired_records(Path(images_dir), Path(masks_dir), mask_suffix)
    rng = random.Random(seed)
    rng.shuffle(records)

    train_end = int(len(records) * train_ratio)
    val_end = train_end + int(len(records) * val_ratio)
    return {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }


class CustomLaneMaskDataset(Dataset):
    def __init__(self, records: Sequence[Dict], transform=None, return_original: bool = False) -> None:
        self.records = list(records)
        self.transform = transform
        self.return_original = return_original

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        image = cv2.imread(record["image_path"])
        if image is None:
            raise FileNotFoundError(f"Unable to read custom image at {record['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(record["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Unable to read custom mask at {record['mask_path']}")
        mask = (mask > 0).astype(np.uint8)

        original_image = image.copy()
        original_mask = mask.copy()

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask)

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)

        sample = {
            "image": image,
            "mask": mask.float().unsqueeze(0),
            "image_path": record["image_path"],
        }
        if self.return_original:
            sample["original_image"] = original_image
            sample["original_mask"] = original_mask
        return sample
