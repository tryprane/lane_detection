from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _normalize_rel_path(path_str: str) -> str:
    # Keep dataset-relative paths portable across Linux and Windows.
    return path_str.strip().lstrip("/").replace("\\", "/")


def _resolve_existing_file(data_root: Path, relative_path: str) -> Path:
    normalized = relative_path.replace("\\", "/")
    candidates = [data_root / normalized]

    posix_path = PurePosixPath(normalized)
    parts = posix_path.parts
    if parts:
        duplicated_root = PurePosixPath(parts[0], *parts)
        candidates.append(data_root / str(duplicated_root))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return data_root / normalized


def _parse_list_line(parts: Sequence[str], line_number: int, list_path: Path) -> Dict:
    if len(parts) < 2:
        raise ValueError(
            f"Invalid CULane list line at {list_path}:{line_number}. "
            "Expected at least image path and segmentation label path."
        )

    image_rel = _normalize_rel_path(parts[0])
    mask_rel = _normalize_rel_path(parts[1])
    flags = [int(value) for value in parts[2:]] if len(parts) > 2 else []
    return {"image_path": image_rel, "mask_path": mask_rel, "lane_flags": flags}


def _read_split_file(list_path: Path) -> List[Dict]:
    records: List[Dict] = []
    with list_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            records.append(_parse_list_line(stripped.split(), line_number, list_path))
    return records


def get_culane_split_records(
    data_root: str,
    train_list: str = "list/train_gt.txt",
    val_list: str = "list/val_gt.txt",
    test_list: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    root = Path(data_root)
    split_files = {
        "train": root / train_list,
        "val": root / val_list,
    }

    if test_list is not None:
        split_files["test"] = root / test_list
    else:
        default_test = root / "list/test_gt.txt"
        split_files["test"] = default_test if default_test.exists() else root / val_list

    missing = [str(path) for path in split_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required CULane split files: {missing}")

    return {split_name: _read_split_file(list_path) for split_name, list_path in split_files.items()}


class CULaneDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        records: Sequence[Dict],
        image_size: Tuple[int, int] = (360, 640),
        transform=None,
        return_original: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.records = list(records)
        self.image_size = image_size
        self.transform = transform
        self.return_original = return_original

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, relative_path: str) -> np.ndarray:
        image_path = _resolve_existing_file(self.data_root, relative_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read CULane image at {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_mask(self, relative_path: str) -> np.ndarray:
        mask_path = _resolve_existing_file(self.data_root, relative_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Unable to read CULane mask at {mask_path}")

        # CULane lane segmentation labels are lane IDs on a single-channel mask.
        # For binary segmentation we treat any non-zero, non-ignore pixel as lane.
        binary_mask = ((mask > 0) & (mask != 255)).astype(np.uint8)
        return binary_mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        image = self._load_image(record["image_path"])
        mask = self._load_mask(record["mask_path"])

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

        mask = mask.float().unsqueeze(0)
        sample = {"image": image, "mask": mask, "image_path": record["image_path"]}

        if self.return_original:
            sample["original_image"] = original_image
            sample["original_mask"] = original_mask

        return sample
