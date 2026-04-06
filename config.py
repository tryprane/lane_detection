from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class ProjectConfig:
    data_root: str = "./data/CULane"
    results_root: str = "./lane_detection/results"
    image_size: Tuple[int, int] = (360, 640)
    train_list: str = "list/train_gt.txt"
    val_list: str = "list/val_gt.txt"
    test_list: str = ""
    seed: int = 42
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    threshold: float = 0.5
    checkpoint_every: int = 1

    def as_dict(self) -> Dict:
        return asdict(self)

    @property
    def image_height(self) -> int:
        return self.image_size[0]

    @property
    def image_width(self) -> int:
        return self.image_size[1]

    @property
    def results_path(self) -> Path:
        return Path(self.results_root)
