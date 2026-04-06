import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _normalize_state_dict_keys(
    state_dict: Dict[str, Any],
    expects_module_prefix: bool,
) -> Dict[str, Any]:
    has_module_prefix = any(key.startswith("module.") for key in state_dict)
    if expects_module_prefix == has_module_prefix:
        return state_dict

    if expects_module_prefix:
        return {f"module.{key}": value for key, value in state_dict.items()}
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def save_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    best_score: Optional[float] = None,
    config: Optional[Dict] = None,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = _unwrap_model(model)

    payload = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_score": best_score,
        "config": config or {},
    }
    torch.save(payload, path)

    if config is not None:
        with path.with_suffix(".json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_state_dict = checkpoint["model_state_dict"]
    expects_module_prefix = any(key.startswith("module.") for key in model.state_dict())
    model_state_dict = _normalize_state_dict_keys(model_state_dict, expects_module_prefix)
    model.load_state_dict(model_state_dict)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
