from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    resolved = ensure_parent_dir(path)
    with resolved.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_npy(array: np.ndarray, path: str | Path) -> None:
    resolved = ensure_parent_dir(path)
    np.save(resolved, array)


def load_npy(path: str | Path, allow_pickle: bool = False) -> np.ndarray:
    return np.load(Path(path), allow_pickle=allow_pickle)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    save_json(config, path)


def load_config(path: str | Path) -> dict[str, Any]:
    return load_json(path)


def save_history(history: Any, path: str | Path) -> None:
    if hasattr(history, "history"):
        history = history.history
    save_json(history, path)


def load_history(path: str | Path) -> dict[str, Any]:
    return load_json(path)
