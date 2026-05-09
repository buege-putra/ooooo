from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class ImageSplit:
    """hasil load satu split dataset cnn"""

    images: np.ndarray
    labels: np.ndarray
    paths: tuple[Path, ...]
    class_names: tuple[str, ...]


def load_image(
    path: str | Path,
    target_size: tuple[int, int],
    normalize: bool = True,
    dtype: np.dtype | str = np.float32,
) -> np.ndarray:
    """load image dari path menjadi array channel-last"""

    image_path = Path(path)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((target_size[1], target_size[0]))
        array = np.asarray(image, dtype=dtype)

    if normalize:
        array = array / np.asarray(255.0, dtype=dtype)
    return array.astype(dtype, copy=False)


def load_image_batch(
    paths: Sequence[str | Path],
    target_size: tuple[int, int],
    normalize: bool = True,
    dtype: np.dtype | str = np.float32,
) -> np.ndarray:
    """load sekumpulan image menjadi array batch"""

    arrays = [load_image(path, target_size=target_size, normalize=normalize, dtype=dtype) for path in paths]
    if not arrays:
        height, width = target_size
        return np.empty((0, height, width, 3), dtype=dtype)
    return np.stack(arrays, axis=0).astype(dtype, copy=False)


def iter_class_image_paths(
    split_dir: str | Path,
    class_names: Sequence[str] | None = None,
    extensions: Sequence[str] = SUPPORTED_IMAGE_EXTENSIONS,
) -> Iterator[tuple[Path, int]]:
    """iterasi path image dan label dari folder kelas"""

    root = Path(split_dir)
    if not root.exists():
        raise FileNotFoundError(f"Split directory not found: {root}")

    resolved_class_names = tuple(class_names) if class_names is not None else discover_class_names(root)
    normalized_extensions = tuple(extension.lower() for extension in extensions)
    for label, class_name in enumerate(resolved_class_names):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in normalized_extensions
        )
        for image_path in image_paths:
            yield image_path, label


def load_split(
    split_dir: str | Path,
    target_size: tuple[int, int],
    class_names: Sequence[str] | None = None,
    normalize: bool = True,
    shuffle: bool = False,
    seed: int | None = None,
    limit_per_class: int | None = None,
    dtype: np.dtype | str = np.float32,
) -> ImageSplit:
    """load satu split dataset dari folder kelas"""

    root = Path(split_dir)
    resolved_class_names = tuple(class_names) if class_names is not None else discover_class_names(root)
    entries_by_class: list[list[Path]] = [[] for _ in resolved_class_names]
    for image_path, label in iter_class_image_paths(root, class_names=resolved_class_names):
        if limit_per_class is None or len(entries_by_class[label]) < limit_per_class:
            entries_by_class[label].append(image_path)

    paths: list[Path] = []
    labels: list[int] = []
    for label, class_paths in enumerate(entries_by_class):
        paths.extend(class_paths)
        labels.extend([label] * len(class_paths))

    order = np.arange(len(paths))
    if shuffle and len(order) > 0:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
        paths = [paths[index] for index in order]
        labels = [labels[index] for index in order]

    images = load_image_batch(paths, target_size=target_size, normalize=normalize, dtype=dtype)
    return ImageSplit(
        images=images,
        labels=np.asarray(labels, dtype=np.int64),
        paths=tuple(paths),
        class_names=resolved_class_names,
    )


def load_dataset(
    root_dir: str | Path,
    target_size: tuple[int, int],
    split_names: Sequence[str] = ("train", "validation", "test"),
    split_dirs: Mapping[str, str | Path] | None = None,
    class_names: Sequence[str] | None = None,
    normalize: bool = True,
    shuffle: bool = False,
    seed: int | None = None,
    limit_per_class: int | None = None,
    dtype: np.dtype | str = np.float32,
) -> dict[str, ImageSplit]:
    """load beberapa split dataset cnn"""

    root = Path(root_dir)
    dataset: dict[str, ImageSplit] = {}
    resolved_class_names = class_names
    if resolved_class_names is None and split_names:
        first_split_name = split_names[0]
        first_split_path = (
            Path(split_dirs[first_split_name])
            if split_dirs and first_split_name in split_dirs
            else root / first_split_name
        )
        resolved_class_names = discover_class_names(first_split_path)

    for split_name in split_names:
        split_path = Path(split_dirs[split_name]) if split_dirs and split_name in split_dirs else root / split_name
        dataset[split_name] = load_split(
            split_path,
            target_size=target_size,
            class_names=resolved_class_names,
            normalize=normalize,
            shuffle=shuffle,
            seed=seed,
            limit_per_class=limit_per_class,
            dtype=dtype,
        )
    return dataset


def discover_class_names(split_dir: str | Path) -> tuple[str, ...]:
    """ambil nama kelas dari subfolder split secara alphabetic"""

    root = Path(split_dir)
    if not root.exists():
        raise FileNotFoundError(f"Split directory not found: {root}")
    return tuple(sorted(path.name for path in root.iterdir() if path.is_dir()))


def batch_iterator(
    images: np.ndarray,
    labels: np.ndarray | None = None,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int | None = None,
    drop_last: bool = False,
) -> Iterator[np.ndarray | tuple[np.ndarray, np.ndarray]]:
    """buat iterator batch untuk array image dan label opsional"""

    image_array = np.asarray(images)
    if image_array.ndim != 4:
        raise ValueError(f"Images must have shape (N, H, W, C). Got {image_array.shape}.")
    if batch_size <= 0:
        raise ValueError(f"Invalid batch_size. Expected a positive integer, got {batch_size}.")

    label_array = None if labels is None else np.asarray(labels)
    if label_array is not None and len(label_array) != len(image_array):
        raise ValueError(f"Labels length {len(label_array)} does not match images length {len(image_array)}.")

    indices = np.arange(len(image_array))
    if shuffle and len(indices) > 0:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        if drop_last and len(batch_indices) < batch_size:
            continue
        batch_images = image_array[batch_indices]
        if label_array is None:
            yield batch_images
        else:
            yield batch_images, label_array[batch_indices]


def split_paths_and_labels(entries: Iterable[tuple[Path, int]]) -> tuple[tuple[Path, ...], np.ndarray]:
    """pisahkan entry path-label"""

    paths: list[Path] = []
    labels: list[int] = []
    for path, label in entries:
        paths.append(path)
        labels.append(label)
    return tuple(paths), np.asarray(labels, dtype=np.int64)
