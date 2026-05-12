from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
_PUNCTUATION_RE = re.compile(r"[^a-z0-9\s]+")


def clean_caption(caption: str) -> str:
    """membersihkan caption dengan lowercase dan penghapusan tanda baca"""
    lowered = caption.lower()
    cleaned = _PUNCTUATION_RE.sub(" ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize_caption(caption: str) -> list[str]:
    """memecah caption bersih menjadi token sederhana"""
    return clean_caption(caption).split()


def parse_caption_file(path: str | Path, separator: str | None = None) -> dict[str, list[str]]:
    """membaca file caption menjadi mapping image id ke daftar caption"""
    captions: dict[str, list[str]] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("image,caption"):
            continue
        if separator is not None:
            image_id, caption = line.split(separator, 1)
        elif "\t" in line:
            image_id, caption = line.split("\t", 1)
        elif "," in line:
            image_id, caption = line.split(",", 1)
        else:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            image_id, caption = parts
        image_id = image_id.split("#", 1)[0].strip()
        captions.setdefault(image_id, []).append(caption.strip())
    return captions


def build_vocabulary(
    captions: Iterable[str],
    min_freq: int = 1,
    max_tokens: int | None = None,
    special_tokens: Sequence[str] = (PAD_TOKEN, START_TOKEN, END_TOKEN),
) -> dict[str, int]:
    """membangun mapping token ke id dari caption training"""
    counter: Counter[str] = Counter()
    for caption in captions:
        counter.update(tokenize_caption(caption))
    words = [word for word, count in counter.most_common() if count >= min_freq]
    if max_tokens is not None:
        words = words[: max(0, max_tokens - len(special_tokens))]
    vocab: dict[str, int] = {}
    for token in special_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def encode_caption(caption: str, token_to_id: Mapping[str, int], add_special_tokens: bool = True) -> list[int]:
    """mengubah caption menjadi daftar token id"""
    tokens = tokenize_caption(caption)
    if add_special_tokens:
        tokens = [START_TOKEN, *tokens, END_TOKEN]
    return [token_to_id[token] for token in tokens if token in token_to_id]


def pad_sequences(sequences: Sequence[Sequence[int]], max_len: int | None = None, pad_value: int = 0) -> np.ndarray:
    """melakukan padding sequence token ke panjang seragam"""
    length = max_len if max_len is not None else max((len(seq) for seq in sequences), default=0)
    output = np.full((len(sequences), length), pad_value, dtype=np.int64)
    for row, sequence in enumerate(sequences):
        truncated = list(sequence)[:length]
        output[row, : len(truncated)] = truncated
    return output


def make_teacher_forcing_sequences(encoded_captions: Sequence[Sequence[int]], pad_value: int = 0, max_len: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """membuat input dan target caption yang digeser satu posisi"""
    inputs = [list(sequence[:-1]) for sequence in encoded_captions]
    targets = [list(sequence[1:]) for sequence in encoded_captions]
    return pad_sequences(inputs, max_len=max_len, pad_value=pad_value), pad_sequences(targets, max_len=max_len, pad_value=pad_value)


def save_vocabulary(token_to_id: Mapping[str, int], path: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
    """menyimpan vocabulary dan metadata preprocessing"""
    payload = {"token_to_id": dict(token_to_id), "metadata": dict(metadata or {})}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_vocabulary(path: str | Path) -> tuple[dict[str, int], dict[str, Any]]:
    """memuat vocabulary dan metadata preprocessing"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in payload["token_to_id"].items()}, dict(payload.get("metadata", {}))
