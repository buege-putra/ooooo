from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from src.captioning.base import BaseCaptionDecoder


def greedy_decode(
    decoder: BaseCaptionDecoder,
    image_features: np.ndarray,
    token_to_id: Mapping[str, int],
    id_to_token: Mapping[int, str] | None = None,
    max_caption_len: int = 30,
    start_token: str = "<start>",
    end_token: str = "<end>",
) -> list[str]:
    """melakukan greedy decoding untuk satu image"""
    features = np.asarray(image_features, dtype=np.float32)
    if features.ndim == 1:
        features = features[None, :]
    if features.shape[0] != 1:
        raise ValueError(f"greedy_decode expects one image feature. Got batch size {features.shape[0]}.")

    reverse_vocab = dict(id_to_token or {idx: token for token, idx in token_to_id.items()})
    start_id = token_to_id[start_token]
    end_id = token_to_id[end_token]
    current = np.array([start_id], dtype=np.int64)
    state = decoder.initial_state(1)
    tokens: list[str] = []
    for step_index in range(max_caption_len):
        probs, state = decoder.step(current, state, image_features=features if step_index == 0 else None)
        next_id = int(np.argmax(probs[0]))
        if next_id == end_id:
            break
        tokens.append(reverse_vocab.get(next_id, str(next_id)))
        current = np.array([next_id], dtype=np.int64)
    return tokens
