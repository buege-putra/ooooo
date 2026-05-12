from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.common.metrics import bleu4_score, meteor_score, time_call


def evaluate_caption_predictions(references: Sequence[Sequence[str] | Sequence[Sequence[str]]], hypotheses: Sequence[Sequence[str]]) -> dict[str, float]:
    """menghitung metrik utama captioning"""
    return {"bleu4": bleu4_score(references, hypotheses), "meteor": meteor_score(references, hypotheses)}


def timed_caption_generation(generator: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """menjalankan generator caption sambil mengukur waktu"""
    return time_call(generator, *args, **kwargs)


def qualitative_record(
    image_id: str,
    rnn_caption: Sequence[str] | str,
    lstm_caption: Sequence[str] | str,
    ground_truth: Sequence[Sequence[str] | str] | Sequence[str] | str,
    scores: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """membentuk record qualitative analysis generik"""
    record: dict[str, Any] = {
        "image_id": image_id,
        "rnn_caption": " ".join(rnn_caption) if not isinstance(rnn_caption, str) else rnn_caption,
        "lstm_caption": " ".join(lstm_caption) if not isinstance(lstm_caption, str) else lstm_caption,
        "ground_truth": ground_truth,
    }
    if scores:
        record.update(scores)
    if extra:
        record.update(extra)
    return record
