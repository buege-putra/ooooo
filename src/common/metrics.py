from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np


def macro_f1_score(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int] | None = None) -> float:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch for y_true and y_pred: {true.shape} != {pred.shape}.")
    if labels is None:
        labels = sorted(set(true.tolist()) | set(pred.tolist()))
    f1_scores: list[float] = []
    for label in labels:
        tp = np.sum((true == label) & (pred == label))
        fp = np.sum((true != label) & (pred == label))
        fn = np.sum((true == label) & (pred != label))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1_scores.append(float(f1))
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def _normalize_references(
    references: Sequence[Sequence[str] | Sequence[Sequence[str]]],
) -> list[list[list[str]]]:
    normalized: list[list[list[str]]] = []
    for reference in references:
        reference_items = list(reference)
        if not reference_items:
            normalized.append([])
            continue
        first_item = reference_items[0]
        if isinstance(first_item, str):
            normalized.append([[str(token) for token in reference_items]])
        else:
            normalized.append([list(item) for item in reference_items])
    return normalized


def bleu4_score(references: Sequence[Sequence[str] | Sequence[Sequence[str]]], hypotheses: Sequence[Sequence[str]]) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    return float(
        corpus_bleu(
            _normalize_references(references),
            [list(hypothesis) for hypothesis in hypotheses],
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        )
    )


def meteor_score(references: Sequence[Sequence[str] | Sequence[Sequence[str]]], hypotheses: Sequence[Sequence[str]]) -> float:
    from nltk.translate.meteor_score import meteor_score as nltk_meteor_score

    scores: list[float] = []
    for normalized_ref, hypothesis in zip(_normalize_references(references), hypotheses):
        scores.append(float(nltk_meteor_score(normalized_ref, list(hypothesis))))
    return float(np.mean(scores)) if scores else 0.0


@dataclass
class TimingResult:
    name: str
    seconds: float


@contextmanager
def timer(name: str = "operation") -> Iterable[TimingResult]:
    result = TimingResult(name=name, seconds=0.0)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.seconds = time.perf_counter() - start


def time_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    value = func(*args, **kwargs)
    return value, time.perf_counter() - start


def aggregate_results(records: Sequence[dict[str, Any]], group_by: str | Sequence[str], metrics: Sequence[str]) -> list[dict[str, Any]]:
    keys = [group_by] if isinstance(group_by, str) else list(group_by)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        group_key = tuple(record[key] for key in keys)
        grouped.setdefault(group_key, []).append(record)

    summary: list[dict[str, Any]] = []
    for group_key, rows in grouped.items():
        item = {key: value for key, value in zip(keys, group_key)}
        for metric in metrics:
            values = [row[metric] for row in rows if metric in row and row[metric] is not None]
            item[f"{metric}_mean"] = float(np.mean(values)) if values else None
            item[f"{metric}_std"] = float(np.std(values)) if values else None
            item[f"{metric}_count"] = len(values)
        summary.append(item)
    return summary
