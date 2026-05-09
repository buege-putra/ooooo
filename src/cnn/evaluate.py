from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.common.metrics import macro_f1_score


def predict_keras(model: Any, images: np.ndarray, batch_size: int | None = None) -> np.ndarray:
    """prediksi label dari model keras"""

    kwargs = {} if batch_size is None else {"batch_size": batch_size}
    probabilities = np.asarray(model.predict(images, verbose=0, **kwargs))
    return np.argmax(probabilities, axis=-1)


def predict_numpy(model: Any, images: np.ndarray) -> np.ndarray:
    """prediksi label dari model lokal"""

    if hasattr(model, "predict"):
        return np.asarray(model.predict(images))
    probabilities = np.asarray(model(images))
    return np.argmax(probabilities, axis=-1)


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] | None = None,
) -> dict[str, float]:
    """hitung metrik klasifikasi generik"""

    return {"macro_f1": macro_f1_score(y_true, y_pred, labels=labels)}


def parameter_count(model: Any) -> int:
    """hitung jumlah parameter model"""

    if hasattr(model, "parameter_count"):
        return int(model.parameter_count())
    if hasattr(model, "count_params"):
        return int(model.count_params())
    if hasattr(model, "get_weights"):
        return int(sum(np.prod(weight.shape) for weight in model.get_weights()))
    raise TypeError("Model does not expose parameter_count, count_params, or get_weights.")


def evaluation_record(
    model_name: str,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    model: Any | None = None,
    labels: Sequence[int] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """buat record evaluasi untuk dipakai notebook"""

    record: dict[str, Any] = {"model_name": model_name}
    record.update(evaluate_predictions(y_true, y_pred, labels=labels))
    if model is not None:
        record["parameter_count"] = parameter_count(model)
    if extra:
        record.update(extra)
    return record
