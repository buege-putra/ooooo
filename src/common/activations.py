from __future__ import annotations

from collections.abc import Callable

import numpy as np


Activation = Callable[[np.ndarray], np.ndarray]


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    positive = x >= 0
    negative = ~positive
    z = np.empty_like(x, dtype=np.float64)
    z[positive] = 1 / (1 + np.exp(-x[positive]))
    exp_x = np.exp(x[negative])
    z[negative] = exp_x / (1 + exp_x)
    return z.astype(np.result_type(x, np.float32), copy=False)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def linear(x: np.ndarray) -> np.ndarray:
    return x


def get_activation(name: str | Activation | None) -> Activation:
    if callable(name):
        return name
    if name is None:
        return linear

    normalized = name.lower()
    activations: dict[str, Activation] = {
        "linear": linear,
        "none": linear,
        "relu": relu,
        "sigmoid": sigmoid,
        "tanh": tanh,
        "softmax": softmax,
    }
    if normalized not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[normalized]


def activation_name(activation: str | Activation | None) -> str:
    if activation is None:
        return "linear"
    if isinstance(activation, str):
        return activation.lower()
    return getattr(activation, "__name__", activation.__class__.__name__)
