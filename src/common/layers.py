from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .activations import activation_name, get_activation
from .base import Layer, Shape


def ensure_numeric_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric. Got dtype {array.dtype}.")
    return array


def validate_weight_count(layer_name: str, weights: Sequence[np.ndarray], expected: int) -> None:
    if len(weights) != expected:
        raise ValueError(f"{layer_name} expects {expected} weight arrays. Got {len(weights)}.")


def validate_weight_shape(layer_name: str, actual: Sequence[int], expected: Sequence[int], label: str) -> None:
    actual_tuple = tuple(actual)
    expected_tuple = tuple(expected)
    if actual_tuple != expected_tuple:
        raise ValueError(
            f"{layer_name} {label} shape mismatch. Got Keras shape "
            f"{actual_tuple}. Expected local shape {expected_tuple}."
        )


class Dense(Layer):
    """implementasi numpy untuk forward propagation dense"""

    def __init__(
        self,
        units: int,
        activation: str | None = None,
        use_bias: bool = True,
        name: str | None = None,
        dtype: np.dtype | str = np.float32,
    ) -> None:
        super().__init__(name=name)
        self.units = int(units)
        self.activation_name = activation_name(activation)
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.dtype = np.dtype(dtype)
        self.kernel: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def build(self, input_shape: Sequence[int | None]) -> None:
        if len(input_shape) < 2:
            raise ValueError(f"{self.name} expects input with rank >= 2. Got {tuple(input_shape)}.")
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(f"{self.name} requires known last input dimension.")
        self.kernel = np.zeros((int(input_dim), self.units), dtype=self.dtype)
        self.bias = np.zeros((self.units,), dtype=self.dtype) if self.use_bias else None
        super().build(input_shape)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if self.kernel is None:
            self.build(inputs.shape)
        assert self.kernel is not None
        outputs = np.matmul(inputs, self.kernel)
        if self.use_bias:
            assert self.bias is not None
            outputs = outputs + self.bias
        return self.activation(outputs)

    def get_weights(self) -> list[np.ndarray]:
        if self.kernel is None:
            return []
        weights = [self.kernel]
        if self.use_bias:
            assert self.bias is not None
            weights.append(self.bias)
        return weights

    @property
    def weight_count(self) -> int:
        return 2 if self.use_bias else 1

    def _expected_kernel_shape(self, loaded_kernel: np.ndarray) -> tuple[int, int]:
        if self.kernel is None:
            return (loaded_kernel.shape[0], self.units)
        return self.kernel.shape

    def set_weights(self, weights: Sequence[np.ndarray]) -> None:
        validate_weight_count(self.name, weights, 2 if self.use_bias else 1)
        kernel = ensure_numeric_array(weights[0], f"{self.name}.kernel")
        if kernel.ndim != 2:
            raise ValueError(f"{self.name} kernel must be rank 2. Got shape {kernel.shape}.")
        validate_weight_shape(self.name, kernel.shape, self._expected_kernel_shape(kernel), "kernel")
        self.kernel = kernel.astype(self.dtype, copy=True)
        if self.use_bias:
            bias = ensure_numeric_array(weights[1], f"{self.name}.bias")
            validate_weight_shape(self.name, bias.shape, (self.units,), "bias")
            self.bias = bias.astype(self.dtype, copy=True)
        self.built = True
        self.input_shape = (None, self.kernel.shape[0])
        self.output_shape = self.compute_output_shape(self.input_shape)

    def compute_output_shape(self, input_shape: Sequence[int | None]) -> Shape:
        return tuple(input_shape[:-1]) + (self.units,)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": self.activation_name,
                "use_bias": self.use_bias,
                "dtype": self.dtype.name,
            }
        )
        return config


class Embedding(Layer):
    """implementasi numpy untuk lookup embedding"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        name: str | None = None,
        dtype: np.dtype | str = np.float32,
    ) -> None:
        super().__init__(name=name)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.dtype = np.dtype(dtype)
        self.embeddings: np.ndarray | None = None

    def build(self, input_shape: Sequence[int | None]) -> None:
        self.embeddings = np.zeros((self.input_dim, self.output_dim), dtype=self.dtype)
        super().build(input_shape)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if self.embeddings is None:
            self.build(inputs.shape)
        assert self.embeddings is not None
        indices = np.asarray(inputs)
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError(f"{self.name} expects integer token indices. Got dtype {indices.dtype}.")
        if np.any(indices < 0) or np.any(indices >= self.input_dim):
            raise ValueError(f"{self.name} received token id outside [0, {self.input_dim}).")
        return self.embeddings[indices]

    @property
    def weight_count(self) -> int:
        return 1

    def get_weights(self) -> list[np.ndarray]:
        return [] if self.embeddings is None else [self.embeddings]

    def set_weights(self, weights: Sequence[np.ndarray]) -> None:
        validate_weight_count(self.name, weights, 1)
        embeddings = ensure_numeric_array(weights[0], f"{self.name}.embeddings")
        validate_weight_shape(self.name, embeddings.shape, (self.input_dim, self.output_dim), "embeddings")
        self.embeddings = embeddings.astype(self.dtype, copy=True)
        self.built = True
        self.output_shape = None

    def compute_output_shape(self, input_shape: Sequence[int | None]) -> Shape:
        return tuple(input_shape) + (self.output_dim,)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "dtype": self.dtype.name,
            }
        )
        return config
