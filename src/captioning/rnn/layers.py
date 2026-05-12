from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.captioning.base import RecurrentCell
from src.common.activations import get_activation
from src.common.layers import ensure_numeric_array, validate_weight_count, validate_weight_shape


class SimpleRNNCell(RecurrentCell):
    """cell simplernn untuk forward satu timestep"""

    def __init__(self, units: int, activation: str = "tanh", name: str | None = None, dtype: np.dtype | str = np.float32) -> None:
        super().__init__(units=units, name=name, dtype=dtype)
        self.activation_name = activation
        self.activation = get_activation(activation)
        self.kernel: np.ndarray | None = None
        self.recurrent_kernel: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def build(self, input_dim: int) -> None:
        super().build(input_dim)
        self.kernel = np.zeros((self.input_dim, self.units), dtype=self.dtype)
        self.recurrent_kernel = np.zeros((self.units, self.units), dtype=self.dtype)
        self.bias = np.zeros((self.units,), dtype=self.dtype)

    def initial_state(self, batch_size: int) -> np.ndarray:
        return np.zeros((batch_size, self.units), dtype=self.dtype)

    def step(self, inputs: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.kernel is None:
            self.build(inputs.shape[-1])
        assert self.kernel is not None and self.recurrent_kernel is not None and self.bias is not None
        output = self.activation(np.asarray(inputs, dtype=self.dtype) @ self.kernel + state @ self.recurrent_kernel + self.bias)
        return output, output

    @property
    def weight_count(self) -> int:
        return 3

    def get_weights(self) -> list[np.ndarray]:
        if self.kernel is None:
            return []
        assert self.recurrent_kernel is not None and self.bias is not None
        return [self.kernel, self.recurrent_kernel, self.bias]

    def set_weights(self, weights: Sequence[np.ndarray]) -> None:
        validate_weight_count(self.name, weights, 3)
        kernel = ensure_numeric_array(weights[0], f"{self.name}.kernel")
        recurrent_kernel = ensure_numeric_array(weights[1], f"{self.name}.recurrent_kernel")
        bias = ensure_numeric_array(weights[2], f"{self.name}.bias")
        if kernel.ndim != 2:
            raise ValueError(f"{self.name} kernel must be rank 2. Got shape {kernel.shape}.")
        expected_kernel = (kernel.shape[0], self.units) if self.kernel is None else self.kernel.shape
        validate_weight_shape(self.name, kernel.shape, expected_kernel, "kernel")
        validate_weight_shape(self.name, recurrent_kernel.shape, (self.units, self.units), "recurrent_kernel")
        validate_weight_shape(self.name, bias.shape, (self.units,), "bias")
        self.kernel = kernel.astype(self.dtype, copy=True)
        self.recurrent_kernel = recurrent_kernel.astype(self.dtype, copy=True)
        self.bias = bias.astype(self.dtype, copy=True)
        self.input_dim = int(kernel.shape[0])
        self.built = True

