from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from src.common.activations import activation_name, get_activation
from src.common.base import Layer, Shape
from src.common.layers import ensure_numeric_array, validate_weight_count, validate_weight_shape


Padding = Literal["valid", "same"]


def _as_pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        pair = (value, value)
    else:
        pair = tuple(value)
    if len(pair) != 2 or pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} must be a positive int or pair of positive ints. Got {value}.")
    return int(pair[0]), int(pair[1])


def _validate_channel_last(inputs: np.ndarray, layer_name: str) -> np.ndarray:
    array = np.asarray(inputs)
    if array.ndim != 4:
        raise ValueError(f"{layer_name} expects input shape (N, H, W, C). Got {array.shape}.")
    return array


def _output_dim(size: int, kernel: int, stride: int, padding: Padding) -> int:
    if padding == "same":
        return int(np.ceil(size / stride))
    if padding == "valid":
        return (size - kernel) // stride + 1
    raise ValueError(f"Unsupported padding: {padding}")


def _padding_for(size: int, kernel: int, stride: int, padding: Padding) -> tuple[int, int]:
    if padding == "valid":
        return 0, 0
    out_size = int(np.ceil(size / stride))
    total = max((out_size - 1) * stride + kernel - size, 0)
    before = total // 2
    return before, total - before


def _sliding_windows(
    inputs: np.ndarray,
    kernel_size: tuple[int, int],
    strides: tuple[int, int],
    padding: Padding,
    pad_value: float = 0.0,
) -> np.ndarray:
    batch, height, width, channels = inputs.shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = strides
    pad_top, pad_bottom = _padding_for(height, kernel_h, stride_h, padding)
    pad_left, pad_right = _padding_for(width, kernel_w, stride_w, padding)
    padded = np.pad(
        inputs,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )
    out_h = _output_dim(height, kernel_h, stride_h, padding)
    out_w = _output_dim(width, kernel_w, stride_w, padding)
    windows = np.empty((batch, out_h, out_w, kernel_h, kernel_w, channels), dtype=inputs.dtype)
    for row in range(out_h):
        for col in range(out_w):
            row_start = row * stride_h
            col_start = col * stride_w
            windows[:, row, col] = padded[:, row_start : row_start + kernel_h, col_start : col_start + kernel_w, :]
    return windows


def _window_valid_counts(
    input_shape: tuple[int, int, int, int],
    kernel_size: tuple[int, int],
    strides: tuple[int, int],
    padding: Padding,
) -> np.ndarray:
    _, height, width, channels = input_shape
    valid = np.ones((1, height, width, channels), dtype=np.float32)
    windows = _sliding_windows(valid, kernel_size, strides, padding, pad_value=0.0)
    return np.sum(windows, axis=(3, 4))


class Conv2D(Layer):
    """layer konvolusi channel-last untuk forward propagation lokal"""

    def __init__(
        self,
        filters: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = 1,
        padding: Padding = "valid",
        activation: str | None = None,
        use_bias: bool = True,
        name: str | None = None,
        dtype: np.dtype | str = np.float32,
    ) -> None:
        super().__init__(name=name)
        self.filters = int(filters)
        self.kernel_size = _as_pair(kernel_size, "kernel_size")
        self.strides = _as_pair(strides, "strides")
        self.padding: Padding = padding
        self.activation_name = activation_name(activation)
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.dtype = np.dtype(dtype)
        self.kernel: np.ndarray | None = None
        self.bias: np.ndarray | None = None

    def build(self, input_shape: Sequence[int | None]) -> None:
        if len(input_shape) != 4:
            raise ValueError(f"{self.name} expects input shape (N, H, W, C). Got {tuple(input_shape)}.")
        channels = input_shape[-1]
        if channels is None:
            raise ValueError(f"{self.name} requires known channel dimension.")
        kh, kw = self.kernel_size
        self.kernel = np.zeros((kh, kw, int(channels), self.filters), dtype=self.dtype)
        self.bias = np.zeros((self.filters,), dtype=self.dtype) if self.use_bias else None
        super().build(input_shape)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        array = _validate_channel_last(inputs, self.name)
        if self.kernel is None:
            self.build(array.shape)
        assert self.kernel is not None
        windows = _sliding_windows(array, self.kernel_size, self.strides, self.padding)
        outputs = np.tensordot(windows, self.kernel, axes=([3, 4, 5], [0, 1, 2]))
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

    def _expected_kernel_shape(self, loaded_kernel: np.ndarray) -> tuple[int, int, int, int]:
        kh, kw = self.kernel_size
        channels = loaded_kernel.shape[2] if self.kernel is None else self.kernel.shape[2]
        return kh, kw, channels, self.filters

    def set_weights(self, weights: Sequence[np.ndarray]) -> None:
        validate_weight_count(self.name, weights, 2 if self.use_bias else 1)
        kernel = ensure_numeric_array(weights[0], f"{self.name}.kernel")
        if kernel.ndim != 4:
            raise ValueError(f"{self.name} kernel must be rank 4. Got Keras shape {kernel.shape}.")
        validate_weight_shape(self.name, kernel.shape, self._expected_kernel_shape(kernel), "kernel")
        self.kernel = kernel.astype(self.dtype, copy=True)
        if self.use_bias:
            bias = ensure_numeric_array(weights[1], f"{self.name}.bias")
            validate_weight_shape(self.name, bias.shape, (self.filters,), "bias")
            self.bias = bias.astype(self.dtype, copy=True)
        self.built = True
        self.input_shape = (None, None, None, self.kernel.shape[2])
        self.output_shape = None

    def compute_output_shape(self, input_shape: Sequence[int | None]) -> Shape:
        if len(input_shape) != 4:
            raise ValueError(f"{self.name} expects input shape (N, H, W, C). Got {tuple(input_shape)}.")
        batch, height, width, _ = input_shape
        kh, kw = self.kernel_size
        sh, sw = self.strides
        out_h = None if height is None else _output_dim(int(height), kh, sh, self.padding)
        out_w = None if width is None else _output_dim(int(width), kw, sw, self.padding)
        return (batch, out_h, out_w, self.filters)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "activation": self.activation_name,
                "use_bias": self.use_bias,
                "dtype": self.dtype.name,
            }
        )
        return config


class LocallyConnected2D(Conv2D):
    """layer konvolusi non-shared dengan bobot per posisi output"""

    def __init__(
        self,
        filters: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = 1,
        padding: Padding = "valid",
        activation: str | None = None,
        use_bias: bool = True,
        name: str | None = None,
        dtype: np.dtype | str = np.float32,
    ) -> None:
        if padding != "valid":
            raise ValueError("LocallyConnected2D supports valid padding only.")
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            name=name,
            dtype=dtype,
        )

    def build(self, input_shape: Sequence[int | None]) -> None:
        if len(input_shape) != 4:
            raise ValueError(f"{self.name} expects input shape (N, H, W, C). Got {tuple(input_shape)}.")
        _, height, width, channels = input_shape
        if height is None or width is None or channels is None:
            raise ValueError(f"{self.name} requires known spatial and channel dimensions.")
        out_h, out_w = self.compute_output_shape(input_shape)[1:3]
        kh, kw = self.kernel_size
        assert out_h is not None and out_w is not None
        self.kernel = np.zeros((out_h, out_w, kh, kw, int(channels), self.filters), dtype=self.dtype)
        self.bias = np.zeros((out_h, out_w, self.filters), dtype=self.dtype) if self.use_bias else None
        Layer.build(self, input_shape)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        array = _validate_channel_last(inputs, self.name)
        if self.kernel is None:
            self.build(array.shape)
        assert self.kernel is not None
        windows = _sliding_windows(array, self.kernel_size, self.strides, self.padding)
        outputs = np.einsum("nxyhwc,xyhwcf->nxyf", windows, self.kernel)
        if self.use_bias:
            assert self.bias is not None
            outputs = outputs + self.bias
        return self.activation(outputs)

    def set_weights(self, weights: Sequence[np.ndarray]) -> None:
        validate_weight_count(self.name, weights, 2 if self.use_bias else 1)
        kernel = ensure_numeric_array(weights[0], f"{self.name}.kernel")
        if kernel.ndim == 3:
            output_size, patch_size, filters = kernel.shape
            kh, kw = self.kernel_size
            if filters != self.filters or patch_size % (kh * kw) != 0:
                raise ValueError(f"{self.name} kernel has incompatible Keras shape {kernel.shape}.")
            channels = patch_size // (kh * kw)
            if self.output_shape is not None and self.output_shape[1] is not None and self.output_shape[2] is not None:
                out_h, out_w = int(self.output_shape[1]), int(self.output_shape[2])
            else:
                side = int(np.sqrt(output_size))
                out_h, out_w = side, side
            if out_h * out_w != output_size:
                raise ValueError(f"{self.name} cannot infer output grid from Keras shape {kernel.shape}.")
            kernel = kernel.reshape(out_h, out_w, kh, kw, channels, filters)
        elif kernel.ndim != 6:
            raise ValueError(f"{self.name} kernel must be rank 3 or 6. Got Keras shape {kernel.shape}.")
        validate_weight_shape(self.name, kernel.shape[2:4] + kernel.shape[5:], self.kernel_size + (self.filters,), "kernel")
        self.kernel = kernel.astype(self.dtype, copy=True)
        if self.use_bias:
            bias = ensure_numeric_array(weights[1], f"{self.name}.bias")
            expected_bias = (self.kernel.shape[0], self.kernel.shape[1], self.filters)
            if bias.ndim == 2:
                bias = bias.reshape(expected_bias)
            validate_weight_shape(self.name, bias.shape, expected_bias, "bias")
            self.bias = bias.astype(self.dtype, copy=True)
        self.built = True
        self.input_shape = (None, None, None, self.kernel.shape[4])
        self.output_shape = (None, self.kernel.shape[0], self.kernel.shape[1], self.filters)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["kernel_layout"] = "(out_h, out_w, kernel_h, kernel_w, in_channels, filters)"
        return config


class _Pooling2D(Layer):
    def __init__(
        self,
        pool_size: int | Sequence[int] = 2,
        strides: int | Sequence[int] | None = None,
        padding: Padding = "valid",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.pool_size = _as_pair(pool_size, "pool_size")
        self.strides = _as_pair(strides if strides is not None else pool_size, "strides")
        self.padding = padding

    def _reduce(self, windows: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _pad_value(self, array: np.ndarray) -> float:
        return 0.0

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        array = _validate_channel_last(inputs, self.name)
        windows = _sliding_windows(array, self.pool_size, self.strides, self.padding, pad_value=self._pad_value(array))
        return self._reduce(windows)

    def compute_output_shape(self, input_shape: Sequence[int | None]) -> Shape:
        if len(input_shape) != 4:
            raise ValueError(f"{self.name} expects input shape (N, H, W, C). Got {tuple(input_shape)}.")
        batch, height, width, channels = input_shape
        ph, pw = self.pool_size
        sh, sw = self.strides
        out_h = None if height is None else _output_dim(int(height), ph, sh, self.padding)
        out_w = None if width is None else _output_dim(int(width), pw, sw, self.padding)
        return (batch, out_h, out_w, channels)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update({"pool_size": self.pool_size, "strides": self.strides, "padding": self.padding})
        return config


class MaxPooling2D(_Pooling2D):
    """pooling maksimum untuk tensor channel-last"""

    def _pad_value(self, array: np.ndarray) -> float:
        if self.padding == "valid":
            return 0.0
        if np.issubdtype(array.dtype, np.floating):
            return float(-np.inf)
        return float(np.iinfo(array.dtype).min)

    def _reduce(self, windows: np.ndarray) -> np.ndarray:
        return np.max(windows, axis=(3, 4))


class AveragePooling2D(_Pooling2D):
    """pooling rata-rata untuk tensor channel-last"""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        array = _validate_channel_last(inputs, self.name)
        windows = _sliding_windows(array, self.pool_size, self.strides, self.padding)
        if self.padding == "same":
            counts = _window_valid_counts(array.shape, self.pool_size, self.strides, self.padding)
            return np.sum(windows, axis=(3, 4)) / counts
        return self._reduce(windows)

    def _reduce(self, windows: np.ndarray) -> np.ndarray:
        return np.mean(windows, axis=(3, 4))


class GlobalAveragePooling2D(Layer):
    """pooling rata-rata seluruh area spatial"""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        array = _validate_channel_last(inputs, self.name)
        return np.mean(array, axis=(1, 2))

    def compute_output_shape(self, input_shape: Sequence[int | None]) -> Shape:
        if len(input_shape) != 4:
            raise ValueError(f"{self.name} expects input shape (N, H, W, C). Got {tuple(input_shape)}.")
        return (input_shape[0], input_shape[-1])


class GlobalMaxPooling2D(Layer):
    """pooling maksimum seluruh area spatial"""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        array = _validate_channel_last(inputs, self.name)
        return np.max(array, axis=(1, 2))

    def compute_output_shape(self, input_shape: Sequence[int | None]) -> Shape:
        if len(input_shape) != 4:
            raise ValueError(f"{self.name} expects input shape (N, H, W, C). Got {tuple(input_shape)}.")
        return (input_shape[0], input_shape[-1])


class Flatten(Layer):
    """ubah dimensi non-batch menjadi satu vektor"""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        array = np.asarray(inputs)
        if array.ndim < 2:
            raise ValueError(f"{self.name} expects input rank >= 2. Got {array.shape}.")
        return array.reshape((array.shape[0], -1))

    def compute_output_shape(self, input_shape: Sequence[int | None]) -> Shape:
        if len(input_shape) < 2:
            raise ValueError(f"{self.name} expects input rank >= 2. Got {tuple(input_shape)}.")
        if any(dim is None for dim in input_shape[1:]):
            return (input_shape[0], None)
        return (input_shape[0], int(np.prod(input_shape[1:])))
