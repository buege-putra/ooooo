from __future__ import annotations

from collections.abc import Sequence
from typing import Literal


PoolingType = Literal["max", "average"]
HeadType = Literal["flatten", "global_average", "global_max"]


class LocallyConnected2DLayer:
    """Layer Keras untuk koneksi lokal 2D tanpa parameter sharing."""

    def __new__(
        cls,
        filters: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = 1,
        padding: str = "valid",
        activation: str | None = None,
        use_bias: bool = True,
        name: str | None = None,
    ):
        import tensorflow as tf
        from tensorflow import keras

        class _LocallyConnected2DLayer(keras.layers.Layer):
            def __init__(self) -> None:
                super().__init__(name=name)
                self.filters = int(filters)
                self.kernel_size = _as_pair(kernel_size, "kernel_size")
                self.strides = _as_pair(strides, "strides")
                self.padding = _validate_padding(padding)
                self.activation = keras.activations.get(activation)
                self.activation_name = keras.activations.serialize(self.activation)
                self.use_bias = bool(use_bias)
                self.kernel = None
                self.bias = None

            def build(self, input_shape):
                height, width, channels = input_shape[1], input_shape[2], input_shape[3]
                if height is None or width is None or channels is None:
                    raise ValueError("LocallyConnected2D requires known H, W, and C.")
                kh, kw = self.kernel_size
                sh, sw = self.strides
                out_h = _output_dim(int(height), kh, sh, self.padding)
                out_w = _output_dim(int(width), kw, sw, self.padding)
                if out_h <= 0 or out_w <= 0:
                    raise ValueError("kernel_size is larger than the input spatial dimensions.")
                self.kernel = self.add_weight(
                    name="kernel",
                    shape=(out_h, out_w, kh, kw, int(channels), self.filters),
                    initializer="glorot_uniform",
                    trainable=True,
                )
                if self.use_bias:
                    self.bias = self.add_weight(
                        name="bias",
                        shape=(out_h, out_w, self.filters),
                        initializer="zeros",
                        trainable=True,
                    )
                super().build(input_shape)

            def call(self, inputs):
                kh, kw = self.kernel_size
                sh, sw = self.strides
                patches = tf.image.extract_patches(
                    images=inputs,
                    sizes=[1, kh, kw, 1],
                    strides=[1, sh, sw, 1],
                    rates=[1, 1, 1, 1],
                    padding=self.padding.upper(),
                )
                input_channels = tf.shape(inputs)[-1]
                patches = tf.reshape(
                    patches,
                    [
                        tf.shape(inputs)[0],
                        tf.shape(patches)[1],
                        tf.shape(patches)[2],
                        kh,
                        kw,
                        input_channels,
                    ],
                )
                outputs = tf.einsum("nxyhwc,xyhwcf->nxyf", patches, self.kernel)
                if self.use_bias:
                    outputs = outputs + self.bias
                if self.activation is not None:
                    outputs = self.activation(outputs)
                return outputs

            def compute_output_shape(self, input_shape):
                kh, kw = self.kernel_size
                sh, sw = self.strides
                out_h = None if input_shape[1] is None else _output_dim(int(input_shape[1]), kh, sh, self.padding)
                out_w = None if input_shape[2] is None else _output_dim(int(input_shape[2]), kw, sw, self.padding)
                return (input_shape[0], out_h, out_w, self.filters)

            def get_config(self):
                config = super().get_config()
                config.update(
                    {
                        "filters": self.filters,
                        "kernel_size": self.kernel_size,
                        "strides": self.strides,
                        "padding": self.padding,
                        "activation": self.activation_name,
                        "use_bias": self.use_bias,
                    }
                )
                return config

        return _LocallyConnected2DLayer()


def _as_pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        pair = (value, value)
    else:
        pair = tuple(value)
    if len(pair) != 2 or pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} must be a positive int or pair of positive ints. Got {value}.")
    return int(pair[0]), int(pair[1])


def _validate_padding(padding: str) -> str:
    value = str(padding).lower()
    if value not in {"valid", "same"}:
        raise ValueError(f"Unsupported padding: {padding}")
    return value


def _output_dim(size: int, kernel: int, stride: int, padding: str) -> int:
    if padding == "same":
        import math

        return int(math.ceil(size / stride))
    return (size - kernel) // stride + 1


def _validate_classifier_args(num_classes: int, conv_layers: int) -> None:
    if conv_layers <= 0:
        raise ValueError(f"conv_layers must be positive. Got {conv_layers}.")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive. Got {num_classes}.")


def _resolve_filter_values(filters: int | Sequence[int], conv_layers: int) -> list[int]:
    values = [int(filters)] * conv_layers if isinstance(filters, int) else [int(value) for value in filters]
    if len(values) != conv_layers or any(value <= 0 for value in values):
        raise ValueError("filters must be a positive int or one value per convolution layer.")
    return values


def _add_pooling(layers: object, model: object, pooling: PoolingType) -> None:
    if pooling == "max":
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    elif pooling == "average":
        model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")


def _add_head(layers: object, model: object, head: HeadType) -> None:
    if head == "flatten":
        model.add(layers.Flatten())
    elif head == "global_average":
        model.add(layers.GlobalAveragePooling2D())
    elif head == "global_max":
        model.add(layers.GlobalMaxPooling2D())
    else:
        raise ValueError(f"Unsupported head: {head}")


def build_conv_cnn(
    input_shape: Sequence[int],
    num_classes: int,
    conv_layers: int = 2,
    filters: int | Sequence[int] = 32,
    kernel_size: int | Sequence[int] = 3,
    pooling: PoolingType = "max",
    head: HeadType = "flatten",
    dense_units: int | None = None,
    dropout_rate: float | None = None,
    activation: str = "relu",
    output_activation: str = "softmax",
    padding: str = "same",
    name: str = "conv_cnn",
):
    """buat model cnn keras dengan conv2d shared-parameter"""

    from tensorflow import keras

    _validate_classifier_args(num_classes=num_classes, conv_layers=conv_layers)
    filter_values = _resolve_filter_values(filters, conv_layers)
    model = keras.Sequential(name=name)
    model.add(keras.layers.Input(shape=tuple(input_shape)))
    kernel_pair = _as_pair(kernel_size, "kernel_size")
    for index, filter_count in enumerate(filter_values):
        model.add(
            keras.layers.Conv2D(
                filters=filter_count,
                kernel_size=kernel_pair,
                padding=padding,
                activation=activation,
                name=f"conv2d_{index + 1}",
            )
        )
        _add_pooling(keras.layers, model, pooling)

    _add_head(keras.layers, model, head)
    if dense_units is not None:
        model.add(keras.layers.Dense(int(dense_units), activation=activation, name="dense_hidden"))
    if dropout_rate is not None and dropout_rate > 0:
        model.add(keras.layers.Dropout(float(dropout_rate), name="dropout"))
    model.add(keras.layers.Dense(num_classes, activation=output_activation, name="classifier"))
    return model


def build_locally_connected_cnn(
    input_shape: Sequence[int],
    num_classes: int,
    conv_layers: int = 1,
    filters: int | Sequence[int] = 32,
    kernel_size: int | Sequence[int] = 3,
    pooling: PoolingType = "max",
    head: HeadType = "flatten",
    dense_units: int | None = None,
    dropout_rate: float | None = None,
    activation: str = "relu",
    output_activation: str = "softmax",
    padding: str = "same",
    name: str = "locally_connected_cnn",
):
    """buat model cnn keras dengan locallyconnected2d non-shared"""

    from tensorflow import keras

    _validate_classifier_args(num_classes=num_classes, conv_layers=conv_layers)
    filter_values = _resolve_filter_values(filters, conv_layers)
    padding = _validate_padding(padding)
    lc2d_layer = LocallyConnected2DLayer
    model = keras.Sequential(name=name)
    model.add(keras.layers.Input(shape=tuple(input_shape)))
    kernel_pair = _as_pair(kernel_size, "kernel_size")
    for index, filter_count in enumerate(filter_values):
        model.add(
            lc2d_layer(
                filters=filter_count,
                kernel_size=kernel_pair,
                padding=padding,
                activation=activation,
                name=f"locally_connected2d_{index + 1}",
            )
        )
        _add_pooling(keras.layers, model, pooling)

    _add_head(keras.layers, model, head)
    if dense_units is not None:
        model.add(keras.layers.Dense(int(dense_units), activation=activation, name="dense_hidden"))
    if dropout_rate is not None and dropout_rate > 0:
        model.add(keras.layers.Dropout(float(dropout_rate), name="dropout"))
    model.add(keras.layers.Dense(num_classes, activation=output_activation, name="classifier"))
    return model
