from __future__ import annotations

from collections.abc import Sequence
from typing import Literal


PoolingType = Literal["max", "average"]
HeadType = Literal["flatten", "global_average", "global_max"]


def _as_pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        pair = (value, value)
    else:
        pair = tuple(value)
    if len(pair) != 2 or pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} must be a positive int or pair of positive ints. Got {value}.")
    return int(pair[0]), int(pair[1])


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
    activation: str = "relu",
    output_activation: str = "softmax",
    name: str = "locally_connected_cnn",
):
    """buat model cnn keras dengan locallyconnected2d non-shared"""

    from tensorflow import keras

    if not hasattr(keras.layers, "LocallyConnected2D"):
        raise RuntimeError("This TensorFlow/Keras installation does not provide LocallyConnected2D.")

    _validate_classifier_args(num_classes=num_classes, conv_layers=conv_layers)
    filter_values = _resolve_filter_values(filters, conv_layers)
    model = keras.Sequential(name=name)
    model.add(keras.layers.Input(shape=tuple(input_shape)))
    kernel_pair = _as_pair(kernel_size, "kernel_size")
    for index, filter_count in enumerate(filter_values):
        model.add(
            keras.layers.LocallyConnected2D(
                filters=filter_count,
                kernel_size=kernel_pair,
                activation=activation,
                name=f"locally_connected2d_{index + 1}",
            )
        )
        _add_pooling(keras.layers, model, pooling)

    _add_head(keras.layers, model, head)
    if dense_units is not None:
        model.add(keras.layers.Dense(int(dense_units), activation=activation, name="dense_hidden"))
    model.add(keras.layers.Dense(num_classes, activation=output_activation, name="classifier"))
    return model
