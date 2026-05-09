from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.common.activations import softmax
from src.common.base import Layer, Sequential
from src.common.layers import Dense
from src.cnn.layers import (
    AveragePooling2D,
    Conv2D,
    Flatten,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    LocallyConnected2D,
    MaxPooling2D,
)


class CNNModel(Sequential):
    """komposisi sequential untuk model cnn numpy"""

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        outputs = self(np.asarray(inputs))
        if outputs.ndim != 2:
            raise ValueError(f"{self.name} expects classifier output rank 2. Got {outputs.shape}.")
        row_sums = np.sum(outputs, axis=-1)
        if np.all(outputs >= 0) and np.allclose(row_sums, 1.0, atol=1e-5):
            return outputs
        return softmax(outputs, axis=-1)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(inputs), axis=-1)

    def forward_single(self, image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"{self.name} expects single image shape (H, W, C). Got {array.shape}.")
        return self.predict_proba(array[np.newaxis, ...])[0]

    def load_keras_weights(self, keras_model: Any) -> None:
        load_keras_weights(self, keras_model)


def _activation_name(keras_layer: Any) -> str:
    activation = getattr(keras_layer, "activation", None)
    return getattr(activation, "__name__", "linear")


def layer_from_keras(keras_layer: Any) -> Layer | None:
    class_name = keras_layer.__class__.__name__
    name = getattr(keras_layer, "name", None)
    config = keras_layer.get_config() if hasattr(keras_layer, "get_config") else {}
    if class_name in {"InputLayer", "Dropout"}:
        return None
    if class_name == "Conv2D":
        return Conv2D(
            filters=int(config["filters"]),
            kernel_size=tuple(config["kernel_size"]),
            strides=tuple(config["strides"]),
            padding=config.get("padding", "valid"),
            activation=_activation_name(keras_layer),
            use_bias=bool(config.get("use_bias", True)),
            name=name,
        )
    if class_name == "LocallyConnected2D":
        return LocallyConnected2D(
            filters=int(config["filters"]),
            kernel_size=tuple(config["kernel_size"]),
            strides=tuple(config.get("strides", (1, 1))),
            activation=_activation_name(keras_layer),
            use_bias=bool(config.get("use_bias", True)),
            name=name,
        )
    if class_name == "MaxPooling2D":
        return MaxPooling2D(
            pool_size=tuple(config["pool_size"]),
            strides=tuple(config["strides"]),
            padding=config.get("padding", "valid"),
            name=name,
        )
    if class_name == "AveragePooling2D":
        return AveragePooling2D(
            pool_size=tuple(config["pool_size"]),
            strides=tuple(config["strides"]),
            padding=config.get("padding", "valid"),
            name=name,
        )
    if class_name == "GlobalAveragePooling2D":
        return GlobalAveragePooling2D(name=name)
    if class_name == "GlobalMaxPooling2D":
        return GlobalMaxPooling2D(name=name)
    if class_name == "Flatten":
        return Flatten(name=name)
    if class_name == "Dense":
        return Dense(
            units=int(config["units"]),
            activation=_activation_name(keras_layer),
            use_bias=bool(config.get("use_bias", True)),
            name=name,
        )
    raise ValueError(f"Unsupported Keras layer for CNNModel: {class_name}")


def from_keras_model(keras_model: Any, name: str | None = None) -> CNNModel:
    layers: list[Layer] = []
    for keras_layer in keras_model.layers:
        local_layer = layer_from_keras(keras_layer)
        if local_layer is not None:
            layers.append(local_layer)
    model = CNNModel(layers=layers, name=name or getattr(keras_model, "name", "CNNModel"))
    build_shape = getattr(keras_model, "input_shape", None)
    if isinstance(build_shape, tuple) and build_shape:
        model.build(build_shape)
    load_keras_weights(model, keras_model)
    return model


def _is_local_weight_layer(layer: Layer) -> bool:
    weighted_layer_names = {"Conv2D", "LocallyConnected2D", "Dense"}
    return bool(layer.get_weights()) or layer.__class__.__name__ in weighted_layer_names


def _is_keras_weight_layer(layer: Any) -> bool:
    return hasattr(layer, "get_weights") and bool(layer.get_weights())


def load_keras_weights(model: CNNModel, keras_model: Any) -> None:
    local_layers = [layer for layer in model.layers if _is_local_weight_layer(layer)]
    keras_layers = [layer for layer in keras_model.layers if _is_keras_weight_layer(layer)]
    if len(local_layers) != len(keras_layers):
        raise ValueError(
            f"{model.name} weight layer count mismatch. Got Keras layers {len(keras_layers)}. "
            f"Expected local layers {len(local_layers)}."
        )
    for local_layer, keras_layer in zip(local_layers, keras_layers):
        try:
            local_layer.set_weights(keras_layer.get_weights())
        except ValueError as exc:
            raise ValueError(f"Failed loading weights for layer {local_layer.name}: {exc}") from exc


def build_cnn_model(layers: Sequence[Layer], name: str = "CNNModel") -> CNNModel:
    return CNNModel(layers=layers, name=name)
