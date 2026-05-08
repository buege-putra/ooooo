from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Sequence

import numpy as np


Shape = tuple[int | None, ...]


class WeightLoadable(ABC):
    """mixin untuk objek yang bisa menerima assignment weight"""

    @abstractmethod
    def get_weights(self) -> list[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def set_weights(self, weights: Sequence[np.ndarray]) -> None:
        raise NotImplementedError


class Layer(WeightLoadable):
    """kontrak minimal layer untuk forward propagation numpy"""

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__
        self.built = False
        self.input_shape: Shape | None = None
        self.output_shape: Shape | None = None

    def build(self, input_shape: Sequence[int | None]) -> None:
        self.input_shape = tuple(input_shape)
        self.output_shape = self.compute_output_shape(self.input_shape)
        self.built = True

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        if not self.built:
            self.build(inputs.shape)
        return self.forward(inputs)

    def get_weights(self) -> list[np.ndarray]:
        return []

    def set_weights(self, weights: Sequence[np.ndarray]) -> None:
        if weights:
            raise ValueError(f"{self.name} expects no weights. Got {len(weights)} arrays.")

    def compute_output_shape(self, input_shape: Sequence[int | None]) -> Shape:
        return tuple(input_shape)

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name, "class_name": self.__class__.__name__}

    def parameter_count(self) -> int:
        return int(sum(np.prod(weight.shape) for weight in self.get_weights()))


class Model(WeightLoadable):
    """model dasar dengan interface prediksi"""

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__
        self.built = False

    def build(self, input_shape: Sequence[int | None]) -> None:
        self.built = True

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        if not self.built:
            self.build(inputs.shape)
        return self.forward(inputs)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self(inputs)

    def get_weights(self) -> list[np.ndarray]:
        return []

    def set_weights(self, weights: Sequence[np.ndarray]) -> None:
        if weights:
            raise ValueError(f"{self.name} expects no weights. Got {len(weights)} arrays.")

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name, "class_name": self.__class__.__name__}

    def parameter_count(self) -> int:
        return int(sum(np.prod(weight.shape) for weight in self.get_weights()))


class Sequential(Model):
    """komposisi sequential untuk layer lokal"""

    def __init__(self, layers: Iterable[Layer] | None = None, name: str | None = None) -> None:
        super().__init__(name=name or "Sequential")
        self.layers: list[Layer] = list(layers or [])
        self.input_shape: Shape | None = None
        self.output_shape: Shape | None = None

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)
        self.built = False

    def build(self, input_shape: Sequence[int | None]) -> None:
        current_shape: Shape = tuple(input_shape)
        self.input_shape = current_shape
        for layer in self.layers:
            if not layer.built:
                layer.build(current_shape)
            current_shape = layer.compute_output_shape(current_shape)
        self.output_shape = current_shape
        self.built = True

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def get_weights(self) -> list[np.ndarray]:
        weights: list[np.ndarray] = []
        for layer in self.layers:
            weights.extend(layer.get_weights())
        return weights

    def set_weights(self, weights: Sequence[np.ndarray]) -> None:
        offset = 0
        for layer in self.layers:
            expected = len(layer.get_weights())
            layer.set_weights(weights[offset : offset + expected])
            offset += expected
        if offset != len(weights):
            raise ValueError(f"{self.name} received {len(weights)} arrays but used {offset}.")

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["layers"] = [layer.get_config() for layer in self.layers]
        return config

    def parameter_count(self) -> int:
        return int(sum(layer.parameter_count() for layer in self.layers))
