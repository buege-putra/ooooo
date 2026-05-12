from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.common.base import Model, WeightLoadable


class RecurrentCell(WeightLoadable):
    """kontrak cell recurrent untuk forward satu timestep"""

    def __init__(self, units: int, name: str | None = None, dtype: np.dtype | str = np.float32) -> None:
        self.units = int(units)
        self.name = name or self.__class__.__name__
        self.dtype = np.dtype(dtype)
        self.built = False
        self.input_dim: int | None = None

    def build(self, input_dim: int) -> None:
        self.input_dim = int(input_dim)
        self.built = True

    @property
    @abstractmethod
    def weight_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def initial_state(self, batch_size: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def step(self, inputs: np.ndarray, state: Any) -> tuple[np.ndarray, Any]:
        raise NotImplementedError

    def forward_sequence(self, inputs: np.ndarray, initial_state: Any | None = None) -> tuple[np.ndarray, Any]:
        if inputs.ndim != 3:
            raise ValueError(f"{self.name} expects input shape (N, T, E). Got {inputs.shape}.")
        state = self.initial_state(inputs.shape[0]) if initial_state is None else initial_state
        outputs: list[np.ndarray] = []
        for timestep in range(inputs.shape[1]):
            output, state = self.step(inputs[:, timestep, :], state)
            outputs.append(output)
        return np.stack(outputs, axis=1), state

    def get_config(self) -> dict[str, Any]:
        return {"name": self.name, "class_name": self.__class__.__name__, "units": self.units, "dtype": self.dtype.name}

    def parameter_count(self) -> int:
        return int(sum(np.prod(weight.shape) for weight in self.get_weights()))


class BaseCaptionDecoder(Model):
    """kontrak decoder caption dengan input feature image dan token"""

    start_token_id: int
    end_token_id: int

    @abstractmethod
    def initial_state(self, batch_size: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def step(self, token_ids: np.ndarray, state: Any, image_features: np.ndarray | None = None) -> tuple[np.ndarray, Any]:
        raise NotImplementedError

    @abstractmethod
    def forward_sequence(self, image_features: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def load_keras_weights(self, keras_model: Any) -> None:
        raise NotImplementedError

    def predict_next_token(self, image_features: np.ndarray, token_ids: np.ndarray, state: Any | None = None) -> tuple[np.ndarray, Any]:
        # image hanya di-inject pada langkah pertama (saat state belum ada)
        if state is None:
            state = self.initial_state(np.asarray(token_ids).shape[0])
            return self.step(token_ids, state, image_features=image_features)
        return self.step(token_ids, state)

    def forward(self, inputs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        image_features, token_ids = inputs
        return self.forward_sequence(image_features, token_ids)

    def __call__(self, inputs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return self.forward(inputs)

    def predict(self, inputs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return self.forward(inputs)

    def set_weights(self, weights: Sequence[np.ndarray]) -> None:
        raise NotImplementedError(f"{self.name} uses load_keras_weights for structured decoder weights.")
