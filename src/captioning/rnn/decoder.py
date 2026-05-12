from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.captioning.base import BaseCaptionDecoder
from src.captioning.rnn.layers import SimpleRNNCell
from src.common.layers import Dense, Embedding


class SimpleRNNDecoder(BaseCaptionDecoder):
    """decoder caption simplernn dengan strategi pre-inject"""

    def __init__(
        self,
        feature_dim: int,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        recurrent_layers: int = 1,
        start_token_id: int = 1,
        end_token_id: int = 2,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "SimpleRNNDecoder")
        self.feature_dim = int(feature_dim)
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.hidden_size = int(hidden_size)
        self.recurrent_layers = int(recurrent_layers)
        self.start_token_id = int(start_token_id)
        self.end_token_id = int(end_token_id)
        self.image_projection = Dense(embed_dim, activation="linear", name="image_projection")
        self.embedding = Embedding(vocab_size, embed_dim, name="token_embedding")
        self.cells = [
            SimpleRNNCell(hidden_size, name=f"rnn_{index + 1}")
            for index in range(self.recurrent_layers)
        ]
        self.output_dense = Dense(vocab_size, activation="softmax", name="token_output")

    def initial_state(self, batch_size: int) -> list[np.ndarray]:
        return [cell.initial_state(batch_size) for cell in self.cells]

    def _run_cells(self, inputs: np.ndarray, state: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
        output = inputs
        next_state: list[np.ndarray] = []
        for index, cell in enumerate(self.cells):
            output, cell_state = cell.step(output, state[index])
            next_state.append(cell_state)
        return output, next_state

    def _inject_image(self, image_features: np.ndarray, state: list[np.ndarray]) -> list[np.ndarray]:
        projected = self.image_projection(np.asarray(image_features, dtype=np.float32))
        _, next_state = self._run_cells(projected, state)
        return next_state

    def step(
        self,
        token_ids: np.ndarray,
        state: list[np.ndarray],
        image_features: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        if image_features is not None:
            state = self._inject_image(image_features, state)
        embedded = self.embedding(np.asarray(token_ids, dtype=np.int64))
        if embedded.ndim == 3:
            embedded = embedded[:, -1, :]
        recurrent_output, next_state = self._run_cells(embedded, state)
        return self.output_dense(recurrent_output), next_state

    def forward_sequence(self, image_features: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        features = np.asarray(image_features, dtype=np.float32)
        tokens = np.asarray(token_ids, dtype=np.int64)
        if features.ndim != 2:
            raise ValueError(f"{self.name} expects image_features shape (N, F). Got {features.shape}.")
        if tokens.ndim != 2:
            raise ValueError(f"{self.name} expects token_ids shape (N, T). Got {tokens.shape}.")
        state = self._inject_image(features, self.initial_state(features.shape[0]))
        outputs: list[np.ndarray] = []
        for timestep in range(tokens.shape[1]):
            probabilities, state = self.step(tokens[:, timestep], state)
            outputs.append(probabilities)
        return np.stack(outputs, axis=1)

    def load_keras_weights(self, keras_model: Any) -> None:
        self.image_projection.set_weights(keras_model.get_layer("image_projection").get_weights())
        self.embedding.set_weights(keras_model.get_layer("token_embedding").get_weights())
        for index, cell in enumerate(self.cells):
            cell.set_weights(keras_model.get_layer(f"rnn_{index + 1}").get_weights())
        self.output_dense.set_weights(keras_model.get_layer("token_output").get_weights())
        self.built = True

    def get_weights(self) -> list[np.ndarray]:
        weights: list[np.ndarray] = []
        weights.extend(self.image_projection.get_weights())
        weights.extend(self.embedding.get_weights())
        for cell in self.cells:
            weights.extend(cell.get_weights())
        weights.extend(self.output_dense.get_weights())
        return weights

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "feature_dim": self.feature_dim,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "hidden_size": self.hidden_size,
                "recurrent_layers": self.recurrent_layers,
                "start_token_id": self.start_token_id,
                "end_token_id": self.end_token_id,
            }
        )
        return config

    def parameter_count(self) -> int:
        return int(sum(np.prod(weight.shape) for weight in self.get_weights()))

