from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.cnn.data import extract_and_save_features as _extract_and_save, extract_features as _extract, load_image, load_image_batch


EncoderName = Literal["inception_v3", "vgg16"]


def build_frozen_encoder(encoder: EncoderName = "inception_v3", pooling: str = "avg") -> Any:
    """membuat encoder image pretrained dalam mode frozen"""
    import tensorflow as tf

    if encoder == "inception_v3":
        model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", pooling=pooling)
    elif encoder == "vgg16":
        model = tf.keras.applications.VGG16(include_top=False, weights="imagenet", pooling=pooling)
    else:
        raise ValueError(f"Unsupported encoder: {encoder}")
    model.trainable = False
    return model


def preprocess_image_array(images: np.ndarray, encoder: EncoderName = "inception_v3", already_normalized: bool = False) -> np.ndarray:
    """menjalankan preprocessing sesuai encoder yang dipakai"""
    import tensorflow as tf

    values = np.asarray(images, dtype=np.float32)
    if already_normalized:
        values = values * 255.0
    if encoder == "inception_v3":
        return tf.keras.applications.inception_v3.preprocess_input(values)
    if encoder == "vgg16":
        return tf.keras.applications.vgg16.preprocess_input(values)
    raise ValueError(f"Unsupported encoder: {encoder}")


def extract_features(
    image_paths: Sequence[str | Path],
    encoder_model: Any,
    target_size: tuple[int, int] = (299, 299),
    encoder: EncoderName = "inception_v3",
    batch_size: int = 32,
) -> np.ndarray:
    """mengekstraksi feature vector image dalam batch kecil"""

    def _preprocess(images: np.ndarray) -> np.ndarray:
        return preprocess_image_array(images, encoder=encoder, already_normalized=True)

    return _extract(image_paths, encoder_model, target_size=target_size, batch_size=batch_size, preprocess_fn=_preprocess)


def extract_and_save_features(
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    encoder_model: Any | None = None,
    encoder: EncoderName = "inception_v3",
    target_size: tuple[int, int] | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """mengekstraksi feature lalu menyimpannya sebagai file npy"""
    model = build_frozen_encoder(encoder) if encoder_model is None else encoder_model
    size = target_size or ((299, 299) if encoder == "inception_v3" else (224, 224))

    def _preprocess(images: np.ndarray) -> np.ndarray:
        return preprocess_image_array(images, encoder=encoder, already_normalized=True)

    return _extract_and_save(image_paths, output_path, model, target_size=size, batch_size=batch_size, preprocess_fn=_preprocess)


def load_preprocessed_image(path: str | Path, target_size: tuple[int, int], encoder: EncoderName = "inception_v3") -> np.ndarray:
    """memuat satu image dan menyiapkannya untuk encoder"""
    image = load_image(path, target_size=target_size, normalize=True)
    return preprocess_image_array(np.expand_dims(image, axis=0), encoder=encoder, already_normalized=True)
