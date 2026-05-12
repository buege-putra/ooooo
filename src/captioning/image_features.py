from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.cnn.data import load_image, load_image_batch


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
    """menjalankan preprocessing sesuai encoder yang dipakai

    args:
        already_normalized: set True jika nilai piksel sudah berada di rentang [0, 1],
            sehingga perlu diskalakan kembali ke [0, 255] sebelum preprocessing encoder.
            jika False (default), input diasumsikan sudah berada di rentang [0, 255].
    """
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
    features: list[np.ndarray] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = load_image_batch(batch_paths, target_size=target_size, normalize=True)
        prepared = preprocess_image_array(images, encoder=encoder, already_normalized=True)
        batch_features = encoder_model.predict(prepared, verbose=0)
        features.append(np.asarray(batch_features))
    if not features:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(features, axis=0).astype(np.float32, copy=False)


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
    features = extract_features(image_paths, model, target_size=size, encoder=encoder, batch_size=batch_size)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, features)
    return features


def load_preprocessed_image(path: str | Path, target_size: tuple[int, int], encoder: EncoderName = "inception_v3") -> np.ndarray:
    """memuat satu image dan menyiapkannya untuk encoder"""
    image = load_image(path, target_size=target_size, normalize=True)
    return preprocess_image_array(np.expand_dims(image, axis=0), encoder=encoder, already_normalized=True)
