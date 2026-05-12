from __future__ import annotations


def build_lstm_decoder(
    feature_dim: int,
    vocab_size: int,
    max_sequence_len: int,
    embed_dim: int,
    hidden_size: int,
    recurrent_layers: int = 1,
    dropout: float = 0.0,
    name: str = "lstm_caption_decoder",
):
    """membuat decoder lstm pre-inject berbasis keras"""
    import tensorflow as tf

    image_input = tf.keras.layers.Input(shape=(feature_dim,), name="image_features")
    token_input = tf.keras.layers.Input(shape=(max_sequence_len,), dtype="int32", name="token_ids")
    image_projection = tf.keras.layers.Dense(embed_dim, activation="linear", name="image_projection")(image_input)
    image_step = tf.keras.layers.Reshape((1, embed_dim), name="image_step")(image_projection)
    token_embedding = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=False, name="token_embedding")(token_input)
    sequence = tf.keras.layers.Concatenate(axis=1, name="pre_inject_sequence")([image_step, token_embedding])
    outputs = sequence
    for index in range(recurrent_layers):
        outputs = tf.keras.layers.LSTM(
            hidden_size,
            return_sequences=True,
            dropout=dropout,
            activation="tanh",
            recurrent_activation="sigmoid",
            name=f"lstm_{index + 1}",
        )(outputs)
    caption_outputs = tf.keras.layers.Lambda(lambda value: value[:, 1:, :], name="drop_image_timestep")(outputs)
    probabilities = tf.keras.layers.Dense(vocab_size, activation="softmax", name="token_output")(caption_outputs)
    return tf.keras.Model(inputs=[image_input, token_input], outputs=probabilities, name=name)
