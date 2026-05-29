#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import AUC, Precision, Recall

# -----------------------------
# Transformer functions
# -----------------------------
def positional_encoding(sequence_length, feature_dim):
    pos = np.arange(sequence_length)[:, np.newaxis]
    i = np.arange(feature_dim)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(feature_dim))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim

    def call(self, inputs):
        pe = positional_encoding(self.sequence_length, self.feature_dim)
        return inputs + pe

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size,
        dropout=dropout
    )(inputs, inputs)

    x = layers.Add()([x, inputs])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    ffn = layers.Dense(ff_dim, activation="relu")(x)
    ffn = layers.Dropout(dropout)(ffn)
    ffn = layers.Dense(inputs.shape[-1])(ffn)

    x = layers.Add()([x, ffn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    return x

def build_model(input_shape):

    inputs = layers.Input(shape=input_shape)

    x = layers.Permute((2, 1))(inputs)

    x = PositionalEncoding(
        x.shape[1],
        x.shape[2]
    )(x)

    x = transformer_encoder(
        x,
        head_size=64,
        num_heads=2,
        ff_dim=128,
        dropout=0.0
    )

    x = layers.GlobalAveragePooling1D()(x)

    for dim in [448, 264, 128, 64]:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs)


def read_data_dir(dir_path, target_chrs):

    long_data = []

    files = os.listdir(dir_path)

    for f in files:

        file_path = os.path.join(dir_path, f)
        df = pd.read_csv(file_path, sep="\t",header=None, usecols=[0, 1, 2, 3])
        df.columns = ["chr", "start", "end", "value"]
        # Convert start to numeric
        df["start"] = pd.to_numeric(df["start"], errors="coerce")
        # Drop invalid rows
        df = df.dropna(subset=["start"])
        # Select chromosome(s)
        if isinstance(target_chrs, str):
            df_chr = df[df["chr"] == target_chrs]
        else:
            df_chr = df[df["chr"].isin(target_chrs)]

        # Track file name
        df_chr["file"] = f
        long_data.append(df_chr)

    # Combine all files
    long_df = pd.concat(long_data, ignore_index=True)

    # Pivot table
    result = (
        long_df
        .pivot(index="file", columns="start", values="value")
        .fillna(0)
    )

    # Sort columns numerically
    result = result.reindex(sorted(result.columns), axis=1)

    return result


# -----------------------------
# Main function
# -----------------------------
def main(data_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    file_list = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".pkl")
    ])

    for i, file_name in enumerate(file_list):

        print(f"\nDataset {i+1}: {file_name}")

        file_path = os.path.join(data_dir, file_name)

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        X_train = data["X_train"]
        y_train = data["y_train"]

        X_val = data["X_val"]
        y_val = data["y_val"]

        input_shape = (
            X_train.shape[1],
            X_train.shape[2]
        )

        model = build_model(input_shape)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                AUC(name="auc"),
                Precision(name="precision"),
                Recall(name="recall")
            ]
        )

        model_file = os.path.join(
            output_dir,
            f"best_transformer_{i+1}.h5"
        )

        checkpoint_cb = ModelCheckpoint(
            model_file,
            save_best_only=True,
            monitor="val_loss",
            mode="min"
        )

        earlystop_cb = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[checkpoint_cb, earlystop_cb],
            verbose=0
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Trained EpiTADformer models"
    )

    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing training pickle files"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save trained models"
    )

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )