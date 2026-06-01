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
def main():
    parser = argparse.ArgumentParser(
        description="Prepare training datasets for EpiTADformer"
    )
    parser.add_argument("--full_data_dir", required=True)
    parser.add_argument("--truth_dir", required=True)
    parser.add_argument("--peak_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--target_chrs", nargs="+", required=True)
    parser.add_argument("--bin_size", type=int, default=100)
    parser.add_argument("--sequence_len", type=int, default=101)
    parser.add_argument("--num_iterations", type=int, default=20)

    args = parser.parse_args()

    if args.sequence_len % 2 == 0:
        raise ValueError("sequence_len must be odd")

    run(
        full_data_dir=args.full_data_dir,
        truth_dir=args.truth_dir,
        peak_dir=args.peak_dir,
        save_dir=args.save_dir,
        target_chrs=args.target_chrs,
        bin_size=args.bin_size,
        sequence_len=args.sequence_len,
        num_iterations=args.num_iterations
    )

if __name__ == "__main__":
    main()