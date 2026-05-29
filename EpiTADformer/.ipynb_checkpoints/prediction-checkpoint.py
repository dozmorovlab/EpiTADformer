#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd
import numpy as np
import pickle
import gc
import os
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.models import load_model

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import DBSCAN


# =========================================================
# Transformer Model
# =========================================================
def positional_encoding(sequence_length, feature_dim):

    pos = np.arange(sequence_length)[:, np.newaxis]
    i = np.arange(feature_dim)[np.newaxis, :]

    angle_rates = 1 / np.power(
        10000,
        (2 * (i // 2)) / np.float32(feature_dim)
    )

    angle_rads = pos * angle_rates

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return tf.cast(
        angle_rads[np.newaxis, ...],
        dtype=tf.float32
    )


@register_keras_serializable()
class PositionalEncoding(Layer):

    def __init__(self, sequence_length, feature_dim, **kwargs):

        super().__init__(**kwargs)

        self.sequence_length = sequence_length
        self.feature_dim = feature_dim

    def call(self, inputs):

        pe = positional_encoding(
            self.sequence_length,
            self.feature_dim
        )

        return inputs + pe

    def get_config(self):

        config = super().get_config()

        config.update({
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
        })

        return config


def transformer_encoder(
    inputs,
    head_size,
    num_heads,
    ff_dim,
    dropout=0
):

    x = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_size,
        dropout=dropout
    )(inputs, inputs)

    x = layers.Add()([x, inputs])

    x = layers.LayerNormalization(
        epsilon=1e-6
    )(x)

    ffn = layers.Dense(
        ff_dim,
        activation="relu"
    )(x)

    ffn = layers.Dropout(dropout)(ffn)

    ffn = layers.Dense(
        inputs.shape[-1]
    )(ffn)

    x = layers.Add()([x, ffn])

    x = layers.LayerNormalization(
        epsilon=1e-6
    )(x)

    return x


# =========================================================
# Helper Functions
# =========================================================
def predict_in_chunks(
    model,
    X,
    batch_size=128,
    chunk_size=50000
):

    results = []

    for start in range(0, len(X), chunk_size):

        chunk = X[start:start + chunk_size]

        preds = model.predict(
            chunk,
            batch_size=batch_size,
            verbose=0
        ).flatten()

        results.append(preds)

        gc.collect()

    return np.concatenate(results)


def save_as_bed(df, filename):

    df.iloc[:, :3].to_csv(
        filename,
        sep='\t',
        header=False,
        index=False
    )


def contains_truth(pred_row, truth_df):

    return (
        (truth_df['region_start'] <= pred_row['region_end']) &
        (truth_df['region_end']   >= pred_row['region_start'])
    ).any()


def flag_outliers_per_chrom(group):

    Q1 = group['TAD_size'].quantile(0.25)
    Q3 = group['TAD_size'].quantile(0.75)

    IQR = Q3 - Q1

    upper_fence = Q3 + 3 * IQR

    group = group.copy()

    group['is_outlier'] = (
        group['TAD_size'] > upper_fence
    )

    return group




def read_data_dir(dir_path, target_chrs):

    target_chrs = set(target_chrs)

    chrom_data = {chrom: [] for chrom in target_chrs}

    files = os.listdir(dir_path)

    for f in files:

        file_path = os.path.join(dir_path, f)

        df = pd.read_csv(
            file_path,
            sep="\t",
            header=None,
            usecols=[0, 1, 3],
            names=["chr", "start", "value"],
            dtype={
                "chr": "category",
                "start": "int32",
                "value": "float32"
            }
        )

        df = df[df["chr"].isin(target_chrs)]

        for chrom, subdf in df.groupby("chr", observed=True):

            subdf = subdf[["start", "value"]].copy()
            subdf["file"] = f

            chrom_data[chrom].append(subdf)

    result = {}

    for chrom in target_chrs:

        if len(chrom_data[chrom]) == 0:
            result[chrom] = pd.DataFrame()
            continue

        long_df = pd.concat(chrom_data[chrom], ignore_index=True)

        result[chrom] = (
            long_df
            .pivot(index="file", columns="start", values="value")
            .fillna(0)
            .sort_index(axis=1)
        )

    return result

    


            


# =========================================================
# Main
# =========================================================
def main(
    full_data_dir,
    model_dir,
    save_dir,
    target_chrs,
    min_files=16,
    bin_size=100,
    sequence_len=101,
    save_iteration_models = False
):
    # Generate output path
    os.makedirs(save_dir, exist_ok=True)
    
    if save_iteration_models:
        os.makedirs(
            os.path.join(save_dir, "Iterations_models"),
            exist_ok=True
        )

    mid_indx = sequence_len // 2

    print("\nLoading feature data...")

    #with open(full_data_dir, "rb") as f:
    #    raw_data = pickle.load(f)

    #filtered_chr_dict = {
    #    k: raw_data[k].astype(np.float32)
    #    for k in target_chrs
    #    if k in raw_data
    #}


    filtered_chr_dict = {}

    # read testing chromosome
    filtered_chr_dict = read_data_dir(
        full_data_dir,
        target_chrs
    )


    # del raw_data
    gc.collect()

    # =====================================================
    # Predict boundary regions from all models
    # =====================================================
    model_list = sorted([
        f for f in os.listdir(model_dir)
        if f.endswith(".h5")
    ])

    # store all model regions in memory
    boundary_files = []
    all_model_bins = []

    for model_file in model_list:

        print(f"\n🚀 Processing Model: {model_file}")

        model_path = os.path.join(
            model_dir,
            model_file
        )

        best_model = load_model(model_path)

        all_regions_list = []
        all_bins_list = []

        for chr_name in target_chrs:

            if chr_name not in filtered_chr_dict:
                continue

            print(f"   Analyzing {chr_name}...")

            data_df = filtered_chr_dict[
                chr_name
            ].fillna(0)

            data_values = data_df.values

            # normalize
            mean = data_values.mean()
            std = data_values.std() + 1e-8

            data_normed = (
                data_values - mean
            ) / std

            # create windows
            windows = sliding_window_view(
                data_normed,
                window_shape=sequence_len,
                axis=1
            )

            X_chr = windows.transpose(
                1,
                0,
                2
            ).astype(np.float32)

            n_windows = X_chr.shape[0]

            print(
                f"      X_chr shape: {X_chr.shape}"
            )

            pred_probs = predict_in_chunks(
                best_model,
                X_chr,
                batch_size=128,
                chunk_size=50000
            )

            y_pred = (
                pred_probs > 0.5
            ).astype(np.int8)

            # genomic coordinates
            starts = (
                np.arange(
                    0,
                    n_windows * bin_size,
                    bin_size
                )
                + (mid_indx * bin_size)
            )

            ends = starts + bin_size

            tmp_df = pd.DataFrame({ #bin 
                "chrom": chr_name,
                "start": starts,
                "end": ends,
                "boundary": y_pred
            })

            all_bins_list.append(tmp_df[tmp_df["boundary"] == 1][["chrom", "start", "end"]])
            

            # DBSCAN clustering
            df_boundary = tmp_df[
                tmp_df["boundary"] == 1
            ].copy()

            if not df_boundary.empty:

                db = DBSCAN(
                    eps=201,
                    min_samples=1
                ).fit(
                    df_boundary[["start"]].values
                )

                df_boundary["cluster"] = (
                    db.labels_
                )

                clustered = (
                    df_boundary
                    .groupby("cluster")
                    .agg(
                        chrom=("chrom", "first"),
                        region_start=("start", "min"),
                        region_end=("end", "max")
                    )
                    .reset_index(drop=True)
                )

                all_regions_list.append(
                    clustered
                )

            # cleanup
            del (
                X_chr,
                data_normed,
                pred_probs,
                y_pred,
                tmp_df
            )

            gc.collect()



        # --------------------Save Bin & Iteration models results-----------------------
        # Save individual model    
        model_tag = model_file.replace(".h5", "")
        if save_iteration_models:

            save_as_bed(
                pd.concat(all_bins_list, ignore_index=True),
                f"{save_dir}/Iterations_models/Bins{model_tag}_boundaryBin_ALL.bed"
            )

            save_as_bed(
                pd.concat(all_regions_list, ignore_index=True),
                f"{save_dir}/Iterations_models/Bins{model_tag}_boundaryRegion_ALL.bed"
            )

         # Save model results with other model list       
        if all_bins_list:

            model_bins = pd.concat(
                all_bins_list,
                ignore_index=True
            )
        
            model_bins["model"] = model_file
        
            all_model_bins.append(model_bins)

        if all_regions_list:
    
            model_regions = pd.concat(
                all_regions_list,
                ignore_index=True
            )
        
            boundary_files.append(
                model_regions
            )

    # =====================================================
    # Consensus boundary bins across models
    # =====================================================
    
    if len(all_model_bins) > 0:
    
        print("\nCreating consensus boundary bins...")
    
        all_bins_df = pd.concat(
            all_model_bins,
            ignore_index=True
        )
    
        consensus_bins = (
            all_bins_df
            .groupby(
                ["chrom", "start", "end"],
                as_index=False
            )
            .agg(
                n_models=("model", "nunique")
            )
        )
    
        consensus_bins = consensus_bins[
            consensus_bins["n_models"] >= min_files
        ]
    
        print(
            f"Found {len(consensus_bins):,} consensus bins "
            f"supported by >= {min_files} models."
        )
    
        consensus_bins[
            ["chrom", "start", "end"]
        ].to_csv(
            os.path.join(
                save_dir,
                f"Consensus_BoundaryBins_ALL.bed"
            ),
            sep="\t",
            header=False,
            index=False
        )
    

    
    
    # =====================================================
    # Consensus regions
    # =====================================================
    print("\nBuilding consensus regions...")

    dfs = boundary_files

    all_chroms = sorted(
        set(
            c
            for df in dfs
            for c in df['chrom'].unique()
        )
    )

    consensus_rows = []

    for chrom in all_chroms:

        subs = [
            df[df['chrom'] == chrom]
            .reset_index(drop=True)
            for df in dfs
        ]

        seen = set()

        for i, candidates in enumerate(subs):

            if candidates.empty:
                continue

            others = [
                subs[j]
                for j in range(len(subs))
                if j != i
            ]

            for _, row in candidates.iterrows():

                key = (
                    row['region_start'],
                    row['region_end']
                )

                if key in seen:
                    continue

                overlap_count = 1

                for other_df in others:

                    if other_df.empty:
                        continue

                    if contains_truth(
                        row,
                        other_df
                    ):
                        overlap_count += 1

                if overlap_count >= min_files:

                    consensus_rows.append({
                        'chrom': chrom,
                        'region_start':
                            row['region_start'],
                        'region_end':
                            row['region_end'],
                        'n_files':
                            overlap_count
                    })

                    seen.add(key)

    if len(consensus_rows) == 0:

        print(
            "\nNo consensus regions found."
        )

        return

    result_df = pd.DataFrame(
        consensus_rows
    )

    result_df = result_df.sort_values([
        'chrom',
        'region_start'
    ])

    # =====================================================
    # Merge overlapping regions
    # =====================================================
    merged = []

    for chrom, grp in result_df.groupby(
        'chrom'
    ):

        grp = grp.sort_values(
            'region_start'
        ).reset_index(drop=True)

        cur_start = int(
            grp.loc[0, 'region_start']
        )

        cur_end = int(
            grp.loc[0, 'region_end']
        )

        cur_n = int(
            grp.loc[0, 'n_files']
        )

        for _, r in grp.iloc[1:].iterrows():

            if r['region_start'] <= cur_end:

                cur_end = max(
                    cur_end,
                    int(r['region_end'])
                )

                cur_n = max(
                    cur_n,
                    int(r['n_files'])
                )

            else:

                merged.append([
                    chrom,
                    cur_start,
                    cur_end,
                    cur_n
                ])

                cur_start = int(
                    r['region_start']
                )

                cur_end = int(
                    r['region_end']
                )

                cur_n = int(
                    r['n_files']
                )

        merged.append([
            chrom,
            cur_start,
            cur_end,
            cur_n
        ])

    # =====================================================
    # Remove outliers
    # =====================================================
    consen_reg = pd.DataFrame(
        merged,
        columns=[
            'chrom',
            'region_start',
            'region_end',
            'n_files'
        ]
    )

    consen_reg['TAD_size'] = (
        consen_reg['region_end']
        - consen_reg['region_start']
    )

    consen_reg = (
        consen_reg
        .groupby(
            'chrom',
            group_keys=False
        )
        .apply(flag_outliers_per_chrom)
    )

    TAD_remaining = consen_reg[
        ~consen_reg['is_outlier']
    ]

    # =====================================================
    # Save final BED
    # =====================================================
    final_output = os.path.join(
        save_dir,
        "Consensus_Filtered_BoundaryRegion_All.bed"
    )

    save_as_bed(
        TAD_remaining,
        final_output
    )

    print(f"\n✅ Final saved: {final_output}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Predict TAD boundaries using EpiTADformer"
    )

    parser.add_argument(
        "--full_data_dir",
        required=True,
        help="Directory containing epigenomic signal files"
    )

    parser.add_argument(
        "--model_dir",
        required=True,
        help="Directory containing trained models"
    )

    parser.add_argument(
        "--save_dir",
        required=True,
        help="Directory to save prediction results"
    )

    parser.add_argument(
        "--target_chrs",
        nargs="+",
        required=True,
        help="Chromosomes to analyze"
    )

    parser.add_argument(
        "--min_files",
        type=int,
        default=16
    )

    parser.add_argument(
        "--bin_size",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--sequence_len",
        type=int,
        default=101,
        help="Directory containing epigenomic signal files"
    )

    parser.add_argument(
        "--save_iteration_models",
        action="store_true",
        help="Save boundary bins and regions from each individual model"
    )
    
    args = parser.parse_args()

    # sequence length must be odd
    if args.sequence_len % 2 == 0:
        raise ValueError(
            "sequence_len must be odd"
        )


    main(
        full_data_dir=args.full_data_dir,
        model_dir=args.model_dir,
        save_dir=args.save_dir,
        target_chrs=args.target_chrs,
        min_files=args.min_files,
        bin_size=args.bin_size,
        sequence_len=args.sequence_len,
        save_iteration_models=args.save_iteration_models
    )