
import argparse
import pandas as pd
import numpy as np
import random
import pickle
import gc
import os

from numpy.lib.stride_tricks import sliding_window_view


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




import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def prepare_training_data(PTBP_expand, data, overlap_Peak,
                          bin_size=100,
                          sequenc_len=51,
                          random_state=12,
                          test_size=0.2,
                          val_size=0.2,
                          shuffle=True):
    """
    Prepares training data (positive and negative samples) for CNN model.

    Parameters:
    - PTBP_expand: DataFrame with true boundaries (start coordinates in col 1)
    - data: DataFrame with shape (features x bins), where columns are bin indices
    - bin_size: Size of one bin (e.g., 100 bp)
    - sequenc_len: Total number of bins in a window (odd number only)
    - random_state: Seed for reproducibility
    - test_size: Proportion for validation/test split (not returned here but used to reduce data)
    - shuffle: Whether to shuffle final training data

    Returns:
    - X_train: Numpy array of shape (n_samples, features, sequence_len)
    - y_train: Corresponding binary labels
    - X_val, y_val, X_test, y_test: validation and testing splits
    """

    np.random.seed(random_state)
    mid_indx = max(sequenc_len // 2, 0)
    num_true = len(PTBP_expand)

    # ----- Positive (boundary) sequences -----
    boundary_indices = (PTBP_expand.iloc[:, 1].values // bin_size).astype(int)
    overlapPeak_indices =  (overlap_Peak.iloc[:, 1].values // bin_size).astype(int)
    
    # Filter out indices that are too close to the edges
    valid_mask = (boundary_indices >= mid_indx) & (boundary_indices < data.shape[1] - mid_indx)
    valid_peak_mask = (overlapPeak_indices >= mid_indx) & (overlapPeak_indices < data.shape[1] - mid_indx)
    if np.sum(~valid_mask) > 0:
        print(f"⚠️ Skipping {np.sum(~valid_mask)} boundary indices near edges.")
    boundary_indices = boundary_indices[valid_mask]
    peak_indices = overlapPeak_indices[valid_peak_mask]    
    num_true = len(boundary_indices)

    data_values = data.values

    # Sliding windows (zero-copy)
    all_windows = sliding_window_view(data_values, window_shape=(sequenc_len,), axis=1)

    # Positive windows (boundary) -> 01 bc python start with 0 indx
    sequenc_true_array = np.stack([all_windows[:, idx - mid_indx] for idx in boundary_indices])
    sequenc_true_targets = np.ones(num_true)

    to_exclude = np.union1d(boundary_indices, peak_indices)
    
    # Now subtract that combined set from your range
    all_possible_nonboundary = np.setdiff1d(
        np.arange(sequenc_len, data.shape[1] - sequenc_len),
        to_exclude
    )
    nonboundary_indices = np.random.choice(all_possible_nonboundary, size=num_true, replace=False)
    sequenc_false_array = np.stack([all_windows[:, idx - mid_indx] for idx in nonboundary_indices])
    sequenc_false_targets = np.zeros(num_true)

    # ----- Training data -----
    train_true_indx = np.random.choice(num_true, int((1 - test_size) * num_true), replace=False)
    train_false_indx = np.random.choice(num_true, int((1 - test_size) * num_true), replace=False)

    X_train = np.concatenate([
        sequenc_true_array[train_true_indx],
        sequenc_false_array[train_false_indx]
    ], axis=0)

    y_train = np.concatenate([
        sequenc_true_targets[train_true_indx],
        sequenc_false_targets[train_false_indx]
    ])

    # ----- Shuffle train -----
    if shuffle:
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

    # ----- Testing data -----
    test_true_indx = np.setdiff1d(np.arange(num_true), train_true_indx)
    test_false_indx = np.setdiff1d(np.arange(num_true), train_false_indx)

    X_test = np.concatenate([
        sequenc_true_array[test_true_indx],
        sequenc_false_array[test_false_indx]
    ], axis=0)

    y_test = np.concatenate([
        sequenc_true_targets[test_true_indx],
        sequenc_false_targets[test_false_indx]
    ])

    if shuffle:
        test_perm = np.random.permutation(len(X_test))
        X_test = X_test[test_perm]
        y_test = y_test[test_perm]

    # ----- Validation data (split from test set) -----
    return X_train, y_train, X_test, y_test





# =========================================================
# Main
# =========================================================
def main(
    full_data_dir,
    truth_dir,
    peak_dir,
    save_dir,
    target_chrs,
    bin_size=100,
    sequence_len=101,
    num_iterations = 20
):
    if sequence_len % 2 == 0:
        raise ValueError("sequence_len must be an odd number.")
    
    ## read groundtruth
    all_chrom_PTBP = {}
    base_dir = truth_dir
    for chr_name in target_chrs:  # Include chromosome 22
        try:
            ptbp_df = pd.read_csv(base_dir, sep='\t', header=None)
            ptbp_df.columns = ["chr", "start", "end"]
            ptbp_df_chr = ptbp_df[ptbp_df["chr"] == chr_name]
            all_chrom_PTBP[f"{chr_name}"] = ptbp_df_chr
        except FileNotFoundError:
            print(f"File not found for chr{chr_name}: {base_dir}")
        except Exception as e:
            print(f"Error loading chr{chr_name}: {e}")

    all_chrom_feature = {}
    ## read feature- chromosome
    all_chrom_feature = read_data_dir(
        full_data_dir,
        target_chrs
    )

     ## Normalize 
    all_chrom_feature_df = {}
    for chr_name in target_chrs:  # include chr22
        try:
            data = all_chrom_feature[chr_name]
            # Normalize each chromosome
            data_log = data.fillna(0)
            mean = data_log.to_numpy().mean()
            std = data_log.to_numpy().std() + 1e-8 
            data_normed = (data_log - mean)/std
            all_chrom_feature_df[chr_name] = data_normed
        except Exception as e:
            print(f"Error loading {chr_name}: {e}")


    
    ## Prepare training daata
    # =================== Setup ===================
    seq_sizes = [sequence_len]
    num_iterations = num_iterations
    val_size = 0.5
    
    # Loop through each sequence size
    for seq_size in seq_sizes:
        print(f"\n==============================")
        print(f"Processing sequence length: {seq_size}")
        print(f"==============================")
    
        # Create save directory for this sequence size
        os.makedirs(save_dir, exist_ok=True)
    
        # =================== Iteration Loop ===================
        for seed in range(1, num_iterations + 1):
            print(f"\n=== Iteration {seed} (seq {seq_size}) ===")
    
            X_train_all = {}
            y_train_all = {}
            X_val_all = {}
            y_val_all = {}
            X_test_all = {}
            y_test_all = {}
    
            # Loop over chromosomes
            for chr_key in target_chrs:
                
                    
                # Check if both PTBP and feature data are available
                if chr_key in all_chrom_PTBP and chr_key in all_chrom_feature_df:
                    PTBP_expand = all_chrom_PTBP[chr_key]
                    data = all_chrom_feature_df[chr_key]
                    peak_path = os.path.join(peak_dir, f"bins_100bp_overlap_{chr_key}.bed")
                    overlap_Peak = pd.read_csv(peak_path, sep='\t', header=None)
            
                    try:
                        X_train, y_train, X_test, y_test = prepare_training_data(
                            PTBP_expand, data, overlap_Peak,
                            bin_size=bin_size,
                            sequenc_len=seq_size,
                            random_state=seed,
                            test_size=0.4,
                            val_size=val_size,
                            shuffle=True
                        )
            
                        # Store into respective dictionaries
                        X_train_all[chr_key] = X_train
                        y_train_all[chr_key] = y_train
                        X_test_all[chr_key] = X_test
                        y_test_all[chr_key] = y_test
            
                        print(f"{chr_key} processed: train = {len(y_train)}")
            
                    except Exception as e:
                        print(f"Error processing {chr_key}: {e}")
                else:
                    print(f"Data missing for {chr_key}")
            
            
            # Concatenate data from all chromosomes
            val_size=0.5
            import gc
            # ----- Training data -----
            X_train_full = np.concatenate(list(X_train_all.values()), axis=0)
            y_train_full = np.concatenate(list(y_train_all.values()), axis=0)
            del X_train_all, y_train_all  # <--- Delete dicts to free RAM
            gc.collect()                  # <--- Force memory cleanup
            
            perm_train = np.random.permutation(len(X_train_full))
            X_train_full_shuffled = X_train_full[perm_train]
            y_train_full_shuffled = y_train_full[perm_train]
            y_train = y_train_full_shuffled
            
            # ----- Test data and Validation data (split from test set) -----
            X_test_full = np.concatenate(list(X_test_all.values()), axis=0)
            y_test_full = np.concatenate(list(y_test_all.values()), axis=0)
            del X_test_all, y_test_all  # <--- Delete dicts to free RAM
            gc.collect()                  # <--- Force memory cleanup
            
            perm_test = np.random.permutation(len(X_test_full))
            X_test_full_shuffled = X_test_full[perm_test]
            y_test_full_shuffled = y_test_full[perm_test]
            
            # ----------
            val_size_int = int(val_size * len(y_test_full_shuffled))
            val_inx = np.random.choice(len(y_test_full_shuffled), size=val_size_int, replace=False)
            test_inx = np.setdiff1d(np.arange(len(y_test_full_shuffled)), val_inx)
            
            X_val_full_shuffled = X_test_full_shuffled[val_inx]
            y_val_full_shuffled = y_test_full_shuffled[val_inx]
            y_val = y_val_full_shuffled
            X_test_full_shuffled = X_test_full_shuffled[test_inx]
            y_test_full_shuffled = y_test_full_shuffled[test_inx]
            y_test = y_test_full_shuffled
            
            X_train_scale = X_train_full_shuffled
            X_val_scale = X_val_full_shuffled
            X_test_scale = X_test_full_shuffled
                    
            
           # # ========== Save result for this iteration ==========
            save_path = os.path.join(save_dir, f"dataset_seed_{seed}.pkl")
    
            data_to_save = {
                "seed": seed,
                "seq_size": seq_size,
                "X_train": X_train_scale,
                "y_train": y_train,
                "X_val": X_val_scale,
                "y_val": y_val,
                "X_test": X_test_scale,
                "y_test": y_test,
            }
    
            with open(save_path, "wb") as f:
                pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    
            print(f"✅ Saved: {save_path}")
            
            # Wipe EVERYTHING for the next seed iteration
            del X_train_scale, y_train_full_shuffled, X_val_scale, y_val_full_shuffled, X_test_scale, y_test
            gc.collect()
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Prepare training datasets for EpiTADformer"
    )

    parser.add_argument(
        "--full_data_dir",
        required=True,
        help="Directory containing epigenomic signal files"
    )

    parser.add_argument(
        "--truth_dir",
        required=True,
        help="Ground truth BED file"
    )

    parser.add_argument(
        "--peak_dir",
        required=True,
        help="Directory containing overlap peak BED files"
    )

    parser.add_argument(
        "--save_dir",
        required=True,
        help="Directory to save output datasets"
    )

    parser.add_argument(
        "--target_chrs",
        nargs="+",
        required=True,
        help="Chromosomes to process"
    )

    parser.add_argument(
        "--bin_size",
        type=int,
        default=100
    )

    parser.add_argument(
        "--sequence_len",
        type=int,
        default=101
    )

    parser.add_argument(
        "--num_iterations",
        type=int,
        default=20
    )

    args = parser.parse_args()

    if args.sequence_len % 2 == 0:
        raise ValueError(
            "sequence_len must be odd"
        )

    main(
        full_data_dir=args.full_data_dir,
        truth_dir=args.truth_dir,
        peak_dir=args.peak_dir,
        save_dir=args.save_dir,
        target_chrs=args.target_chrs,
        bin_size=args.bin_size,
        sequence_len=args.sequence_len,
        num_iterations=args.num_iterations
    )


    
    