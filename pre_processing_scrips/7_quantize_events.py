"""Quantize continuous events by percentile using the generated value dictionary.

Applies tokenization to train, validation, and test sets based on rules:
- Unknown ITEMID_UOM -> '<UNK>'
- Continuous variable (sufficient unique values) -> 'ITEMID_UOM:bin_index'
- Discrete variable or continuous w/ few values -> 'ITEMID_UOM:value'

Saves the token-to-index mapping.
Operates on a specific time window (t_hours), number of bins (n_bins), and seed.
"""

import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import sys

def quantize_events(root_dir, t_hours, n_bins, seed):
    """Quantizes events based on the value dictionary and data splits."""

    # --- 1. Load Value Dictionary and Split Files ---
    dict_dir = os.path.join(root_dir, 'dictionaries')
    value_dict_path = os.path.join(dict_dir, f'{t_hours}-{seed}-values.npy')
    split_dir = os.path.join(root_dir, 'splits')
    train_split_file = os.path.join(split_dir, f'{seed}-{t_hours}-train.csv')
    valid_split_file = os.path.join(split_dir, f'{seed}-{t_hours}-valid.csv')
    test_split_file = os.path.join(split_dir, f'{seed}-{t_hours}-test.csv')

    if not os.path.exists(value_dict_path):
        print(f"Error: Value dictionary not found at {value_dict_path}"); sys.exit(1)
    if not os.path.exists(train_split_file):
        print(f"Error: Train split file not found at {train_split_file}"); sys.exit(1)
    if not os.path.exists(valid_split_file):
        print(f"Error: Valid split file not found at {valid_split_file}"); sys.exit(1)
    if not os.path.exists(test_split_file):
        print(f"Error: Test split file not found at {test_split_file}"); sys.exit(1)

    try:
        V_full = np.load(value_dict_path, allow_pickle=True).item()
        print(f"Loaded value dictionary with {len(V_full)} keys.")
        train_files_df = pd.read_csv(train_split_file)
        valid_files_df = pd.read_csv(valid_split_file)
        test_files_df = pd.read_csv(test_split_file)
        train_paths = train_files_df['Paths'].tolist()
        valid_paths = valid_files_df['Paths'].tolist()
        test_paths = test_files_df['Paths'].tolist()
    except Exception as e:
        print(f"Error loading dictionary or split files: {e}"); sys.exit(1)

    # --- 2. Determine Percentiles for Continuous Variables ---
    print(f"Calculating percentiles for binning (n_bins = {n_bins})...")
    # Filter V_full to get variables suitable for binning
    V_to_bin = {} # Dictionary of variables to be binned
    P = {}        # Dictionary storing percentile boundaries {key: np.array([...])}

    for key, subdict in V_full.items():
        cont_values = subdict.get('cont', np.array([]))
        if len(cont_values) > 0: # Must have continuous values
            unique_cont_values = np.unique(cont_values)
            if len(unique_cont_values) >= n_bins: # Must have at least n_bins unique values
                V_to_bin[key] = subdict # Keep this variable for binning
                try:
                    # Calculate percentile boundaries (n_bins + 1 boundaries)
                    percentiles = np.linspace(0, 100, n_bins + 1)
                    boundaries = np.percentile(cont_values, percentiles)
                    # Ensure boundaries are unique (handle cases with many identical values)
                    unique_boundaries = np.unique(boundaries)
                    if len(unique_boundaries) < 2: # Need at least two boundaries for binning
                         print(f"Warning: Could not find distinct percentile boundaries for {key}. Skipping binning for this key.")
                         del V_to_bin[key] # Remove from binning list
                         continue # Skip to next key

                    # Add small epsilon to max boundary to include max value in last bin
                    # unique_boundaries[-1] += 1e-9
                    P[key] = unique_boundaries
                except Exception as e:
                    print(f"Warning: Error calculating percentiles for {key}: {e}. Skipping binning.")
                    if key in V_to_bin: del V_to_bin[key]
            # else: (Implicitly) Continuous variables with < n_bins unique values will be treated as discrete later

    print(f"Identified {len(P)} variables for percentile binning.")


    bin_boundaries_filename = os.path.join(dict_dir, f'{t_hours}_{seed}_{n_bins}-bin_boundaries.npy')
    try:
        # Ensure directory exists (it should, but good practice)
        os.makedirs(os.path.dirname(bin_boundaries_filename), exist_ok=True)
        np.save(bin_boundaries_filename, P)
        print(f"Saved bin boundaries to {bin_boundaries_filename}")
    except Exception as e:
        print(f"Error saving bin boundaries: {e}")

    # --- 3. Define Tokenization Function ---
    token_column_name = f'TOKEN_{n_bins}'

    def tokenize_row(row):
        key = row['ITEMID_UOM']
        value = row['VALUE']

        if pd.isna(key): return '<UNK>' # Handle missing key

        key_str = str(key) # Ensure key is string for lookups

        if key_str not in V_full:
            return '<UNK>' # Unknown item/uom pair seen only in valid/test

        # Check if this key is one we decided to bin
        if key_str in P:
            # Try to convert value to float for binning comparison
            try:
                float_value = float(value)
                if np.isnan(float_value): # Treat NaN as discrete
                     return f"{key_str}:nan"

                # Find the bin index using percentile boundaries
                # np.searchsorted returns the index where the value would be inserted
                # to maintain order. Subtract 1 to get the bin index (0 to n_bins-1).
                # 'right' ensures value == boundary falls into the left bin, except for the lowest boundary.
                boundaries = P[key_str]
                # Ensure value is within the min/max range observed in training, clip otherwise
                # clipped_value = np.clip(float_value, boundaries[0], boundaries[-1]) # Clip to range
                # bin_index = np.searchsorted(boundaries, clipped_value, side='right') - 1
                # Handle edge cases: value exactly on boundary or outside range
                bin_index = np.searchsorted(boundaries, float_value, side='right') -1

                # Ensure bin_index is within [0, n_bins-1]
                # Values less than the first boundary go to bin 0.
                # Values greater than or equal to the last boundary go to bin n_bins-1.
                bin_index = max(0, min(bin_index, n_bins - 1))

                return f"{key_str}:{bin_index}"

            except (ValueError, TypeError):
                # Value couldn't be converted to float, treat as discrete string
                 return f"{key_str}:{str(value)}"
        else:
            # Variable is discrete or continuous but not binned
            # Represent it as 'key:value'
            return f"{key_str}:{str(value)}"


    # --- 4. Apply Tokenization to Datasets ---
    print("Applying tokenization...")
    all_tokens = set(['<PAD>', '<UNK>']) # Initialize with padding and unknown tokens

    datasets = {'train': train_paths, 'valid': valid_paths, 'test': test_paths}
    for phase, paths in datasets.items():
        print(f" Processing {phase} set ({len(paths)} files)...")
        for relative_path in tqdm(paths, desc=f"Tokenizing {phase}"):
            full_path = os.path.join(root_dir, relative_path)
            if not os.path.exists(full_path):
                print(f"Warning: File not found: {full_path}. Skipping.")
                continue

            try:
                # Load timeseries, keeping only necessary columns + VALUE
                # Need ITEMID_UOM and VALUE for tokenization
                ts = pd.read_csv(full_path, usecols=['Hours', 'ITEMID_UOM', 'VALUE']) # Add other needed cols if necessary
                if ts.empty: continue

                # Apply tokenization function row-wise
                ts[token_column_name] = ts.apply(tokenize_row, axis=1)

                # Add tokens from this file to the global set (only for training set?)
                # Original code seemed to only collect from training set for the vocab
                if phase == 'train':
                    all_tokens.update(ts[token_column_name].unique())

                # Select columns to save (Hours, Token, maybe others?)
                # Let's save Hours and the new Token column, plus original VALUE for reference?
                # The final array creation script only needs Hours and Token index.
                ts_to_save = ts[['Hours', token_column_name]] # Keep only essential columns

                # Overwrite the original file with the tokenized version
                # This simplifies the next step (array creation)
                ts_to_save.to_csv(full_path, index=False)

            except pd.errors.EmptyDataError:
                # print(f"Warning: Empty file: {full_path}. Skipping.")
                continue
            except Exception as e:
                print(f"Error processing file {full_path} during tokenization: {e}")
                continue # Skip this file on error

    print(f"Finished tokenization. Found {len(all_tokens)} unique tokens in training data (+<PAD>, <UNK>).")

    # --- 5. Create and Save Token-to-Index Mapping ---
    # Sort tokens for consistent mapping
    sorted_tokens = sorted(list(all_tokens))
    # Assign index 0 to '<PAD>' token explicitly if needed by model, or start from 1
    # Original code started from 1, implying 0 might be reserved for padding/masking.
    # Let's follow that: index 0 = padding, index 1 = <UNK>, rest start from 2
    token2index = {token: i for i, token in enumerate(sorted_tokens)}
    # Adjust if padding needs index 0:
    if '<PAD>' in token2index:
        pad_idx = token2index['<PAD>']
        # Swap index 0 and pad_idx if <PAD> is not already at 0
        if pad_idx != 0:
             current_token_at_0 = sorted_tokens[0]
             token2index[current_token_at_0] = pad_idx
             token2index['<PAD>'] = 0


    print(f"Vocabulary size (including padding/unk): {len(token2index)}")

    # Define output directory for token maps (e.g., root_dir/dictionaries/)
    dict_dir = os.path.join(root_dir, 'dictionaries')
    os.makedirs(dict_dir, exist_ok=True)
    # Define output filename
    token_map_filename = os.path.join(dict_dir, f'{t_hours}_{seed}_{n_bins}-token2index.npy')

    # Save the token2index dictionary
    try:
        np.save(token_map_filename, token2index)
        print(f"Token to index map saved to {token_map_filename}")
    except Exception as e:
        print(f"Error saving token map to {token_map_filename}: {e}")


if __name__ == '__main__':


    quantize_events(data_path, 48, 20, 0)
    print("Event quantization and token mapping complete.")

