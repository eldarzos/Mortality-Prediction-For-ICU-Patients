"""Create final NumPy arrays for model input.

Reads tokenized timeseries data, converts tokens to indices, pads/truncates
sequences to a fixed length, and collects mortality labels and LOS.
Saves the final arrays (X, Y, LOS, paths) in a dictionary format.
Operates on a specific time window (t_hours), number of bins (n_bins), and seed.
"""

import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import sys

def create_arrays(root_dir, t_hours, n_bins, seed, max_len=10000):
    """Creates the final NumPy arrays for model input."""

    # --- 1. Load Token Map ---
    dict_dir = os.path.join(root_dir, 'dictionaries')
    token_map_path = os.path.join(dict_dir, f'{t_hours}_{seed}_{n_bins}-token2index.npy')

    if not os.path.exists(token_map_path):
        print(f"Error: Token map not found at {token_map_path}"); sys.exit(1)

    try:
        token2index = np.load(token_map_path, allow_pickle=True).item()
        print(f"Loaded token map with {len(token2index)} tokens.")
        # Determine padding index (assuming it's 0, adjust if necessary based on script 7)
        padding_idx = 0
        if '<PAD>' in token2index:
             padding_idx = token2index['<PAD>']
        else:
             print("Warning: '<PAD>' token not found in map. Assuming padding index is 0.")
        print(f"Using padding index: {padding_idx}")
        token_column_name = f'TOKEN_{n_bins}' # Get token column name used in script 7
    except Exception as e:
        print(f"Error loading token map: {e}"); sys.exit(1)


    # --- 2. Process Timeseries Files ---
    subjects = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()]
    print(f"Found {len(subjects)} subject directories.")

    all_sequences = []
    all_mortality = []
    all_los = []
    all_paths = [] # Store relative paths for reference

    print(f"Processing tokenized timeseries files for t_hours={t_hours}...")
    for subject in tqdm(subjects, desc="Creating arrays"):
        subject_dir = os.path.join(root_dir, subject)
        # Find the tokenized timeseries files for the current t_hours
        potential_ts_files = [f for f in os.listdir(subject_dir) if f.startswith('episode') and f.endswith(f'_timeseries_{t_hours}.csv')]

        for ts_file in potential_ts_files:
            episode_num = ts_file.split('_timeseries_')[0].split('episode')[1]
            ep_file = f'episode{episode_num}.csv'
            ts_path = os.path.join(subject_dir, ts_file)
            ep_path = os.path.join(subject_dir, ep_file)

            if not os.path.exists(ep_path):
                # print(f"Warning: Episode file {ep_file} not found for {ts_file}. Skipping.")
                continue

            try:
                # Read episode data for labels
                episode_data = pd.read_csv(ep_path)
                if episode_data.empty: continue # Skip if episode file is empty

                mortality = episode_data['Mortality'].iloc[0]
                los = episode_data['Length of Stay'].iloc[0]
                if pd.isna(mortality) or pd.isna(los): continue # Skip if labels are missing

                # Read tokenized timeseries data (Hours, TOKEN_n)
                ts = pd.read_csv(ts_path, usecols=['Hours', token_column_name])
                if ts.empty: continue # Skip if timeseries is empty

                # Map tokens to indices
                # Use .get(token, unk_index) for safety against unseen tokens (though should not happen)
                unk_index = token2index.get('<UNK>', 1) # Default UNK index to 1
                ts['TokenIndex'] = ts[token_column_name].map(lambda x: token2index.get(x, unk_index))

                # Prepare sequence array [Hours, TokenIndex]
                sequence = ts[['Hours', 'TokenIndex']].values.astype(np.float32)

                # Pad or truncate sequence
                seq_len = sequence.shape[0]
                if seq_len == 0: continue # Skip empty sequences after processing

                if seq_len < max_len:
                    # Pad sequence
                    pad_width = max_len - seq_len
                    # Pad 'Hours' with t_hours (or another indicator like -1?) - Original used t_hours
                    # Pad 'TokenIndex' with padding_idx
                    padded_sequence = np.pad(
                        sequence,
                        pad_width=((0, pad_width), (0, 0)), # Pad only at the end
                        mode='constant',
                        constant_values=((0, 0), (t_hours, padding_idx)) # Pad hours with t_hours, index with padding_idx
                        # Correction: constant_values should be single values per axis pair?
                        # Let's pad hours with a distinct value like -1 or keep t_hours? Using t_hours as per original.
                    )
                    # Need to structure padding correctly. Pad time first, then tokens.
                    time_col = sequence[:, 0:1]
                    token_col = sequence[:, 1:2]

                    padded_time = np.pad(time_col, ((0, pad_width), (0, 0)), mode='constant', constant_values=t_hours)
                    padded_tokens = np.pad(token_col, ((0, pad_width), (0, 0)), mode='constant', constant_values=padding_idx)
                    final_sequence = np.concatenate((padded_time, padded_tokens), axis=1)

                elif seq_len > max_len:
                    # Truncate sequence (keep the latest events)
                    final_sequence = sequence[-max_len:, :]
                    # print(f'Warning: Long timeseries truncated ({seq_len} -> {max_len}) for {subject}/{ts_file}')
                else:
                    # Length is exactly max_len
                    final_sequence = sequence

                # Append processed data
                all_sequences.append(final_sequence)
                all_mortality.append(int(mortality))
                all_los.append(los)
                all_paths.append(os.path.join(subject, ts_file)) # Store relative path

            except pd.errors.EmptyDataError:
                # print(f"Warning: Empty file encountered for {ts_file} or {ep_file}. Skipping.")
                continue
            except KeyError as e:
                 print(f"Warning: Missing expected column {e} in {ts_file} or {ep_file}. Skipping.")
                 continue
            except Exception as e:
                print(f"Error processing stay {episode_num} in subject {subject}: {e}")
                continue # Skip this stay on error

    if not all_sequences:
        print("Error: No valid sequences were processed. Cannot save arrays.")
        sys.exit(1)

    # --- 3. Stack and Save Arrays ---
    print("Stacking arrays...")
    try:
        X = np.stack(all_sequences, axis=0).astype(np.float32) # Shape: (n_stays, max_len, 2)
        Y = np.array(all_mortality).astype(np.int32)           # Shape: (n_stays,)
        LOS = np.array(all_los).astype(np.float32)             # Shape: (n_stays,)
        Paths = np.array(all_paths)                            # Shape: (n_stays,)

        print(f"Final array shapes: X={X.shape}, Y={Y.shape}, LOS={LOS.shape}, Paths={Paths.shape}")

        # Define output directory for final arrays (e.g., root_dir/arrays/)
        array_dir = os.path.join(root_dir, 'arrays')
        os.makedirs(array_dir, exist_ok=True)
        # Define output filename
        output_filename = os.path.join(array_dir, f'{t_hours}_{seed}_{n_bins}-arrays.npz') # Use npz for named arrays

        # Save arrays into a dictionary within a npz file
        np.savez(
            output_filename,
            X=X,
            Y=Y,
            LOS=LOS,
            paths=Paths
        )
        print(f"Final arrays saved to {output_filename}")

    except Exception as e:
        print(f"Error stacking or saving final arrays: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # data_dir = 
    create_arrays(data_dir, 48, 20, 0, 10000)
    print("Array creation complete.")
