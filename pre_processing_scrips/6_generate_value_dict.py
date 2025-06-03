"""Generate a dictionary characterizing variables based on the training data.

For each unique variable identifier (ITEMID_UOM), it collects all observed values
from the training set timeseries files and categorizes them as continuous or discrete.
Saves the dictionary for later use in quantization.
Operates on a specific time window (t_hours) and seed.
"""

import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def try_to_float(v):
    """Attempt to convert value to float, return original if fails."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return str(v) # Keep as string if conversion fails

def generate_value_dict(root_dir, t_hours, seed):
    """Generates the value dictionary from training data."""
    # Path to the training split file
    split_dir = os.path.join(root_dir, 'splits')
    train_split_file = os.path.join(split_dir, f'{seed}-{t_hours}-train.csv')

    if not os.path.exists(train_split_file):
        print(f"Error: Training split file not found at {train_split_file}")
        return

    try:
        train_files_df = pd.read_csv(train_split_file)
        if 'Paths' not in train_files_df.columns:
             print(f"Error: 'Paths' column not found in {train_split_file}")
             return
        train_file_paths = train_files_df['Paths'].tolist()
    except Exception as e:
        print(f"Error reading training split file {train_split_file}: {e}")
        return

    if not train_file_paths:
        print("Warning: No training files listed in the split file. Value dictionary will be empty.")
        value_dict = {}
    else:
        print(f"Processing {len(train_file_paths)} training files for t_hours={t_hours}, seed={seed}...")
        # Use defaultdict for easier handling of new keys
        value_dict = defaultdict(lambda: {'disc': set(), 'cont': []}) # Use set for discrete to store uniques

        for relative_path in tqdm(train_file_paths, desc="Generating value dictionary"):
            full_path = os.path.join(root_dir, relative_path)
            if not os.path.exists(full_path):
                print(f"Warning: Training file not found: {full_path}. Skipping.")
                continue

            try:
                # Read only necessary columns
                ts = pd.read_csv(full_path, usecols=['ITEMID_UOM', 'VALUE'])

                for _, row in ts.iterrows():
                    # Use ITEMID_UOM created in script 4
                    key = row['ITEMID_UOM']
                    value = row['VALUE']

                    if pd.isna(key): # Skip if key is missing
                        continue

                    # Attempt to convert value to float
                    processed_value = try_to_float(value)

                    if isinstance(processed_value, float):
                        # Check for NaN floats specifically
                        if not np.isnan(processed_value):
                             value_dict[key]['cont'].append(processed_value)
                        else:
                             # Treat NaN floats as discrete 'nan' string
                             value_dict[key]['disc'].add('nan')
                    else:
                        # Value is discrete (string)
                        value_dict[key]['disc'].add(str(processed_value)) # Ensure it's a string

            except pd.errors.EmptyDataError:
                # print(f"Warning: Empty timeseries file: {full_path}. Skipping.")
                continue
            except Exception as e:
                print(f"Error processing file {full_path}: {e}")
                continue # Skip this file on error

    # Post-processing the dictionary
    final_value_dict = {}
    print("Finalizing dictionary...")
    for key, values in tqdm(value_dict.items(), desc="Converting lists"):
        final_value_dict[key] = {
            'disc': sorted(list(values['disc'])), # Convert set to sorted list
            'cont': np.array(values['cont'], dtype=np.float32) if values['cont'] else np.array([], dtype=np.float32) # Convert list to numpy array
        }
        # Optional: Print stats for some keys
        # if np.random.rand() < 0.001: # Print for a small fraction
        #    print(f" Key: {key}, Discrete: {len(final_value_dict[key]['disc'])}, Continuous: {len(final_value_dict[key]['cont'])}")


    # Define output directory for dictionaries (e.g., root_dir/dictionaries/)
    dict_dir = os.path.join(root_dir, 'dictionaries')
    os.makedirs(dict_dir, exist_ok=True)

    # Define output filename including t_hours and seed
    output_filename = os.path.join(dict_dir, f'{t_hours}-{seed}-values.npy')

    # Save the dictionary
    try:
        np.save(output_filename, final_value_dict)
        print(f"Value dictionary saved to {output_filename}")
        num_keys = len(final_value_dict)
        num_cont_keys = sum(1 for k in final_value_dict if len(final_value_dict[k]['cont']) > 0)
        print(f" Dictionary contains {num_keys} unique ITEMID_UOM keys.")
        print(f" {num_cont_keys} keys have continuous values.")
    except Exception as e:
        print(f"Error saving value dictionary to {output_filename}: {e}")


if __name__ == '__main__':

    generate_value_dict(data_path, 48, 0)
    print("Value dictionary generation complete.")

