"""Split patient stays into training, validation, and testing sets.

Performs a stratified split based on the mortality label to ensure
similar class distribution across sets.
Saves the file paths and labels for each set.
Uses a fixed seed for reproducibility.
Operates on a specific time window (t_hours).
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_train_test(root_dir, t_hours, seed, test_size=0.1, valid_size=1000):
    """Splits the data for a given time window and seed."""
    subjects = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()]
    print(f"Found {len(subjects)} subject directories.")

    ts_paths = [] # Paths relative to root_dir
    mortality_labels = []
    los_values = []

    print(f"Collecting stay information for t_hours = {t_hours}...")
    for subject in tqdm(subjects, desc="Scanning subjects"):
        subject_dir = os.path.join(root_dir, subject)
        # Look for truncated timeseries files specific to t_hours
        potential_ts_files = [f for f in os.listdir(subject_dir) if f.startswith('episode') and f.endswith(f'_timeseries_{t_hours}.csv')]

        for ts_file in potential_ts_files:
            episode_num = ts_file.split('_timeseries_')[0].split('episode')[1]
            ep_file = f'episode{episode_num}.csv'
            ep_path = os.path.join(subject_dir, ep_file)

            if not os.path.exists(ep_path):
                # print(f"Warning: Episode file {ep_file} not found for {ts_file} in {subject}. Skipping stay.")
                continue

            try:
                # Read mortality and LOS from the episode file
                episode_data = pd.read_csv(ep_path)
                if episode_data.empty or 'Mortality' not in episode_data.columns or 'Length of Stay' not in episode_data.columns:
                    # print(f"Warning: Missing required columns or empty episode file {ep_file} in {subject}. Skipping.")
                    continue

                mortality = episode_data['Mortality'].iloc[0]
                los = episode_data['Length of Stay'].iloc[0]

                if pd.isna(mortality) or pd.isna(los):
                     # print(f"Warning: NaN value for Mortality or LOS in {ep_file} for subject {subject}. Skipping stay.")
                     continue

                # Store relative path, mortality, and LOS
                relative_ts_path = os.path.join(subject, ts_file) # Path relative to root_dir
                ts_paths.append(relative_ts_path)
                mortality_labels.append(int(mortality))
                los_values.append(los)

            except Exception as e:
                print(f"Error processing episode file {ep_file} in subject {subject}: {e}")
                continue # Skip this stay on error

    if not ts_paths:
        print(f"Error: No valid stays found for t_hours = {t_hours}. Cannot perform split.")
        return

    print(f"Found {len(ts_paths)} valid stays for t_hours = {t_hours}.")
    print(f"Mortality distribution: {pd.Series(mortality_labels).value_counts(normalize=True)}")

    # Create DataFrame for splitting
    data_df = pd.DataFrame({
        'Paths': ts_paths,
        'Mortality': mortality_labels,
        'LOS': los_values
    })

    # Split off the test set first (stratified)
    print(f"Splitting data using seed {seed}...")
    train_val_df, test_df = train_test_split(
        data_df,
        test_size=test_size,
        stratify=data_df['Mortality'],
        random_state=seed
    )

    # Split the remaining data into training and validation sets (stratified)
    # Ensure valid_size doesn't exceed the number of samples available
    actual_valid_size = min(valid_size, train_val_df.shape[0])
    if actual_valid_size < valid_size:
         print(f"Warning: Requested validation size ({valid_size}) is larger than available non-test samples ({train_val_df.shape[0]}). Using {actual_valid_size}.")

    # Calculate fraction for validation split if train_val_df is not empty
    if train_val_df.shape[0] > 0:
        validation_frac = actual_valid_size / train_val_df.shape[0]
        train_df, valid_df = train_test_split(
            train_val_df,
            test_size=validation_frac, # Use fraction here
            stratify=train_val_df['Mortality'],
            random_state=seed # Use the same seed for reproducibility
        )
    else:
        # Handle case where train_val_df might be empty after test split
        print("Warning: No samples left for training/validation after test split.")
        train_df = pd.DataFrame(columns=data_df.columns)
        valid_df = pd.DataFrame(columns=data_df.columns)


    print(f"Split sizes: Train={len(train_df)}, Validation={len(valid_df)}, Test={len(test_df)}")
    if len(train_df) > 0: print(f" Train Mortality: {train_df['Mortality'].value_counts(normalize=True)}")
    if len(valid_df) > 0: print(f" Valid Mortality: {valid_df['Mortality'].value_counts(normalize=True)}")
    if len(test_df) > 0: print(f" Test Mortality:  {test_df['Mortality'].value_counts(normalize=True)}")


    # Define output directory for split files (e.g., root_dir/splits/)
    split_dir = os.path.join(root_dir, 'splits')
    os.makedirs(split_dir, exist_ok=True)

    # Define output filenames including t_hours and seed
    train_filename = os.path.join(split_dir, f'{seed}-{t_hours}-train.csv')
    valid_filename = os.path.join(split_dir, f'{seed}-{t_hours}-valid.csv')
    test_filename = os.path.join(split_dir, f'{seed}-{t_hours}-test.csv')

    # Save the splits
    train_df.to_csv(train_filename, index=False)
    valid_df.to_csv(valid_filename, index=False)
    test_df.to_csv(test_filename, index=False)

    print(f"Saved train split ({len(train_df)} samples) to {train_filename}")
    print(f"Saved valid split ({len(valid_df)} samples) to {valid_filename}")
    print(f"Saved test split ({len(test_df)} samples) to {test_filename}")


if __name__ == '__main__':


    split_train_test(data_path, 48, 0, 0.2, 10000)
    print("Data splitting complete.")

