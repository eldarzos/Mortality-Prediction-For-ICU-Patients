"""Truncate episode timeseries data to a specified length (t_hours).

Filters stays shorter than t_hours and clips events occurring after t_hours.
Creates a combined ITEMID_UOM identifier.
Saves output as episode{i}_timeseries_{t_hours}.csv.
"""

import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def truncate_timeseries(root_dir, t_hours):
    """Truncates timeseries files in subject directories."""
    subjects = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()]
    print(f"Found {len(subjects)} subject directories.")

    patient_count = 0 # Subjects with at least one valid stay >= t_hours
    stay_count = 0    # Stays >= t_hours with events within the window
    event_count = 0   # Events within the t_hours window across all valid stays

    itemids = set()
    uoms = set()
    itemid_uoms = set()

    for subject in tqdm(subjects, desc=f"Truncating timeseries to {t_hours}h"):
        subject_dir = os.path.join(root_dir, subject)
        # Find episode files (episode*.csv) and corresponding timeseries (episode*_timeseries.csv)
        episode_files = sorted([f for f in os.listdir(subject_dir) if f.startswith('episode') and f.endswith('.csv') and 'timeseries' not in f])

        subject_has_valid_stay = False

        for ep_file in episode_files:
            episode_num = ep_file.split('episode')[1].split('.csv')[0]
            ts_file = f'episode{episode_num}_timeseries.csv'
            ep_path = os.path.join(subject_dir, ep_file)
            ts_path = os.path.join(subject_dir, ts_file)

            if not os.path.exists(ts_path):
                # print(f"Warning: Timeseries file {ts_file} not found for episode {ep_file} in {subject}. Skipping.")
                continue

            try:
                # Read episode data to check Length of Stay (LOS)
                episode_data = pd.read_csv(ep_path)
                if episode_data.empty:
                    # print(f"Warning: Episode file {ep_file} is empty in {subject}. Skipping.")
                    continue

                los_days = episode_data['Length of Stay'].iloc[0]
                if pd.isna(los_days):
                    # print(f"Warning: Length of Stay is missing in {ep_file} for subject {subject}. Skipping stay.")
                    continue

                los_hours = los_days * 24.0
                # Filter stays shorter than the target truncation time
                if los_hours < t_hours:
                    # print(f"Info: Stay {episode_num} in {subject} has LOS {los_hours:.2f}h < {t_hours}h. Skipping.")
                    continue

                # Read the full timeseries
                ts = pd.read_csv(
                    ts_path,
                    usecols=['Hours', 'ITEMID', 'VALUE', 'VALUEUOM'] # Load only necessary columns
                )

                # Filter events based on the time window [0, t_hours)
                ts_truncated = ts[(ts['Hours'] >= 0) & (ts['Hours'] < t_hours)].copy()

                if ts_truncated.empty:
                    # print(f"Info: No events found within first {t_hours}h for stay {episode_num} in {subject}. Skipping stay.")
                    continue

                # Create combined ITEMID_UOM identifier
                # Ensure components are strings and handle potential NaNs
                ts_truncated['ITEMID'] = ts_truncated['ITEMID'].astype(str)
                ts_truncated['VALUEUOM'] = ts_truncated['VALUEUOM'].fillna('').astype(str)
                # Create tuple (or string) representation for the combined key
                # Using tuple is safer if ITEMIDs could contain separators
                # ts_truncated['ITEMID_UOM'] = list(zip(ts_truncated['ITEMID'], ts_truncated['VALUEUOM']))
                # Or use string concatenation if preferred and separator is safe:
                ts_truncated['ITEMID_UOM'] = ts_truncated['ITEMID'] + '_' + ts_truncated['VALUEUOM']


                # Update unique sets (use .update() for sets)
                itemids.update(ts_truncated['ITEMID'].unique())
                uoms.update(ts_truncated['VALUEUOM'].unique())
                itemid_uoms.update(ts_truncated['ITEMID_UOM'].unique())

                # Select final columns and save truncated timeseries
                final_ts = ts_truncated[['Hours', 'ITEMID_UOM', 'VALUE']] # Reorder for consistency maybe? Check later scripts. Let's keep original order for now + ITEMID_UOM
                final_ts = ts_truncated[['Hours', 'ITEMID', 'VALUE', 'VALUEUOM', 'ITEMID_UOM']] # Keep original + new ID for now

                output_filename = os.path.join(subject_dir, f'episode{episode_num}_timeseries_{t_hours}.csv')
                final_ts.to_csv(output_filename, index=False)

                # Update counts
                stay_count += 1
                event_count += final_ts.shape[0]
                subject_has_valid_stay = True

            except pd.errors.EmptyDataError:
                # print(f"Warning: Empty file encountered for episode {episode_num} in {subject}. Skipping.")
                continue
            except Exception as e:
                print(f"Error processing episode {episode_num} in subject {subject}: {e}")
                continue # Skip this episode on error

        if subject_has_valid_stay:
            patient_count += 1

    # Output summary statistics
    print(f"\n--- Truncation Summary ({t_hours}h) ---")
    print(f"Subjects with >=1 valid stay: {patient_count}")
    print(f"Total valid stays (>= {t_hours}h with events): {stay_count}")
    print(f"Total events within {t_hours}h window: {event_count}")
    print(f"Unique ITEMIDs: {len(itemids)}")
    print(f"Unique VALUEUOMs: {len(uoms)}")
    print(f"Unique (ITEMID_UOM) pairs: {len(itemid_uoms)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f"Truncate timeseries to a desired ICU stay length (t_hours).")
    parser.add_argument('root_dir', type=str,
                        help="Directory containing subject sub-directories (output of 3_subject2episode.py).")
    parser.add_argument('-t', '--t-hours', type=int, required=True,
                        help='Maximum number of hours to keep in timeseries (e.g., 24 or 48).')
    args = parser.parse_args()

    if args.t_hours <= 0:
        raise ValueError("t_hours must be a positive integer.")

    print(f"Starting truncation process for t_hours = {args.t_hours}...")
    truncate_timeseries(args.root_dir, t_hours=args.t_hours)
    print(f"Finished truncation for t_hours = {args.t_hours}.")
