"""Extract episode information and timeseries data for each ICU stay
from validated per-subject directories.

Generates:
- episode{i}.csv: Static data for the i-th ICU stay (Age, Gender, LOS, Mortality).
- episode{i}_timeseries.csv: Time-aligned events for the i-th ICU stay.
"""

import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


# Mapping for gender consistent with original script
g_map = {'F': 1, 'M': 2, 'OTHER': 3, '': 0}

def subject2episode(root_path):
    """Processes each subject directory to extract episode data."""
    subject_dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d)) and d.isdigit()]
    print(f"Found {len(subject_dirs)} subject directories.")

    for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
        subject_path = os.path.join(root_path, subject_dir)
        stays_path = os.path.join(subject_path, 'stays.csv')
        events_path = os.path.join(subject_path, 'events.csv')

        if not os.path.exists(stays_path):
            # print(f"Warning: stays.csv not found for subject {subject_dir}. Skipping.")
            continue
        if not os.path.exists(events_path):
            # print(f"Warning: events.csv not found for subject {subject_dir}. Skipping.")
            continue

        try:
            # Read stays data
            stays = pd.read_csv(stays_path, header=0)
            # Convert relevant datetime columns
            datetime_cols = ['INTIME', 'OUTTIME', 'DOB', 'DOD', 'DEATHTIME', 'ADMITTIME', 'DISCHTIME']
            for c in datetime_cols:
                if c in stays.columns:
                    stays[c] = pd.to_datetime(stays[c], errors='coerce') # Coerce errors to NaT
            stays.sort_values(by=['INTIME', 'OUTTIME'], inplace=True)

            # Read events data
            events = pd.read_csv(events_path, header=0, index_col=None)
            if events.empty:
                # print(f"Info: events.csv is empty for subject {subject_dir}. Skipping.")
                continue

            events['CHARTTIME'] = pd.to_datetime(events['CHARTTIME'], errors='coerce')
            # Ensure IDs are strings, handling potential float reads due to NaNs
            events['HADM_ID'] = events['HADM_ID'].astype(str)
            events['ICUSTAY_ID'] = events['ICUSTAY_ID'].astype(str)
            # Ensure VALUEUOM is string, fill NaNs
            events['VALUEUOM'] = events['VALUEUOM'].fillna('').astype(str)
            # Ensure VALUE is string
            events['VALUE'] = events['VALUE'].astype(str)
            # Drop rows where CHARTTIME failed to parse
            events.dropna(subset=['CHARTTIME'], inplace=True)


            # Process each stay (episode) for the subject
            for i in range(stays.shape[0]):
                stay_row = stays.iloc[i]
                stay_id = str(stay_row['ICUSTAY_ID'])
                intime = stay_row['INTIME']
                outtime = stay_row['OUTTIME']

                if pd.isna(intime) or pd.isna(outtime):
                    print(f"Warning: Missing INTIME or OUTTIME for ICUSTAY_ID {stay_id} in subject {subject_dir}. Skipping stay.")
                    continue

                # --- Create static episode data (episode{i+1}.csv) ---
                ep_data = {
                    'Icustay':        stay_id,
                    'Age':            stay_row['AGE'],
                    'Length of Stay': stay_row['LOS'], # LOS in days from stays.csv
                    'Mortality':      stay_row['MORTALITY'] # In-hospital mortality
                }
                # Map gender using the predefined map
                gender = stay_row.get('GENDER', '') # Use .get for safety
                ep_data['Gender'] = g_map.get(gender, g_map['OTHER']) # Default to OTHER if unknown

                ep_df = pd.DataFrame([ep_data]) # Create DataFrame from single dict

                # Save static episode data
                ep_filename = os.path.join(subject_path, f'episode{i+1}.csv')
                ep_df.to_csv(ep_filename, index=False)


                # --- Create timeseries data (episode{i+1}_timeseries.csv) ---
                # Filter events belonging to the current ICUSTAY_ID
                # AND occurring within the INTIME and OUTTIME
                episode_events = events[
                    (events['ICUSTAY_ID'] == stay_id) &
                    (events['CHARTTIME'] >= intime) &
                    (events['CHARTTIME'] <= outtime)
                ].copy() # Use copy to avoid SettingWithCopyWarning

                if episode_events.empty:
                    # print(f"Info: No events found within stay {stay_id} for subject {subject_dir}.")
                    # Create an empty timeseries file for consistency? Or skip? Let's skip creating empty files.
                    continue

                # Calculate time difference in hours from INTIME
                # Use total_seconds() for accurate calculation
                episode_events['Hours'] = (episode_events['CHARTTIME'] - intime).dt.total_seconds() / 3600.0
                # Drop original CHARTTIME and potentially redundant IDs
                episode_events = episode_events.drop(columns=['CHARTTIME', 'ICUSTAY_ID', 'HADM_ID', 'SUBJECT_ID'], errors='ignore')
                # Keep relevant columns: Hours, ITEMID, VALUE, VALUEUOM
                episode_events = episode_events[['Hours', 'ITEMID', 'VALUE', 'VALUEUOM']]
                # Sort events by time
                episode_events = episode_events.sort_values(by='Hours')

                # Save timeseries data
                ts_filename = os.path.join(subject_path, f'episode{i+1}_timeseries.csv')
                episode_events.to_csv(ts_filename, index=False) # Keep header 'Hours'

        except pd.errors.EmptyDataError:
            # print(f"Warning: Empty stays.csv or events.csv found for subject {subject_dir} after loading. Skipping.")
            continue
        except Exception as e:
            print(f"Error processing subject {subject_dir}: {e}")
            continue # Skip this subject on error

    print("Finished structuring episodes.")


if __name__ == '__main__':

    # data_path = <enter path for preprocwssing data>
    subject2episode(data_path)
