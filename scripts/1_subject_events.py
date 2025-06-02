"""Extract subject information and events from MIMIC-III csvs."""

import argparse
import csv
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import logging # Added for logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    """Main function to extract per-subject data."""
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Read in tables
    logger.info('Reading in tables...')
    try:
        pats = pd.read_csv(
            os.path.join(args.mimic3_path, 'PATIENTS.csv'),
            header=0, index_col=0, usecols=['SUBJECT_ID', 'GENDER', 'DOB', 'DOD'])
        # Convert dates, coercing errors to NaT (Not a Time)
        pats['DOB'] = pd.to_datetime(pats['DOB'], errors='coerce')
        pats['DOD'] = pd.to_datetime(pats['DOD'], errors='coerce')

        admits = pd.read_csv(
            os.path.join(args.mimic3_path, 'ADMISSIONS.csv'),
            header=0, index_col=0, usecols=[
                'SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME',
                'DEATHTIME', 'ADMISSION_TYPE', 'ETHNICITY', 'DIAGNOSIS'])
        admits['ADMITTIME'] = pd.to_datetime(admits['ADMITTIME'], errors='coerce')
        admits['DISCHTIME'] = pd.to_datetime(admits['DISCHTIME'], errors='coerce')
        admits['DEATHTIME'] = pd.to_datetime(admits['DEATHTIME'], errors='coerce')

        stays = pd.read_csv(
            os.path.join(args.mimic3_path, 'ICUSTAYS.csv'),
            header=0, index_col=0)
        stays['INTIME'] = pd.to_datetime(stays['INTIME'], errors='coerce')
        stays['OUTTIME'] = pd.to_datetime(stays['OUTTIME'], errors='coerce')
    except FileNotFoundError as e:
        logger.error(f"Error reading input CSV file: {e}. Please check --mimic3-path.")
        return # Exit if essential files are missing
    except Exception as e:
        logger.error(f"Error during table reading or initial date conversion: {e}")
        return

    logger.info(f"Initial stays: ICUSTAY_IDs={len(stays['ICUSTAY_ID'].unique())}, HADM_IDs={len(stays['HADM_ID'].unique())}, SUBJECT_IDs={len(stays['SUBJECT_ID'].unique())}")

    # Remove icustays with transfers
    logger.info('Removing icustays with transfers...')
    original_rows = stays.shape[0]
    stays = stays.loc[
        (stays['FIRST_WARDID'] == stays['LAST_WARDID']) &
        (stays['FIRST_CAREUNIT'] == stays['LAST_CAREUNIT'])]
    stays = stays.drop(columns=['FIRST_WARDID', 'LAST_WARDID', 'FIRST_CAREUNIT', 'LAST_CAREUNIT'], errors='ignore')
    logger.info(f" Stays after removing transfers: {stays.shape[0]} (removed {original_rows - stays.shape[0]})")
    logger.info(f" ICUSTAY_IDs={len(stays['ICUSTAY_ID'].unique())}, HADM_IDs={len(stays['HADM_ID'].unique())}, SUBJECT_IDs={len(stays['SUBJECT_ID'].unique())}")


    # Merge on subject admission and subject
    try:
        stays = stays.merge(
            admits,
            left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])
        stays = stays.merge(
            pats,
            left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])
    except Exception as e:
        logger.error(f"Error during merging of tables: {e}")
        return

    # Filter admissions on number of ICU stays and age
    logger.info('Filtering admissions on number of ICU stays (keeping only 1) and age (>=18)...')
    # Count ICU stays per admission
    icu_counts = stays.groupby('HADM_ID')['ICUSTAY_ID'].transform('count')
    stays = stays[icu_counts == 1]
    logger.info(f" Stays after keeping HADM_IDs with 1 ICUSTAY: {stays.shape[0]}")

    # --- Robust Age Calculation ---
    logger.info("Calculating patient age...")
    # Identify rows where both INTIME and DOB are valid dates
    valid_dates_mask = stays['INTIME'].notna() & stays['DOB'].notna()
    logger.info(f" Found {valid_dates_mask.sum()} stays with valid INTIME and DOB out of {len(stays)}.")

    # Initialize AGE column with NaN
    stays['AGE'] = np.nan

    # Calculate age only for rows with valid dates
    if valid_dates_mask.any(): # Proceed only if there are valid dates
        try:
            # Calculate timedelta only for valid rows
            time_diff_valid = stays.loc[valid_dates_mask, 'INTIME'] - stays.loc[valid_dates_mask, 'DOB']

            # Convert timedelta to age in years
            # Handle potential errors during days conversion gracefully
            # NaT timedeltas will result in NaN days
            age_in_days_valid = time_diff_valid.dt.days
            stays.loc[valid_dates_mask, 'AGE'] = age_in_days_valid / 365.25

        except OverflowError as e:
            # This might still occur if the *difference* is too large, even with valid dates
            logger.error(f"OverflowError encountered during age calculation: {e}")
            logger.error("This suggests extreme date differences (> ~292 years) for some valid date pairs.")
            # AGE will remain NaN for these specific overflow cases due to initialization
        except Exception as e:
            logger.error(f"Unexpected error during age calculation for valid dates: {e}")
            # AGE will remain NaN for these cases

    # Handle ages > 89 (shifted dates in MIMIC result in INTIME < DOB -> negative age)
    # Use 91.4 as the standard MIMIC representation for age > 89.
    # This condition implicitly handles the original negative age check.
    negative_age_mask = stays['AGE'] < 0
    if negative_age_mask.any():
        logger.info(f" Found {negative_age_mask.sum()} stays with calculated age < 0 (likely >89 years old). Setting age to 91.4.")
        stays.loc[negative_age_mask, 'AGE'] = 91.4

    # Handle remaining NaN ages (due to NaT inputs or calculation errors)
    nan_age_mask = stays['AGE'].isna()
    if nan_age_mask.any():
        logger.warning(f" Found {nan_age_mask.sum()} stays with invalid/missing age after calculation. Filling with 91.4 (standard for >89/unknown).")
        # Option: Fill with a default (91.4 is often used), or drop these rows. Filling is safer.
        stays['AGE'].fillna(91.4, inplace=True)
        # To drop instead: stays.dropna(subset=['AGE'], inplace=True)

    # Filter stays based on final calculated/imputed age (>= 18)
    original_count_before_age_filter = stays.shape[0]
    stays = stays.loc[stays['AGE'] >= 18]
    logger.info(f" Stays after age filter (>=18): {stays.shape[0]} (removed {original_count_before_age_filter - stays.shape[0]})")
    logger.info(f" Final unique counts: ICUSTAY_IDs={len(stays['ICUSTAY_ID'].unique())}, HADM_IDs={len(stays['HADM_ID'].unique())}, SUBJECT_IDs={len(stays['SUBJECT_ID'].unique())}")


    # Add mortality info
    logger.info('Adding mortality info...')
    # In-unit mortality
    mortality_inunit = (
        stays['DOD'].notna() &
        (stays['INTIME'] <= stays['DOD']) &
        (stays['OUTTIME'] >= stays['DOD']))
    mortality_inunit = (
        mortality_inunit |
        (stays['DEATHTIME'].notna() &
        (stays['INTIME'] <= stays['DEATHTIME']) &
        (stays['OUTTIME'] >= stays['DEATHTIME'])))
    stays['MORTALITY_INUNIT'] = mortality_inunit.astype(int)

    # In-hospital mortality (using this for the label 'MORTALITY')
    mortality_inhospital = (stays['DOD'].notna()) & \
                (stays['ADMITTIME'] <= stays['DOD']) & \
                (stays['DISCHTIME'] >= stays['DOD'])
    mortality_inhospital = (mortality_inhospital) | \
                (stays['DEATHTIME'].notna()) & \
                (stays['ADMITTIME'] <= stays['DEATHTIME']) & \
                (stays['DISCHTIME'] >= stays['DEATHTIME'])
    stays['MORTALITY'] = mortality_inhospital.astype(int)


    # Break up stays by subject
    logger.info('Breaking up stays by subject...')
    subjects = stays['SUBJECT_ID'].unique()
    for subject_id in tqdm(subjects, desc="Saving stays per subject"):
        subject_stays = stays.loc[stays['SUBJECT_ID'] == subject_id].sort_values(by='INTIME')
        if not subject_stays.empty:
            dn = os.path.join(args.output_path, str(subject_id))
            if not os.path.exists(dn):
                os.makedirs(dn)
            try:
                subject_stays.to_csv(os.path.join(dn, 'stays.csv'), index=False)
            except Exception as e:
                logger.error(f"Error saving stays.csv for subject {subject_id}: {e}")


    # Read events table and break up by subject
    logger.info('Reading events tables and breaking up by subject...')
    subjects_with_stays = set(str(s) for s in subjects)
    nb_rows = {
        'CHARTEVENTS':  330712484, # Approx rows from MIMIC documentation
        'LABEVENTS':    27854056,
        'OUTPUTEVENTS': 4349219
    }
    event_columns = { # Define specific columns to load
        'CHARTEVENTS': ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM'],
        'LABEVENTS': ['SUBJECT_ID', 'HADM_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM'],
        'OUTPUTEVENTS': ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']
    }
    event_dtypes = { # Define dtypes for memory efficiency
        'SUBJECT_ID': int, 'HADM_ID': float, 'ICUSTAY_ID': float,
        'ITEMID': int, 'VALUE': str, 'VALUEUOM': str
    }
    obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']


    # --- Process Events Table ---
    # (Keeping the event processing logic from the previous version)
    for table in args.event_tables:
        logger.info(f" Processing {table}...")
        tn = os.path.join(args.mimic3_path, table + '.csv')
        if not os.path.exists(tn):
            logger.warning(f" {table}.csv not found in {args.mimic3_path}. Skipping.")
            continue

        chunk_size = 1000000 # Process in chunks
        output_files = {} # Cache open file writers

        try:
            reader = pd.read_csv(
                tn,
                usecols=event_columns[table],
                dtype=event_dtypes,
                parse_dates=['CHARTTIME'],
                chunksize=chunk_size,
                low_memory=False # Added to potentially help with mixed types if any
            )

            for chunk in tqdm(reader, total=int(np.ceil(nb_rows[table] / chunk_size)), desc=f"Reading {table}"):
                chunk['SUBJECT_ID'] = chunk['SUBJECT_ID'].astype(str)
                chunk = chunk[chunk['SUBJECT_ID'].isin(subjects_with_stays)]
                if chunk.empty: continue

                chunk['HADM_ID'] = chunk['HADM_ID'].fillna(-1).astype(int).astype(str)
                if 'ICUSTAY_ID' in chunk.columns:
                    chunk['ICUSTAY_ID'] = chunk['ICUSTAY_ID'].fillna(-1).astype(int).astype(str)
                else:
                    chunk['ICUSTAY_ID'] = '-1'

                chunk['VALUEUOM'] = chunk['VALUEUOM'].fillna('').astype(str)
                chunk['VALUE'] = chunk['VALUE'].astype(str)
                # Convert CHARTTIME to datetime, coercing errors
                chunk['CHARTTIME'] = pd.to_datetime(chunk['CHARTTIME'], errors='coerce')
                chunk.dropna(subset=['CHARTTIME'], inplace=True) # Drop rows where date conversion failed

                chunk = chunk[obs_header] # Ensure correct column order

                for subject_id, group in chunk.groupby('SUBJECT_ID'):
                    dn = os.path.join(args.output_path, subject_id)
                    fn = os.path.join(dn, 'events.csv')

                    if subject_id not in output_files:
                        file_exists = os.path.exists(fn)
                        # Ensure directory exists before opening file
                        os.makedirs(dn, exist_ok=True)
                        output_files[subject_id] = open(fn, 'a', newline='', encoding='utf-8') # Added encoding
                        writer = csv.writer(output_files[subject_id])
                        if not file_exists or os.path.getsize(fn) == 0:
                            writer.writerow(obs_header)
                        output_files[subject_id].writer = writer

                    output_files[subject_id].writer.writerows(group.values.tolist())
        except Exception as e:
             logger.error(f"Error processing chunk for {table}: {e}")
             # Continue processing other chunks/tables if possible
        finally:
            logger.info(f"Closing event files for {table}...")
            for f in output_files.values():
                f.close()

    logger.info('Finished extracting events.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract per-subject data from MIMIC-III CSV files.')
    parser.add_argument(
        'mimic3_path', type=str,
        help='Directory containing MIMIC-III CSV files (PATIENTS.csv, ADMISSIONS.csv, ICUSTAYS.csv, etc.).')
    parser.add_argument(
        'output_path', type=str,
        help='Directory where per-subject data folders should be written.')
    parser.add_argument(
        '--event_tables', '-e', type=str, nargs='+',
        default=['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS'],
        choices=['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS'],
        help='Tables from which to read events.')
    args = parser.parse_args()

    main(args)
    logger.info("Script 1 finished.")

