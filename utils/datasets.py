"""Datasets module for loading pre-split EHR data."""

import logging
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class EHR(Dataset):
    """
    EHR Dataset - Stores pre-processed sequences and labels.

    Parameters
    ----------
    X: numpy.ndarray
        Array containing patient sequences for this split, shape (n_patients, max_len, 2)
    Y: numpy.ndarray
        Array containing patient outcomes for this split, shape (n_patients,)
    n_tokens_expected: int
        The expected vocabulary size (max token index + 1) based on the token map.
    t_hours: int, optional
        Time horizon (used for dynamic label tiling if applicable).
    dt: float, optional
        Time step between intervals (used for dynamic label tiling if applicable).
    dynamic: bool, optional
        Whether the model expects dynamically tiled labels.
    logger: logging.Logger, optional
    """
    def __init__(self, X, Y, n_tokens_expected, t_hours=48, dt=1.0, dynamic=True,
                 logger=logging.getLogger(__name__)):

        self.logger = logger
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y_original = torch.tensor(Y, dtype=torch.float32) # Store original labels

        # --- Data Validation and Clamping ---
        if self.X.shape[0] > 0: # Only process if there's data
            token_indices = self.X[:, :, 1]
            # Handle potential NaNs introduced during padding or processing
            token_indices = torch.nan_to_num(token_indices, nan=0.0) # Replace NaN with 0 (padding index)

            max_id = int(token_indices.max().item())
            min_id = int(token_indices.min().item())
            logger.info(f"🔍 Initial Token ID range in loaded data: min={min_id}, max={max_id}")
            logger.info(f"  Expected vocabulary size (n_tokens): {n_tokens_expected}")

            # Clamp values to be within the valid range [0, n_tokens_expected - 1]
            # This prevents IndexError in the embedding layer.
            # Values outside this range indicate an issue in preprocessing/tokenization.
            if max_id >= n_tokens_expected or min_id < 0:
                logger.warning(f"⚠️ Token IDs found outside expected range [0, {n_tokens_expected-1}]. Clamping values.")
                self.X[:, :, 1] = torch.clamp(token_indices, min=0, max=n_tokens_expected - 1)
                # Log new range after clamping
                max_id_clamped = int(self.X[:, :, 1].max().item())
                min_id_clamped = int(self.X[:, :, 1].min().item())
                logger.info(f"  Clamped Token ID range: min={min_id_clamped}, max={max_id_clamped}")
            else:
                 self.X[:, :, 1] = token_indices # Assign back NaN-handled tensor
                 logger.info(f"  Token IDs are within the expected range.")
        else:
             logger.warning("⚠️ EHR Dataset initialized with zero samples.")


        # --- Dynamic Label Tiling ---
        if dynamic:
            # Tile the original Y label across time steps if needed by the model
            num_intervals = int(t_hours / dt)
            if self.Y_original.ndim == 1 and self.Y_original.shape[0] > 0:
                 # Reshape Y to (n_patients, 1) before tiling
                 self.Y = self.Y_original.unsqueeze(1).repeat(1, num_intervals)
            elif self.Y_original.shape[0] == 0:
                 # Handle empty case: create empty tensor with correct dimensions
                 self.Y = torch.empty((0, num_intervals), dtype=torch.float32)
            else:
                 # Y might already be tiled or have unexpected shape
                 logger.warning(f"Dynamic label tiling requested, but Y shape is {self.Y_original.shape}. Using original Y.")
                 self.Y = self.Y_original
        else:
            self.Y = self.Y_original # Use original labels if not dynamic

        self.len = self.X.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Returns the sequence and the (potentially tiled) label
        return self.X[idx], self.Y[idx]


def get_dataloaders(array_path, token_map_path, # Paths to data and token map
                    train_split_path=None, valid_split_path=None, test_split_path=None, # Paths to split files
                    validation=True, # If True, load train+valid; If False, load test
                    t_hours=48, dt=1.0, dynamic=True, # Dataset parameters
                    shuffle=True, pin_memory=True, batch_size=128, # DataLoader parameters
                    logger=logging.getLogger(__name__)):
    """
    Creates PyTorch DataLoaders based on pre-defined data splits.

    Parameters:
        array_path (str): Path to the .npz file containing 'X', 'Y', 'paths' arrays.
        token_map_path (str): Path to the .npy file containing the token2index map.
        train_split_path (str, optional): Path to train split CSV file. Required if validation=True.
        valid_split_path (str, optional): Path to validation split CSV file. Required if validation=True.
        test_split_path (str, optional): Path to test split CSV file. Required if validation=False.
        validation (bool): If True, return (train_loader, valid_loader). If False, return (test_loader, None).
        t_hours, dt, dynamic: Parameters passed to the EHR Dataset class.
        shuffle, pin_memory, batch_size: Parameters for the DataLoader.
        logger: Logger instance.

    Returns:
        tuple: (train_loader, valid_loader) or (test_loader, None)
    """
    pin_memory = pin_memory and torch.cuda.is_available()

    # --- Load Full Data Arrays and Token Map ---
    try:
        logger.info(f"Loading full dataset arrays from: {array_path}")
        # Use np.load directly for .npz file
        arrs = np.load(array_path, allow_pickle=True)
        X_all = arrs['X']
        Y_all = arrs['Y']
        Paths_all = arrs['paths'] # Relative paths from array creation step
        logger.info(f" Loaded data shapes: X={X_all.shape}, Y={Y_all.shape}, Paths={Paths_all.shape}")

        logger.info(f"Loading token map from: {token_map_path}")
        token2index = np.load(token_map_path, allow_pickle=True).item()
        n_tokens = len(token2index)
        logger.info(f" Token map loaded. Vocabulary size (n_tokens): {n_tokens}")

    except FileNotFoundError as e:
        logger.error(f"Error: Required file not found - {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data arrays or token map: {e}")
        raise

    # Create a mapping from relative path to index in the full arrays
    path_to_idx = {path: i for i, path in enumerate(Paths_all)}

    # --- Load Data Based on Splits ---
    if validation:
        if not train_split_path or not valid_split_path:
            raise ValueError("train_split_path and valid_split_path are required when validation=True")

        # Load train split
        try:
            train_df = pd.read_csv(train_split_path)
            train_paths = train_df['Paths'].tolist()
            # Get indices corresponding to train paths
            train_indices = [path_to_idx[p] for p in train_paths if p in path_to_idx]
            if len(train_indices) != len(train_paths):
                 logger.warning(f"Mismatch between paths in {train_split_path} and paths in {array_path}")
            X_train, Y_train = X_all[train_indices], Y_all[train_indices]
            logger.info(f"Loaded training set: {len(X_train)} samples based on {train_split_path}")
        except Exception as e:
            logger.error(f"Error loading training data based on {train_split_path}: {e}")
            raise

        # Load validation split
        try:
            valid_df = pd.read_csv(valid_split_path)
            valid_paths = valid_df['Paths'].tolist()
            # Get indices corresponding to validation paths
            valid_indices = [path_to_idx[p] for p in valid_paths if p in path_to_idx]
            if len(valid_indices) != len(valid_paths):
                 logger.warning(f"Mismatch between paths in {valid_split_path} and paths in {array_path}")
            X_valid, Y_valid = X_all[valid_indices], Y_all[valid_indices]
            logger.info(f"Loaded validation set: {len(X_valid)} samples based on {valid_split_path}")
        except Exception as e:
            logger.error(f"Error loading validation data based on {valid_split_path}: {e}")
            raise

        # Create Datasets
        train_dataset = EHR(X_train, Y_train, n_tokens, t_hours, dt, dynamic, logger)
        valid_dataset = EHR(X_valid, Y_valid, n_tokens, t_hours, dt, dynamic, logger)

        # Create DataLoaders
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      pin_memory=pin_memory,
                                      num_workers=4 if pin_memory else 0) # Use workers if pinning memory
        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=batch_size,
                                      shuffle=False, # No shuffle for validation
                                      pin_memory=pin_memory,
                                      num_workers=4 if pin_memory else 0)

        return train_dataloader, valid_dataloader

    else: # Load test set
        if not test_split_path:
            raise ValueError("test_split_path is required when validation=False")

        # Load test split
        try:
            test_df = pd.read_csv(test_split_path)
            test_paths = test_df['Paths'].tolist()
            # Get indices corresponding to test paths
            test_indices = [path_to_idx[p] for p in test_paths if p in path_to_idx]
            if len(test_indices) != len(test_paths):
                 logger.warning(f"Mismatch between paths in {test_split_path} and paths in {array_path}")
            X_test, Y_test = X_all[test_indices], Y_all[test_indices]
            logger.info(f"Loaded test set: {len(X_test)} samples based on {test_split_path}")
        except Exception as e:
            logger.error(f"Error loading test data based on {test_split_path}: {e}")
            raise

        # Create Dataset
        test_dataset = EHR(X_test, Y_test, n_tokens, t_hours, dt, dynamic, logger)

        # Create DataLoader
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False, # No shuffle for test
                                     pin_memory=pin_memory,
                                     num_workers=4 if pin_memory else 0)

        return test_dataloader, None # Return None for the second value (no valid_loader)

