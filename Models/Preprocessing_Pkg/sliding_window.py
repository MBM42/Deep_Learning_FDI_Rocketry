"""
sliding_window.py

This module implements the SlidingWindow class, a Dataset subclass that generates sliding window samples from cached .npy data on the fly.

Each .npy file represents one simulation; sliding windows are computed within each file so that windows never cross simulation boundaries. 
The label for a window corresponds to the label of the last timestamp in that  window. 
Global normalization is applied using precomputed statistics.

It provides methods for:
- Saving a specified number of window samples from a simulation for preprocessing debug: save_debug_samples()

    
Author: Miguel Marques
Date: 29-03-2025
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from Preprocessing_Pkg.npy_preprocessing import NPYDataset
from typing import List, Dict, Any
from tqdm import tqdm
from FNN_settings import *


class SlidingWindow(Dataset):
    """
    Parameters:
        data_list (List[Dict[str, np.ndarray]]): List of cached data dicts (each with keys "features" and "labels").
        settings (FNNSettings)    
        npy_dataset(NPYDataset): instance of NPYDataset class to provide global normalization statistics (global_mean, global_std, global_min, global_max).

    Extra Attributes:
        data_list (List[Dict[str, np.ndarray]]): List of cached data dicts (each with keys "features" and "labels").
        window_size (int): Number of consecutive timestamps per sliding window.
        scaler_type (str): 'zscore' for z-score normalization or 'minmax' for min–max normalization.
                        -zscore: Removes the mean to center the data and divides by the standard deviation to scale it to a variance of 1.
                        -minmax: Scales the data to a fixed range [0, 1].
        flatten_window (bool): If True, flatten each window into a 1D tensor.
        windows_per_file (List[int]): Number of sliding windows for each cached data entry.
        cumulative_windows (List[int]): Cumulative window counts for mapping global indices.
    """
    
    def __init__(
        self,
        data_list: List[Dict[str, np.ndarray]],
        settings: FNNSettings,
        npy_dataset: NPYDataset,
    ) -> None:
        self.data_list = data_list
        self.window_size = settings.window_size
        self.scaler_type = settings.scaler_type.lower()
        self.flatten_window = settings.flatten_window
        self.npy_dataset = npy_dataset

        # Global normalization statistics from npy_dataset
        self.global_mean = npy_dataset.global_mean  # Global mean vector (shape: (feature_dim,))
        self.global_std = npy_dataset.global_std    # Global std vector (shape: (feature_dim,))
        self.global_min = npy_dataset.global_min    # Global min vector for min–max scaling.
        self.global_max = npy_dataset.global_max    # Global max vector for min–max scaling.

        # Precompute the number of sliding windows per cached data entry and its the cumulative indices.
        """
        self.windows_per_file: List[int]   # Each index holds the number of sliding windows of the corresponding file
        self.cumulative_windows: List[int] # Each index holds the upper global index (of the total available windows) for the corresponding file  
        
        Example, if self.window_size is 50 and one has three .npy files with 100, 150, and 120 rows:
        - File 1: num_windows = max(0, 100 - 50 + 1) = 51
        - File 2: num_windows = max(0, 150 - 50 + 1) = 101
        - File 3: num_windows = max(0, 120 - 50 + 1) = 71
        
          Then:
            self.windows_per_file = [51, 101, 71]
            self.cumulative_windows = [51, 152, 223]
        
        The cumulative_windows list allows to map a global window idx (provided by the DataLoader) to the specific .npy file and the local idx within that file.
        """
        cumulative_total = 0
        self.windows_per_file = []
        self.cumulative_windows = []
        for data in tqdm(self.data_list, desc="Precomputing number of sliding windows and global idx, per file", unit="file"):
            features = data["features"]
            num_rows = features.shape[0]
            num_windows = max(0, num_rows - self.window_size + 1)
            self.windows_per_file.append(num_windows)
            cumulative_total += num_windows
            self.cumulative_windows.append(cumulative_total)
        # Convert from list to np.array for faster search in __getitem__
        self.cumulative_windows = np.array(self.cumulative_windows) 

    def __len__(self) -> int:
        """Returns the total number of sliding window samples available."""
        return int(self.cumulative_windows[-1]) if self.cumulative_windows.size > 0 else 0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a sample (sliding window).

        The label for a window is taken from the last timestamp in that window.
        Normalization is applied using the specified scaler_type.
        """

        # Ensure self.data_list is non-empty
        if not self.data_list:
            raise ValueError("No cached data available in self.data_list.")
        
        # Determine which file (simulation) the index corresponds to (binary search)
        file_index = int(np.searchsorted(self.cumulative_windows, idx, side="right"))
        local_idx = idx if file_index == 0 else idx - self.cumulative_windows[file_index - 1]

        data = self.data_list[file_index]
        features = data["features"]
        labels = data["labels"]

        if local_idx + self.window_size > features.shape[0]:
            raise IndexError("Requested window exceeds available data rows.")

        window = features[local_idx: local_idx + self.window_size]

        # Apply normalization.
        if self.scaler_type == "zscore":
            window = (window - self.global_mean) / self.global_std
        elif self.scaler_type == "minmax":
            denom = self.global_max - self.global_min
            denom[denom == 0] = 1.0
            window = (window - self.global_min) / denom
        else:
            raise ValueError("Unknown scaler_type. Use 'zscore' or 'minmax'.")

        # Extract label from the last timestamp of the window.
        label_vector = labels[local_idx + self.window_size - 1]
        label_val = np.argmax(label_vector)

        # Flatten the window if required.
        window_out = window.reshape(-1) if self.flatten_window else window

        return torch.tensor(window_out, dtype=torch.float32), torch.tensor(label_val, dtype=torch.long)


    def save_debug_samples(self, fault_type: str, num_windows: int, model_dir_path: Path, output_file: str = "debug_samples.txt") -> None:
        """
        Saves a specified number of window samples from a cached data entry for debugging purposes.
        
        The method searches for a cached data dictionary whose 'source' key (if available) contains the given fault_type.
        If not found, it uses the first cached data entry. It then extracts raw windows, applies normalization based on
        the scaler_type, and writes the raw and normalized windows along with the label (from the last timestamp of each window)
        to a text file.
        
        Parameters:
            fault_type (str): keyword to select a cached data entry based on its source.
            num_windows (int): number of windows to extract.
            model_dir_path (Path): model directory where the debug file will be saved.
            output_file (str): [Optional] Name of the output txt file.
        """
        
        # Attempt to select a cached data entry matching the fault_type (if a 'source' key exists)
        debug_data = None
        for data in self.data_list:
            if "source" in data and fault_type in str(data["source"]):
                debug_data = data
                break
        if debug_data is None:
            search_result = str(f"No cached data matching fault_type '{fault_type}' found; First cached entry used instead.")
            # Fallback to the first cached data entry
            debug_data = self.data_list[0]
        else:
            search_result = str(f"Cached data matching fault_type '{fault_type}' found.")
        
        # Extract features and labels
        raw_features = debug_data["features"]
        labels = debug_data["labels"]

        num_rows = raw_features.shape[0]
        max_available_windows = max(0, num_rows - self.window_size + 1)

        # Check and correct if requested windows exceed available windows
        if num_windows > max_available_windows:
            num_windows = max_available_windows
        if num_windows == 0:
            raise ValueError("No windows can be generated with the given window size and available data length.")

        # Prepare to save windows and labels
        windows, normalized_windows, window_labels = [], [], []
        
        for idx in range(num_windows):
            window = raw_features[idx: idx + self.window_size]
            windows.append(window)

            # Apply normalization
            if self.scaler_type == "zscore":
                norm_window = (window - self.npy_dataset.global_mean) / self.npy_dataset.global_std
            elif self.scaler_type == "minmax":
                denom = self.npy_dataset.global_max - self.npy_dataset.global_min
                denom[denom == 0] = 1.0
                norm_window = (window - self.npy_dataset.global_min) / denom
            else:
                raise ValueError("Unknown scaler_type. Use 'zscore' or 'minmax'.")
            normalized_windows.append(norm_window)

            # Label from the last timestamp in the window
            label = labels[idx + self.window_size - 1]
            window_labels.append(label)

        # File path
        output_path = model_dir_path / output_file
        
        # Writing to the output file
        with open(output_path, 'w') as f:
            f.write("Debug samples extracted from cached data\n")
            f.write(search_result)
            for i, (window, norm_window, label) in enumerate(zip(windows, normalized_windows, window_labels)):
                f.write(f"Window {i + 1}:\n")
                f.write("Raw window:\n")
                np.savetxt(f, window, fmt="%.5e")
                f.write("\nNormalized window:\n")
                np.savetxt(f, norm_window, fmt="%.5e")
                f.write("\nLabel:\n")
                f.write(f"{label.tolist()}\n")
                f.write("-" * 80 + "\n")
