"""
"preprocessing.py"

Converts raw CSV files to .npy (faster to read for model training)

Functionalities:
    - Preselects features and labels through the provided "selected_features" and "selected_labels"
    - Saves data as data types with as few as possible bits. Optimizing memory usage.
    - Optionally clips the fault timestamps to the first "n_first_timestamps" number of timestamps.
      This prepares the data so that training is focused on fault onset.

Author: Miguel Marques
Date: 01-04-2025
"""

import os 
import shutil 
from pathlib import Path 
import pandas as pd 
import numpy as np
from Headers_Pkg import *

# Real Data Directories
# input_dir = Path("/mnt/c/Users/migue/Desktop/Data/Data_Small")
# output_r_path = Path("../Data/Data_Small_npy")

# Debug Directories
input_dir = Path("/mnt/c/Users/migue/Desktop/Debug_Data" )
output_r_path = Path("../Data/Debug_Data_npy")

# Feature and Labels Selection
features_headers = debug_features
labels_headers = debug_labels

# Data Clipping
# - Number of timestamps of each fault to keep
# - If set to None no clipping is performed
n_first_timestamps: int | None = None

# Everything except for the "Normal" column shall be trimmed
fault_headers  = [h for h in labels_headers if h.lower() != "Normal"]
fault_indices  = [labels_headers.index(h) for h in fault_headers]

# ───────────────────────────── Helper Routine ─────────────────────────────
def trim_fault_labels(
    label_array: np.ndarray,
    n: int,
    cols_to_trim: list[int]
) -> np.ndarray:
    """
    For each column index in *cols_to_trim* keep only the first *n* consecutive
    '1's of *every* positive segment; convert the remainder of that segment to 0.

    Parameters
    - label_array : ndarray of shape (timesteps, n_labels) : Original labels.
    - n           : int : Number of timesteps to keep from the start of each segment.
    - cols_to_trim: list[int] : Numeric indices of the label columns to trim.

    Returns
    - ndarray with the same shape and dtype as *label_array*.
    """

    if n is None or n <= 0 or not cols_to_trim:
        return label_array

    out  = label_array.copy()
    rows = out.shape[0]

    for col in cols_to_trim:
        col_vec = out[:, col]
        # indices where the value changes (prepend 0 so diff works at index 0)
        change_idx = np.flatnonzero(np.diff(np.concatenate(([0], col_vec))))
        # starts of 1-segments are the even-positioned indices in change_idx
        for start_pos, start_idx in enumerate(change_idx[::2]):
            end_idx = (change_idx[2 * start_pos + 1]
                       if 2 * start_pos + 1 < len(change_idx)
                       else rows)            # open segment to the end
            keep_until = min(start_idx + n, end_idx)
            col_vec[keep_until:end_idx] = 0   # zero-out the remainder
        out[:, col] = col_vec

    return out

# ───────────────────────────── Main ─────────────────────────────
def main():

    # Dir Handling 
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Get the current directory path
    output_dir = current_dir / output_r_path
    
    # Check that the input directory exists.
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist.")
        return

    # If the output directory already exists: ask user if he wants to overwrite it
    if output_dir.exists():
        answer = input(f"{output_dir} already exists. Overwrite its content? (y/n): ").strip().lower()
        if answer != 'y':
            print("Exiting without changes.")
            return
        else:
            shutil.rmtree(output_dir)
            print(f"Deleted existing {output_dir}")

    # Create the output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recursively find all CSV files in the input directory
    csv_files = list(input_dir.rglob("*.csv"))
    total_files = len(csv_files)
    print(f"Found {total_files} CSV files to process.")

    for i, csv_file in enumerate(csv_files, 1):
        # Compute the relative path (e.g., /Block/HFM_01_2.*/.*\.csv)
        rel_path = csv_file.relative_to(input_dir)
        # Build the corresponding output file path
        output_file = output_dir / rel_path
        output_file = output_file.with_suffix('.npy')
        # Ensure that the output directory for this file exists.
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Read Original Raw CSV
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue
        
        # Extracting Features and Labels
        try:
            # Select only the desired columns.
            features = df[features_headers].values.astype(np.float32)
            # Convert labels from bool to int8.
            labels = df[labels_headers].values.astype(bool).astype(np.int8)
        except Exception as e:
            print(f"Error processing columns in {csv_file}: {e}")
            continue
        
        # Optional data clipping
        if n_first_timestamps is not None:
            labels = trim_fault_labels(labels, n_first_timestamps, fault_indices)
        
        # Saving as binary file in NumPy
        data = {"features": features, "labels": labels, "feature_names": features_headers, "labels_names": labels_headers}
        try:
            np.save(output_file, data)
        except Exception as e:
            print(f"Error saving {output_file}: {e}")
        
        # Progress Feedback
        if i % 100 == 0 or i == total_files:
            print(f"Processed {i}/{total_files} files.")

if __name__ == "__main__":
    main()
