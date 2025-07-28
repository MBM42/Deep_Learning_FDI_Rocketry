"""
plot_fault.py

Standalone script to plot the occurence of faults (fault state)

It plots the Fault State of a given fault in a simulation as a function of its timestamps.

The script searches for simulations recursively and plots one simulation at a time.
To close the current plot and produce the next simulation's plot the user shall input the required arguments 
in the command line

Parameters:
- folder_path [Path]: path to the folder containing the .npy files
- n_sim_per_fault [int]: This variable limits the number of plots per fault. Once the limit has been
                  reached the script will only look into the remaining fault limits that haven't
                  reached it yes.
- target_fault [str]: If set to "None" the script plots all faults (but "Normal") present in labels_h.
                      If set to a specific fault (e.g.: "PLB_12_Block_Leak") the script will only produce
                      fault plots for this fault type and ignore any limit set by "n_sim_per_fault".

Author: Miguel Marques
Date: 14-04-2025
"""

import ast
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Tuple, List

# Define the folder containing the .npy files
current_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = Path(current_dir) / ".." / "Data" / "Normalized" / "Validation" # Adjust between "Validation" and "Test"
model_path = Path(current_dir) / ".." / "Model_Config.txt"

# Plot limit per fault type
n_sim_per_fault = 1

# Plot Specific Fault if desired; set to None to process all faults
#target_fault: Optional[str] = "PLB_12_Block_Leak"
target_fault: Optional[str] = None

def process_file(filepath: Path, labels_h: list[str], target_fault: Optional[str] = None) -> Optional[Tuple[Figure, List[str]]]:
    """
    Processes a single .npy file to produce a plot of the fault state using.
    Converts the one-hot encoded 'labels' array into a new state array with values:
       0  => Normal
       1,2,... => Fault states in the order they appear
    
    Args:
        filepath (Path): Path to the .npy file.
        target_fault (str) If provided, only files with this fault (in the detected fault list) are processed
        labels_h (list[str]): headers of the classes

    Returns:
        A tuple (fig, unique_fault_names) where:
           - fig is the generated matplotlib Figure.
           - unique_fault_names is the list of unique fault names (excluding Normal) detected in the file.
    """

    # Load file data
    try:
        data = np.load(filepath, allow_pickle=True).item()
    except Exception as e:
        raise ValueError(f"Error loading file {filepath}: {e}")

    # Handle files with unexpected content
    if not isinstance(data, dict) or "labels" not in data:
        raise ValueError(f"File {filepath}: Not a valid dictionary with key 'labels'.")
    labels_arr = data["labels"]
    if labels_arr.ndim != 2 or labels_arr.shape[1] != len(labels_h):
        raise ValueError(f"File {filepath}: Unexpected 'labels' shape. Expected 2D array with {len(labels_h)} columns.")

    # Convert the one-hot encoded rows to class indices
    class_indices = np.argmax(labels_arr, axis=1)

    # Return if file has only Normal timestamps
    if np.all(class_indices == 0):
        return None

    # Collect unique Fault states in order of appearance
    unique_faults = []
    for idx in np.where(class_indices != 0)[0]:
        val = class_indices[idx]
        if val not in unique_faults:
            unique_faults.append(val)
    
    # Build a list of the Fault names corresponding to the undique Fault states
    unique_fault_names = [labels_h[val] for val in unique_faults]

    # If "target_fault" is specified process the file only if the fault is present
    if target_fault is not None:
        if target_fault not in unique_fault_names:
            return None

    # Build mapping: Normal (0) stays 0; each unique fault is assigned an incremental integer starting at 1
    mapping = {0: 0}
    for i, fault_val in enumerate(unique_faults, start=1):
        mapping[fault_val] = i

    # Create new state array using the mapping
    new_state = np.array([mapping.get(x, 0) for x in class_indices])
    
    # Configure y-axis ticks and labels
    num_states = len(unique_faults) + 1  # one for Normal plus one for each fault type
    y_ticks = list(range(num_states))
    y_tick_labels = ["Normal"] + unique_fault_names

    # Create time axis with timestamps from 0 to num_samples - 1
    num_samples = len(class_indices)
    time_axis = np.arange(num_samples)

    # Plot the data.
    fig, ax = plt.subplots()
    ax.step(time_axis, new_state, where="post", linewidth=2)
    ax.set_xlabel("Timestamps")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.grid(True)
    plt.tight_layout()

    return fig, unique_fault_names


def main():

    # Check if folder exists
    if not folder_path.exists():
        print(f"Folder {folder_path} not found.")
        return
    
    # List of label names
    with open(model_path, "r") as f:
        for line in f:
            if "Selected labels:" in line:
                labels_h = ast.literal_eval(line.split(":",1)[1].strip())
                break

    npy_files = list(folder_path.rglob("*.npy"))
    total_files = len(npy_files)
    print(f"Found {total_files} .npy files in folder '{folder_path}'.")

    # Initialize fault_counts
    if target_fault is None:
        fault_counts = {
            label: 0
            for label in labels_h
            # Excludes "Normal" and "*Leak", but includes "*Block_Leak"
            if label != "Normal" and not (label.endswith("Leak") and "Block" not in label) # Excludes "Normal" and "*Leak"
        }

    # Go through the simulations
    for idx, filepath in enumerate(npy_files, start=1):
        result = process_file(filepath, labels_h, target_fault=target_fault)
        
        # Progress Counter
        if idx % 100 == 0:
            print(f"Processed {idx} / {total_files} files.")
        
        if result is None:
            continue
        
        fig, unique_fault_names = result

        if target_fault is None:
            # Rretrieving the type of fault in multi-fault mode
            if len(unique_fault_names) == 1:
                rep_fault = unique_fault_names[0]
            if len(unique_fault_names) == 2: # Assumed to be a Block_Leak simulation, where the first fault is "*Block_Leak"
                rep_fault = unique_fault_names[0] 
            if len(unique_fault_names) == 3: # Assumed to be a Block_Leak simulation, where the second fault is "*Block_Leak"
                rep_fault = unique_fault_names[1]
            if fault_counts[rep_fault] >= n_sim_per_fault:
                plt.close(fig)
                continue
            fault_counts[rep_fault] += 1
            print(f"Processed {fault_counts[rep_fault]} simulation(s) for fault type '{rep_fault}' from file {filepath}")
            
        # Display the plot and await user confirmation
        fig.show()
        user_input = input("Press Enter to continue or type 'q' to exit: ").strip().lower()
        if user_input == "q":
            plt.close(fig)
            print("Exiting.")
            return
        plt.close(fig)

        # Exit if all faults have reached their limit
        if target_fault is None:
            if all(count >= n_sim_per_fault for count in fault_counts.values()):
                print("All fault type counters have reached the limit. Exiting.")
                break
    
    # Print summary of plotted sims
    for key, value in fault_counts.items():
        print(f"{key}: {value}")
            
if __name__ == "__main__":
    main()
