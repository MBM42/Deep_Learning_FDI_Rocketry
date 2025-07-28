"""
debug_plot_fault.py

Script to visualize every preprocessed simulation (.npy) by plotting both:
  1) The z-score–normalized features over time (line plot)
  2) The ground truth fault state over time (step plot)

Each figure has two stacked subplots (shared x-axis):
  - Top: normalized feature1...feature5 vs. timestamp index
  - Bottom: state (0=Normal,1-4=Fault) with onset marker

Usage:
  Adjust `folder_path` below to point to your .npy data directory, then run:
      python debug_plot_fault_and_features.py
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# === USER CONFIGURATION ===
current_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = Path(current_dir) / ".." / "Data" / "Debug_Data_npy"

# Labels for the state plot
LABEL_NAMES = [
    "Normal",
    "Fault1",
    "Fault2",
    "Fault3",
    "Fault4",
]
# Feature column names
FEATURE_NAMES = [f"feature{i+1}" for i in range(5)]


def plot_simulation(filepath: Path):
    """
    Loads a .npy file and returns a matplotlib Figure with two subplots:
      - Top: z-score normalized features over time
      - Bottom: ground-truth state with fault onset marker
    """
    # Load data
    try:
        data = np.load(filepath, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading {filepath.name}: {e}")
        return None

    # Validate structure
    if not isinstance(data, dict) or 'features' not in data or 'labels' not in data:
        print(f"File {filepath.name} missing 'features' or 'labels'.")
        return None

    features = data['features']  # shape: (T, 5)
    labels = data['labels']      # shape: (T, 5)
    if features.ndim != 2 or labels.ndim != 2 or features.shape[0] != labels.shape[0]:
        print(f"Unexpected shapes in {filepath.name}: features{features.shape}, labels{labels.shape}")
        return None

    # Z-score normalization per feature
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    stds[stds == 0] = 1.0
    features_norm = (features - means) / stds

    # Compute state indices and onset
    class_indices = np.argmax(labels, axis=1)
    onset_idxs = np.where(class_indices != 0)[0]
    onset_idx = onset_idxs[0] if onset_idxs.size > 0 else None
    rep_fault = class_indices[onset_idx] if onset_idx is not None else 0

    T = class_indices.size
    time_axis = np.arange(T)

    # Create figure with 2 subplots
    fig, (ax_feat, ax_state) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 8),
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # Top: normalized features
    for i in range(features_norm.shape[1]):
        ax_feat.plot(time_axis, features_norm[:, i], label=FEATURE_NAMES[i])
    ax_feat.set_ylabel('Z-score normalized feature')
    ax_feat.legend(loc='upper right', ncol=2)
    ax_feat.grid(True)

    # Bottom: state
    ax_state.step(time_axis, class_indices, where='post', linewidth=1.5)
    ax_state.set_xlabel('Timestamp Index')
    ax_state.set_ylabel('State')

    # Fault onset marker
    if onset_idx is not None:
        for ax in (ax_feat, ax_state):
            ax.axvline(onset_idx, color='red', alpha=0.5, linestyle='--', linewidth=1)
        # Annotate onset index
        ylim = ax_state.get_ylim()
        ax_state.text(onset_idx, ylim[1], f'{onset_idx}',
                      color='red', ha='center', va='bottom')

    ax_state.set_yticks(range(len(LABEL_NAMES)))
    ax_state.set_yticklabels(LABEL_NAMES)
    ax_state.grid(True)

    # Title
    fig.suptitle(f"{filepath.name} — {LABEL_NAMES[rep_fault]}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def main():
    npy_path = Path(folder_path)
    if not npy_path.exists():
        print(f"Directory '{folder_path}' does not exist.")
        return

    files = sorted(npy_path.rglob('*.npy'))
    print(f"Found {len(files)} .npy files in '{folder_path}'")

    for filepath in files:
        fig = plot_simulation(filepath)
        if fig is None:
            continue
        fig.show()
        user = input("Press Enter to continue or 'q' to quit: ").strip().lower()
        if user == 'q':
            plt.close(fig)
            print("Exiting.")
            return
        plt.close(fig)

    print("Finished plotting all simulations.")


if __name__ == '__main__':
    main()
