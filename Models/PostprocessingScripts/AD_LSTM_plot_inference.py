"""
AD_LSTM_plot_inference.py

Standalone script that loads a trained LSTMPredictor model for Anomaly Detection (AD) that launches an interactive per‑simulation plotting loop.

Params:
    plot_features (bool): If set to true also plots the normalized features.
    target_fault (str): The user might lock in which specific fault type the script will look for.
    n_sim_per_file (int): in order to avoid that the script runs through all faults of a specific type, one can
                          can limit the amount of plots per fault.
Author: Miguel Marques
Date: 05-05-2025
"""

import os
import sys
import ast
from pathlib import Path
import numpy as np
import torch
import joblib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# === USER CONTROL ===
plot_features = True     # False: to skip features plot

# === Fault Selecton ===
n_simulations = 3        # Max number of files to plot; None for no limit
target_fault = None      # e.g. "PLB_12_Block_Leak"

# === DIR HANDLING ===
current_dir = os.path.dirname(os.path.abspath(__file__))
# Paths to model and artifacts
model_path = Path(current_dir) / "best_model.pth"
cov_path   = Path(current_dir) / "cov.pkl"
threshold_path = Path(current_dir) / "threshold.pkl"
scaler_path  = Path(current_dir) / "Data" / "scaler.pkl"
# Path to test .npy files (anomaly data)
test_data_path = Path(current_dir) / ".." / ".." / ".." / "Data" / "New_Data_npy"

# Import model definition
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from AD_LSTM_main import LSTMPredictor

# === CONFIG UTILS ===
def get_config_values(config_filename: str = "Model_Config.txt"):
    """
    Read window_size, pred_length, features_h, labels_h, batch_size from config.
    """
    window_size = pred_length = features_h = labels_h = batch_size = None
    cfg = Path(current_dir) / config_filename
    with open(cfg, "r") as f:
        for line in f:
            if "- window_size (p):" in line and window_size is None:
                window_size = int(line.split(":")[1].strip())
            elif "- pred_length (l):" in line and pred_length is None:
                pred_length = int(line.split(":")[1].strip())
            elif "Selected features:" in line and features_h is None:
                features_h = ast.literal_eval(line.split(":",1)[1].strip())
            elif "Selected labels:" in line and labels_h is None:
                labels_h = ast.literal_eval(line.split(":",1)[1].strip())
            elif "Batch Size:" in line and batch_size is None:
                batch_size = int(line.split(":")[1].strip())
            if all(v is not None for v in [window_size, pred_length, features_h, labels_h, batch_size]):
                break
    return window_size, pred_length, features_h, labels_h, batch_size

# === INFER UTILS ===
def infer_file(feats, cls, model, cov, threshold, device, batch_size, window_size):
    num_windows = feats.shape[0] - window_size
    windows = np.stack([feats[i:i+window_size+1] for i in range(num_windows)], axis=0)
    inputs  = windows[:, :window_size, :]
    targets = windows[:, window_size, :]
    true_labels = cls[window_size:]
    # Batch inference
    x = torch.tensor(inputs, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False, pin_memory=True)
    y_pred = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            out = model(xb)            # [B, pred_length, F]
            forecast = out[:, 0, :].cpu().numpy()
            real_next = targets[offset:offset+forecast.shape[0]]
            offset += forecast.shape[0]
            err = (real_next - forecast).reshape(forecast.shape[0], -1)
            dists = cov.mahalanobis(err)
            flags = (dists > threshold).astype(int)
            y_pred.extend(flags.tolist())
    return true_labels.tolist(), y_pred

# === MAIN ===
def main():
    # Load configuration
    window_size, pred_length, features_h, labels_h, batch_size = get_config_values()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model and artifacts
    model = LSTMPredictor(len(features_h), hidden_dim=None, num_layers=None, pred_length=pred_length, drop_prob=None)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    cov = joblib.load(cov_path)
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold file not found - Run AD_LSTM-inference.py first")
    threshold = joblib.load(threshold_path)
    # Load scaler
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    # Gather test files
    test_npy = [p for p in test_data_path.rglob("*.npy") if "Normal" not in p.parts]
    print(f"Found {len(test_npy)} test simulations for plotting.")
    plotted = 0
    for filepath in test_npy:
        if n_simulations and plotted >= n_simulations:
            break
        # Load data
        data = np.load(filepath, allow_pickle=True).item()
        feats = data['features']
        labs  = data['labels']
        f_names = data.get('feature_names', [])
        l_names = data.get('labels_names', [])
        # Select features and labels
        idxs = [f_names.index(f) for f in features_h]
        feats = feats[:, idxs]
        l_idxs = [l_names.index(l) for l in labels_h]
        cls = np.argmax(labs[:, l_idxs], axis=1)
        # Determine representative fault for this simulation
        non_zero = [lbl for lbl in cls if lbl != 0]
        rep_fault_idx = non_zero[0] if non_zero else None
        rep_fault_name = labels_h[rep_fault_idx] if rep_fault_idx is not None else None
        # Filter by target_fault if set
        if target_fault and rep_fault_name != target_fault:
            continue
        # Normalize features
        feats = scaler.transform(feats)
        # Inference
        y_true, y_pred = infer_file(feats, cls, model, cov, threshold, device, batch_size, window_size)
        # Plotting
        time_axis = np.arange(len(cls))
        pred_time = np.arange(window_size, len(cls))
        if plot_features:
            fig, (ax_f, ax_s) = plt.subplots(2,1,sharex=True,figsize=(12,8),gridspec_kw={'height_ratios':[2,1]})
            for i in range(feats.shape[1]):
                ax_f.plot(time_axis, feats[:,i], label=features_h[i], alpha=0.6)
            ax_f.set_ylabel("Normalized Features")
            ax_f.grid(True)
            ax_f.legend(loc='upper left', bbox_to_anchor=(1.02,1),framealpha=0.9)
            fig.subplots_adjust(right=0.75)
        else:
            fig, ax_s = plt.subplots(figsize=(12,4))
        # Binary true labels (normal vs fault)
        true_bin = (cls != 0).astype(int)
        ax = ax_s if plot_features else fig.axes[0]
        # Plot true anomalies and predictions
        ax.step(time_axis, true_bin, where='post', label='True Fault', color='blue')
        ax.step(pred_time, y_pred, where='post', label='Predicted Anomaly', color='red', alpha=0.6)
        # Annotate onset and detection
        onsets = np.where(np.diff(cls) > 0)[0] + 1
        for onset in onsets:
            ax.axvline(onset, linestyle='--', color='blue', alpha=0.5)
            ax.text(onset, 1.05, str(onset), ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='blue'))
        # First detection
        try:
            det_idx = y_pred.index(1)
            det_time = pred_time[det_idx]
            ax.axvline(det_time, linestyle='--', color='red', alpha=0.5)
            ax.text(det_time, -0.1, str(det_time), ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='red'))
        except ValueError:
            pass
        ax.set_yticks([0,1])
        ax.set_yticklabels(['Normal','Fault'])
        ax.set_xlabel('Timestamp')
        ax.set_ylim(-0.2,1.2)
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02,1),framealpha=0.9)
        fig.subplots_adjust(right=0.75)
        title = f"Inference for file: {filepath.name}"
        if rep_fault_name:
            title += f" (Fault: {rep_fault_name})"
        fig.suptitle(title, fontsize=12)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        fig.show()
        cmd = input("Press Enter for next, or 'q' to quit: ").strip().lower()
        plt.close(fig)
        if cmd == 'q':
            break
        plotted += 1
    print(f"\nPlotting complete. Total files plotted: {plotted}")

if __name__ == "__main__":
    main()
