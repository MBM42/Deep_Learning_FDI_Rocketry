"""
LSTM_inference.py

Standalone script to perform inference using a trained Long Short-Term Recurrent Neural Network.
Computes aggregated metrics over the entire validation set (GPU‑batched), then
launches an interactive per‑simulation plotting loop using raw class indices.

Params:
    plot_features (bool): If set to true also plots the normalized features.
    target_fault (str): The user might lock in which specific fault type the script will look for.
    n_sim_per_file (int): in order to avoid that the script runs through all faults of a specific type, one can
                          can limit the amount of plots per fault.
    best_model (bool): False: Uses last trained model; True: Uses the model with the best metrics

Author: Miguel Marques
Date: 14-04-2025
"""

import ast
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === MODE ===
plot_features = True     # False: to skip features plot
best_model = False       # False: Uses last trained model; True: Uses the model with the best metrics

# Fault Selection
target_fault = None      # e.g. "PLB_12_Block_Leak"
n_sim_per_fault = 3      # max plots per fault type

# Dir Handling
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = Path(current_dir) / "Data" / "Normalized" / "Validation"
final_model_path = Path(current_dir) / "final_model.pth"
best_model_path = Path(current_dir) / "best_model.pth"

if best_model:
    model_path = best_model_path
else:
    model_path = final_model_path

# Add project root to import path, in order to provide the LSTMModel architecture
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from LSTM_main import LSTMModel      # Model Definition


def get_config_values(config_filename: str = "Model_Config.txt"):
    """
    Read window_size, hidden_dim, drop_prob, features_h, labels_h, num_layers from config.
    
    Params:
        config_filename (str): Name of the configuration file (Default: "Model_Config.txt")
    """

    window_size = hidden_dim = drop_prob = features_h = labels_h = num_layers = None
    cfg = Path(current_dir) / config_filename
    with open(cfg, "r") as f:
        for line in f:
            if "- Sequence Window:" in line and window_size is None:
                window_size = int(line.split(":")[1].strip())
            elif "Neurons per Hidden Layer:" in line and hidden_dim is None:
                hidden_dim = int(line.split(":")[1].strip())
            elif "Dropout Rate:" in line and drop_prob is None:
                drop_prob = float(line.split(":")[1].strip())
            elif "Selected features:" in line and features_h is None:
                features_h = ast.literal_eval(line.split(":",1)[1].strip())
            elif "Selected labels:" in line and labels_h is None:
                labels_h = ast.literal_eval(line.split(":",1)[1].strip())
            elif "Number of LSTM Layers:" in line and num_layers is None:
                num_layers = ast.literal_eval(line.split(":",1)[1].strip())
            if all(v is not None for v in [window_size, hidden_dim, drop_prob, features_h, labels_h, num_layers]):
                break
    return window_size, hidden_dim, drop_prob, features_h, labels_h, num_layers


def inference_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    first_pred_conf: dict[str, list[float]],
    first_correct_pred_conf: dict[str, list[float]],
    first_correct_pred_stats: dict[str, list[int]],
    detection_times: dict[str, list[float]],
    exclude_normal: bool = True
) -> None:
    """
    Compute and print:
    - Accuracy (including normal class) 
    - Macro precision/recall/F1 (Normal class excludade if exclude_normal is set to True)
    - Per-class: Precision, 
                 Recall,
                 AvgConf(first non-Normal pred),
                 AvgConf(first correct pred),
                 First-correc-rate = (#first-pred-already-correct) / (#eventual-correct) as %
                 Avg Detection Time

    Params:
        y_true (np.ndarray): Array for the true labels
        y_pred (np.ndarray): Array for the predicted labels
        class_labels (list[str]): List of label headers
        first_pred_conf:          {label: [confidences of first non-Normal pred per file]}
        first_correct_pred_conf:  {label: [confidences of first correct pred per file]}
        first_correct_pred_stats: {label: [n of files with the first pred being the correct one, n of files with correct pred]}
        detection_times:          {label: [detection time per file]}
        exclude_normal (bool): bool to exclude the normal class from precison/recall/F1 calculations
    """

    assert y_true.shape == y_pred.shape, "Shapes of true labels and predictions must match"
    
    # Accuracy calculation
    acc = accuracy_score(y_true, y_pred)
    
    # Exclude Normal class from the remaining performance metrics
    if exclude_normal:
        mask = y_true != 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    # Overall precision / recall / f1 score
    p_macro = precision_score(y_true, y_pred, average='macro', zero_division=0.0)
    r_macro = recall_score(y_true, y_pred, average='macro', zero_division=0.0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0.0)
    print("\n====== Inference Metrics ======")
    print(f"Accuracy:           {acc:.4f}")
    print(f"Precision (macro):  {p_macro:.4f}")
    print(f"Recall    (macro):  {r_macro:.4f}")
    print(f"F1 Score  (macro):  {f1_macro:.4f}")
    
    
    # 2) Per-class table
    print("\nPer-Class Metrics:")
    header = (
        f"{'Class':20s} | {'P':>5s} | {'R':>5s} | "
        f"{'FirstPredConf':>13s} | {'FirstCorrConf':>13s} | "
        f"{'IdentAcc':>11s} | {'AvgDetTime':>10s}"
    )
    print(header)
    print("-" * len(header))

    for cls in range(1, len(class_labels)):
        name = class_labels[cls]
        p_cls = precision_score(y_true, y_pred, labels=[cls], average='macro', zero_division=0.0)
        r_cls = recall_score   (y_true, y_pred, labels=[cls], average='macro', zero_division=0.0)

        fp = first_pred_conf.get(name, [])
        fp_str = f"{sum(fp)/len(fp):.3f}" if fp else "   N/A   "

        fc = first_correct_pred_conf.get(name, [])
        fc_str = f"{sum(fc)/len(fc):.3f}" if fc else "   N/A   "

        dt = detection_times.get(name, [])
        dt_str = f"{sum(dt)/len(dt):.1f}" if dt else "   N/A   "

        first_corr, total_corr = first_correct_pred_stats.get(name, (0,0))
        nr_str = f"{first_corr/ total_corr*100:.1f}%" if total_corr else "   N/A   "

        print(
            f"{name:20s} | {p_cls:5.3f} | {r_cls:5.3f} | "
            f"{fp_str:>13s} | {fc_str:>13s} | "
            f"{nr_str:>11s} | {dt_str:>10s}"
        )
    print("=" * len(header), "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window_size, hidden_dim, drop_prob, features_h, labels_h, num_layers = get_config_values()

    # Load model
    input_dim = len(features_h)
    model = LSTMModel(input_dim, hidden_dim, num_layers, len(labels_h), drop_prob).to(device)
    model.load_state_dict(torch.load(model_path, map_location = device))
    model.eval()

    if not data_path.exists():
        print(f"Folder {data_path} not found.")
        return

    npy_files = sorted(data_path.rglob("*.npy"))
    print(f"Found {len(npy_files)} .npy files.\n")

    # === AGGREGATED METRICS LOOP (GPU‑BATCHED) ===
    all_preds = []
    all_labels = []

    # Detection time by class skipping "Normal"
    detection_times_by_class = { label: [] for label in labels_h[1:] }
    
    # Prediction statistcs dict skipping "Normal"
    first_correct_pred_conf = { label: [] for label in labels_h[1:] }
    first_pred_conf         = { label: [] for label in labels_h[1:] }
    # Each key is mapped to two values [n of files with the first pred being the correct one, n of files with correct pred]
    first_correct_pred_stats     = { label: [0, 0] for label in labels_h[1:] } 

    print("Computing aggregated inference metrics across all simulations...\n")
    for filepath in tqdm(npy_files, desc="Evaluating Inference Metrics", unit="file"):
        # Load Normalized data
        data = np.load(filepath, allow_pickle=True).item()
        features = data["features"]
        labels_arr = data["labels"]

        # True labels indices
        class_idx = np.argmax(labels_arr, axis=1)

        # Skip all‑normal or too short
        if np.all(class_idx == 0) or features.shape[0] < window_size:
            continue

        # true onset
        diff_onsets = np.where(np.diff(class_idx) > 0)[0] + 1
        if len(diff_onsets) > 0:
            onset = diff_onsets[0]
        elif class_idx[0] != 0:
            onset = 0
        else:
            onset = np.where(class_idx != 0)[0][0]
        pred_time = np.arange(window_size - 1, len(class_idx))

        num_windows = features.shape[0] - window_size + 1
        # Stack and flatten
        windows = np.stack([
            features[i : i + window_size]
            for i in range(num_windows)
        ], axis=0)

        with torch.no_grad():
            batch   = torch.tensor(windows, dtype=torch.float32, device=device)
            outputs = model(batch)                                # [num_windows, num_classes]
            probs   = torch.softmax(outputs, dim=1).cpu().numpy() # [num_windows, num_classes]
            preds   = torch.argmax(outputs, dim=1).cpu().numpy()  # [num_windows]

        # Align true‐label at each window (label of last timestamp in the window)
        true_labels = class_idx[window_size - 1:]

        # Find the very first prediciton and first correct prediction in this simulation
        first_pred_done    = False
        first_correct_done = False
        for j, (p, t) in enumerate(zip(preds, true_labels)):
            # 1) First non-Normal prediction
            if not first_pred_done and p != 0:
                cls = labels_h[p]
                first_pred_conf[cls].append(probs[j, p])
                first_pred_done = True
                if p == t:
                    first_correct_pred_stats[cls][0] += 1   # First prediction is the correct one
            # 2) First correct (fault) prediction
            if not first_correct_done and p == t and p != 0:
                cls = labels_h[p]
                first_correct_pred_conf[cls].append(probs[j, p])
                first_correct_pred_stats[cls][1] += 1      # Fault is correctly predicted
                detected_ts = pred_time[j]
                detection_times_by_class[cls].append(detected_ts - onset)
                first_correct_done = True
            
            if first_correct_done:
                break
        
        all_preds.append(preds)
        all_labels.append(true_labels)

    if all_preds and all_labels:
        y_pred_all = np.concatenate(all_preds)
        y_true_all = np.concatenate(all_labels)
        inference_metrics(
            y_true_all, 
            y_pred_all, 
            labels_h, 
            first_pred_conf, 
            first_correct_pred_conf, 
            first_correct_pred_stats,
            detection_times_by_class, 
            exclude_normal=True
        )
    else:
        print("No valid windows collected for global metrics.\n")

    # ===  INTERACTIVE PLOTTING LOOP ===
    fault_counts = {}
    for filepath in npy_files:
        # Load Normalized data
        data = np.load(filepath, allow_pickle=True).item()
        features = data["features"]
        labels_arr = data["labels"]
        class_idx = np.argmax(labels_arr, axis=1)

        # Skip all‑normal or too short
        if np.all(class_idx == 0) or features.shape[0] < window_size:
            continue

        # Determine representative fault for the current simulation
        unique_faults = []
        for i, lbl in enumerate(class_idx):
            if lbl != 0 and lbl not in unique_faults:
                unique_faults.append(lbl)
        rep_fault = unique_faults[0] if unique_faults else None
        
        # Overview approach, limit per-fault plots
        if target_fault is None and rep_fault is not None:
            fault_counts.setdefault(rep_fault, 0)
            if fault_counts[rep_fault] >= n_sim_per_fault:
                continue
            fault_counts[rep_fault] += 1
        
        # If a specific fault has been set, skip non-matching sims
        if target_fault and (rep_fault is None or labels_h[rep_fault] != target_fault):
            continue

        # Batched inference for this file
        num_windows = features.shape[0] - window_size + 1
        windows = np.stack([
            features[i : i + window_size]
            for i in range(num_windows)
        ], axis=0)
        with torch.no_grad():
            batch = torch.tensor(windows, dtype=torch.float32, device=device)
            outputs = model(batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        # Prepare axes
        time_axis = np.arange(len(class_idx))
        pred_time = np.arange(window_size - 1, len(class_idx))

        if plot_features:
            fig, (ax_f, ax_s) = plt.subplots(
                    2, 1, sharex=True,
                    figsize=(20, 12),
                    gridspec_kw={'height_ratios': [2, 1]})
            
            # Plot all the feature curves
            for i in range(features.shape[1]):
                ax_f.plot(time_axis, features[:, i],
                        label=features_h[i], alpha=0.6)
            ax_f.set_ylabel("Normalized Features", fontsize=10)
            ax_f.grid(True)

            # Move the legend outside
            n_feats = features.shape[1]
            ncol    = min(4, n_feats)
            leg_f = ax_f.legend(
                loc='upper left',
                bbox_to_anchor=(1.02, 1),
                ncol=ncol,
                framealpha=0.9,
                fontsize=10
            )
            # Shrink the axes to make room on the right
            fig.subplots_adjust(right=0.75)
        else:
            fig, ax_s = plt.subplots(figsize=(20, 6))

        # Densify label range
        all_lbls = sorted(set(class_idx) | set(preds))
        dense_map = {lbl: i for i, lbl in enumerate(all_lbls)}
        y_true = [dense_map[lbl] for lbl in class_idx]
        y_pred = [dense_map[lbl] for lbl in preds]

        # Plot true labels and predictions
        ax_s.step(time_axis, y_true,
          where="post",
          label="True Label",
          color="blue")
        ax_s.step(pred_time, y_pred,
          where="post",
          label="Prediction",
          color="red",
          alpha=0.6)

        # Annotate True Label Onset
        onsets = np.where(np.diff(class_idx) > 0)[0] + 1
        for onset in onsets:
            x = time_axis[onset]
            y = dense_map[class_idx[onset]]
            ax_s.axvline(x, linestyle="--", color="blue", alpha=0.5)
            ax_s.text(x, y + 0.2, str(onset),
                      ha="center", va="bottom", fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.2",
                                facecolor="white",
                                edgecolor="blue"))
            
        # Annotate Preds == True Label Onset
        match_idx = None
        for j, t_pred in enumerate(pred_time):
            # Find the true‑label index in effect at t_pred
            k = np.searchsorted(time_axis, t_pred, side='right') - 1
            if k < 0:
                continue

            true_lbl = class_idx[k]
            pred_lbl = preds[j]

            # Require both a match AND that it's not the "Normal" class (0)
            if pred_lbl == true_lbl and true_lbl != 0:
                match_idx = j
                break

        if match_idx is not None:
            x_match = pred_time[match_idx]
            y_match = dense_map[preds[match_idx]]

            # Red dashed line
            ax_s.axvline(x_match,
                        linestyle="--",
                        color="red",
                        alpha=0.5)

            # Red box with timestamp
            ax_s.text(x_match,
                    y_pred[0] - 0.3,
                    str(x_match),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2",
                                facecolor="white",
                                edgecolor="red"))


        # y‑axis labels
        ax_s.set_yticks(range(len(all_lbls)))
        ax_s.set_yticklabels([labels_h[lbl] for lbl in all_lbls], fontsize=10)
        ax_s.set_ylim(-0.5, len(all_lbls) - 0.5)
        ax_s.set_xlabel("Timestamp", fontsize=10)
        ax_s.grid(True)
        leg_l = ax_s.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            framealpha=0.9,
            ncol=1,
            fontsize=10
        )
        fig.subplots_adjust(right=0.75)
        fig.suptitle(f"Inference for file: {filepath.name}", fontsize=14, fontweight='bold', y=0.95)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.show()

        # User control
        cmd = input("Press Enter for next, or 'q' to quit: ").strip().lower()
        plt.close(fig)
        if cmd == 'q':
            break

    print("\nPlotting complete. Per‑fault counts:")
    for idx, cnt in fault_counts.items():
        print(f"  {labels_h[idx]}: {cnt}")

if __name__ == "__main__":
    main()
