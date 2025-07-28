"""
ad_aux.py

This module provides complementary functions to contextualize the training of an "LSTMPredictor" anomaly detection model.

Author: Miguel Marques
Date: 03-05-2025
"""

import numpy as np
from sklearn.metrics import *
from typing import Tuple, Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use('TkAgg')
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter, defaultdict
from tqdm import tqdm

def compute_detection_time(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[Optional[int], Optional[int]]:
    """
    Compute detection time and fault label for a simulation.

    Parameters:
        y_true (np.ndarray): 1D array of true labels (0 = normal, !=0 = fault).
        y_pred (np.ndarray): 1D array of binary predictions (0 = normal, 1 = anomaly).

    Returns:
        det_time    (int or None): detect_idx − onset_idx, or None
        fault_label (int or None): index of the first true fault, or None
    """
    try:
        onset_idx = next(i for i, lbl in enumerate(y_true) if lbl != 0)
        fault_label = int(y_true[onset_idx])
    except StopIteration:
        return None, None

    try:
        detect_idx = next(i for i, pr in enumerate(y_pred) if pr == 1)
    except StopIteration:
        return None, fault_label

    return detect_idx - onset_idx, fault_label

def compute_test_stats(npy_paths, selected_labels):
    """
    Compute per-class instance counts, file counts, and overall summary metrics for a list of .npy files.

    Args:
        npy_paths (list of str):
            Paths to .npy files, each containing a dict with a 'labels' key holding
            a sequence of class labels for each time step.
        selected_labels (list[str]):
            List of class names corresponding to one-hot columns or integer label indices (0-based).

    Returns:
        instance_counts (Counter):
            Mapping from each class label to the total number of timestamp occurrences.
        file_counts (dict):
            Mapping from each class label to the number of distinct files in which
            a label appears at least once.
        total_points (int):
            Total number of timestamps processed across all files and classes.
        total_normal (int):
            Total number of 'Normal' timestamps.
        total_faults (int):
            Total number of 'Fault' timestamps.
        imbalance (float):
            Timestamps ratio: 'Fault' / 'Normal' indicating the degree of class imbalance.
    """
    sel_list = list(selected_labels)
    sel_set = set(sel_list)

    instance_counts = Counter()
    file_counts = defaultdict(int)

    for filepath in npy_paths:
        data = np.load(filepath, allow_pickle=True).item()
        labels = data.get('labels')
        if labels is None:
            continue
        labels = np.array(labels)

        counts = labels.sum(axis=0)
        for idx, cnt in enumerate(counts):
            cls_name = sel_list[idx]
            if cls_name in sel_set and cnt > 0:
                instance_counts[cls_name] += int(cnt)
                file_counts[cls_name] += 1

    total_points  = sum(instance_counts.values())
    total_normal  = instance_counts.get('Normal', instance_counts.get(0, 0))
    total_faults  = total_points - total_normal
    imbalance     = (abs(total_normal - total_faults) / total_points * 100)  if total_normal else float('inf')

    return instance_counts, file_counts, total_points, total_normal, total_faults, imbalance

def compute_train_stats(dataset):
    """
    Compute number of simulation files and total timestamps in train_cache and val_cache.

    Args:
        dataset: An object with attributes `train_cache` and `val_cache`, each a list of arrays
                 of shape (T_i, F), where T_i is the number of timestamps in simulation i.

    Returns:
        num_train_files (int): Number of simulations in train_cache.
        total_train_timestamps (int): Sum of timestamps across all train simulations.
        num_val_files (int): Number of simulations in val_cache.
        total_val_timestamps (int): Sum of timestamps across all validation simulations.
    """
    train_cache = getattr(dataset, 'train_cache', [])
    val_cache   = getattr(dataset, 'val_cache', [])

    num_train_files      = len(train_cache)
    total_train_timestamps = sum(arr.shape[0] for arr in train_cache)

    num_val_files        = len(val_cache)
    total_val_timestamps   = sum(arr.shape[0] for arr in val_cache)

    return num_train_files, total_train_timestamps, num_val_files, total_val_timestamps

def performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    filter_normal: bool
) -> object:
    """
    Compute inference metrics and return an object with these as attributes.

    Parameters:
        y_true (np.ndarray): ground truth labels
        y_pred (np.ndarray): predicted labels
        labels (list[str]): labels names (via indexing we retrieve their name)
        filter_normal (bool): if set to True the 'Normal' class is excluded from all metrics but acc. and MCC.

    Returns object with the following attributes:
        acc (float)
        precision_macro (float)
        recall_macro (float)
        f1_macro (float)
        mcc (float)
        precision_per_class (dict[str, float])
        recall_per_class (dict[str, float])
    """
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Filter Normal class
    if filter_normal:
        mask = y_true != 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    # Remaining Overall Metrics
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per class metrics
    precision_per_class = {}
    recall_per_class = {}
    for cls_name in labels[1:]:
        idx = labels.index(cls_name)
        y_true_c = (y_true == idx)
        precision_per_class[cls_name] = precision_score(y_true_c, y_pred, zero_division=0)
        recall_per_class[cls_name] = recall_score(y_true_c, y_pred, zero_division=0)
    
    class Metrics:
        pass
    metrics = Metrics()
    metrics.acc = float(acc)
    metrics.precision_macro = float(precision_macro)
    metrics.recall_macro = float(recall_macro)
    metrics.f1_macro = float(f1_macro)
    metrics.mcc = float(mcc)
    metrics.precision_per_class = precision_per_class
    metrics.recall_per_class = recall_per_class
    return metrics

def infer_file(
    feats: np.ndarray,
    cls:   np.ndarray,
    model: torch.nn.Module,
    cov,
    threshold: float,
    device: torch.device,
    batch_size: int,
    window_size: int
) -> tuple[list[int], list[int]]:
    """
    Batched inference over one simulation:
      - Builds overlapping windows of length window_size+1 so we can predict the next timestamp
      - Feeds them through the LSTM in batches of size batch_size
      - For each window, compares the 1-step forecast to the true next timestamp
      - Scores anomaly via Mahalanobis distance against `cov`

    Args:
      feats       (T×F array):   input feature time-series
      cls         (T-vector):    true class indices per timestamp
      model       : trained LSTMPredictor
      cov         : EmpiricalCovariance fitted on training errors
      threshold   : anomaly threshold (Mahalanobis distance)
      device      : cpu or cuda
      batch_size  : number of windows per forward pass
      window_size : how many history steps the model consumes

    Returns:
      y_true (List[int]):  the true labels for each predicted “next” timestamp  
      y_pred (List[int]):  0/1 anomaly flags per timestamp
    """

    num_windows = feats.shape[0] - window_size
    # Stack and flatten
    windows = np.stack([
        feats[i : i + window_size + 1]
        for i in range(num_windows)
    ], axis=0)

    inputs      = windows[:, :window_size, :]   # (n_windows, window_size, F)
    targets     = windows[:,  window_size, :]   # (n_windows,            F)
    true_labels = cls[window_size:]             # align labels to the “next” step   

    # Batch Windows
    x = torch.tensor(inputs, dtype=torch.float32)
    ds = TensorDataset(x)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    y_true, y_pred = [], []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(device)                   # [B, window_size, F]
            out = model(xb)                            # [B, 1, F]
            forecast = out[:, 0, :].cpu().numpy()      # [B, F]

            real_next = targets[offset : offset + forecast.shape[0]]
            offset += forecast.shape[0]

            err = (real_next - forecast).reshape(forecast.shape[0], -1)
            dists = cov.mahalanobis(err)
            flags = (dists > threshold).astype(np.int8)

            y_pred.extend(flags.tolist())

    y_true = true_labels.tolist()
    return y_true, y_pred

def plot_loss_curve(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    save_path: Path
) -> None:
    """
    Plot and save train/val loss curves

    Parameters:
        train_losses: list of per-epoch training losses
        val_losses:   list of per-epoch validation losses
        save_path:    path to the folder where 
    """
    # Full path
    save_path = save_path / "LossPlot.png"

    epochs = list(range(1, len(train_losses) + 1))
    ticks = epochs  # x-axis ticks at each epoch

    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, '-', color=mcolors.TABLEAU_COLORS['tab:orange'], label="Train")
    ax.plot(epochs, val_losses,   '-', color=mcolors.TABLEAU_COLORS['tab:green'],  label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.set_xlim(left=0, right=len(epochs) + 1)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
