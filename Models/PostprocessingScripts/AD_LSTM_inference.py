"""
AD_LSTM_inference.py

Standalone script that loads a trained LSTMPredictor model for Anomaly Detection (AD) and performs the following tasks:
- Setting an AD threshold;
- Performs AD inference and gathers the performance metrics;
- Plots inference of simulations iteratively

Author: Miguel Marques
Date: 05-05-2025
"""

import os
import ast
import sys
import torch
import joblib
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import *

# AD Threshold Settings
PERCENTILE = 99

# Dir Handling
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = Path(current_dir) / "Data" / "Normalized"
model_path = Path(current_dir) / "best_model.pth"
cov_path = Path(current_dir) / "cov.pkl"
m_dists_path = Path(current_dir) / "m_dists.npy"
scaler_path = Path(current_dir) / "Data" / "scaler.pkl"
test_data_path = Path(current_dir) / ".." / ".." / ".." / "Data" / "New_Data_npy"

# Creating loggers and respective handlers
logger = logging.getLogger("logger_inf")        # Logger object for console
logger.setLevel(logging.DEBUG)                  # DEBUG and above 
console_handler = logging.StreamHandler()       # Console handler
console_handler.setLevel(logging.DEBUG)         # DEBUG and above
logger.addHandler(console_handler)

# Add project root to import path, in order to provide the AD_LSTM architecture
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from AD_LSTM_main import LSTMPredictor # Model Definition

def get_config_values(config_filename: str = "Model_Config.txt"):
    """
    Read window_size, pred_length, num_layers, hidden_dim, drop_prob, features_h, labels_h from config.
    
    Params:
        config_filename (str): Name of the configuration file (Default: "Model_Config.txt")
    """

    window_size = pred_length = num_layers = hidden_dim = drop_prob = features_h = labels_h = batch_size = filter_normal = None
    cfg = Path(current_dir) / config_filename
    with open(cfg, "r") as f:
        for line in f:
            if "- window_size (p):" in line and window_size is None:
                window_size = int(line.split(":")[1].strip())
            elif "- pred_length (l):" in line and pred_length is None:
                pred_length = int(line.split(":")[1].strip())
            elif "Number of LSTM Layers:" in line and num_layers is None:
                num_layers = int(line.split(":")[1].strip())
            elif "Neurons per Hidden Layer:" in line and hidden_dim is None:
                hidden_dim = int(line.split(":")[1].strip())
            elif "Dropout Rate:" in line and drop_prob is None:
                drop_prob = float(line.split(":")[1].strip())
            elif "Selected features:" in line and features_h is None:
                features_h = ast.literal_eval(line.split(":",1)[1].strip())
            elif "Selected labels:" in line and labels_h is None:
                labels_h = ast.literal_eval(line.split(":",1)[1].strip())
            elif "Batch Size:" in line and batch_size is None:
                batch_size = int(line.split(":")[1].strip())
            elif "- Exclude Normal Class:" in line and filter_normal is None:
                filter_normal = ast.literal_eval(line.split(":", 1)[1].strip())
            if all(v is not None for v in [window_size, pred_length, num_layers, hidden_dim, drop_prob, features_h, labels_h, batch_size, filter_normal]):
                break
    return window_size, pred_length, num_layers, hidden_dim, drop_prob, features_h, labels_h, batch_size, filter_normal

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


def log_performance(
    detection_times_by_class,
    missed_detections_by_class,
    p_metrics,
    logger=None
):
    """
    Log per-class average detection times, missed detections, and detailed classification performance metrics.

    Args:
        detection_times_by_class (dict):
            Mapping from each fault class to a list of detection delays (floats) for that class.
        missed_detections_by_class (dict):
            Mapping from each fault class to the count of missed detections.
        p_metrics: object returned by performance_metrics(), with attributes:
            acc, precision_macro, recall_macro, f1_macro, mcc,
            precision_per_class (dict), recall_per_class (dict)
        logger (logging.Logger, optional):
            Logger instance to use; defaults to module logger if None.
    """
    # Log per-class average detection times
    logger.info("=============== Performance Metrics ==============")
    
    # Log overall performance metrics
    logger.info(f"Accuracy: {p_metrics.acc:.4f}")
    logger.info(f"Precision: {p_metrics.precision_macro:.4f}")
    logger.info(f"Recall: {p_metrics.recall_macro:.4f}")
    logger.info(f"F1 Score: {p_metrics.f1_macro:.4f}")
    logger.info(f"MCC: {p_metrics.mcc:.4f}")

    # Precision and Recall per class
    prec_map = getattr(p_metrics, 'precision_per_class', {})
    rec_map  = getattr(p_metrics, 'recall_per_class', {})
    classes  = set(prec_map.keys()) | set(rec_map.keys())
    for cls_name in classes:
        p_val = prec_map.get(cls_name)
        r_val = rec_map.get(cls_name)
        logger.info(f"- {cls_name}: Precision = {p_val:.4f}, Recall = {r_val:.4f}")
    
    # Per Class Detection Time
    logger.info(f"\nDetection time:")
    for cls_name, delays in detection_times_by_class.items():
        if delays:
            avg_delay = sum(delays) / len(delays)
            logger.info(f"{cls_name}: {avg_delay:.1f} t_stamps")
        else:
            logger.info(f"{cls_name}: missed")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window_size, pred_length, num_layers, hidden_dim, drop_prob, features_h, labels_h, batch_size, filter_normal = get_config_values()
    input_dim = len(features_h)

    # Load model
    model = LSTMPredictor(
        input_dim,
        hidden_dim,
        num_layers,
        pred_length,
        drop_prob
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location = device))
    model.eval()

    # Load fitted covariance and Mahalanobis distances
    cov     = joblib.load(cov_path)
    m_dists = np.load(m_dists_path)
    
    # Setting Anomaaly Detection Threshold
    threshold = np.percentile(m_dists, PERCENTILE)
    logger.info(f"Anomaly‐detection threshold at {PERCENTILE}th percentile: {threshold:.4f}\n")

    # -------------------- Model Testing --------------------

    # Load Scaler
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)    

    # Gather Test Data
    test_npy = [path for path in test_data_path.rglob("*.npy") if "Normal" not in path.parts]

    # Variables to store overall inference results
    all_y_true, all_y_pred = [], []

    # Detection times per fault class (skip normal)
    detection_times_by_class = {label: [] for label in labels_h[1:]}

    # Missed detections per fault class (skip normal)
    missed_detections_by_class = {label: [] for label in labels_h[1:]}

    for filepath in tqdm(test_npy, desc="Evaluating Model Performance", unit="file"):
        data  = np.load(filepath, allow_pickle=True).item()
        feats   = data['features']
        labs    = data['labels']
        f_names = data['feature_names']
        l_names = data['labels_names']

        # Selecting Features
        idxs  = [f_names.index(f) for f in features_h]
        feats = feats[:, idxs]
        
        # Selecting Labels & Transforming from one-hot -> class idx
        l_idxs = [l_names.index(l) for l in labels_h]
        cls   = np.argmax(labs[:, l_idxs], axis=1)

        # Apply Normalization to Features with fitted scaler
        if scaler is None:
            raise RuntimeError(f"Fitted Scaler not loaded.")
        feats = scaler.transform(feats)

        # Variables to store file inference results
        y_true, y_pred = [], []

        # Perform Inference over the Test file
        y_true, y_pred = infer_file(
            feats,
            cls,
            model,
            cov,
            threshold,
            device,
            batch_size,
            window_size
        )

        # Compute detection time and append per class
        det_time, fault_label = compute_detection_time(y_true, y_pred)
        if fault_label is not None:
            class_name = labels_h[fault_label]
            if det_time is not None:
                detection_times_by_class[class_name].append(det_time)
            else:
                missed_detections_by_class[class_name].append(filepath.name)

        # Append file results
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    # Compute Performance Metrics
    y_true_arr = np.array(all_y_true, dtype=np.int8) # Cast from list to np.ndarray
    y_pred_arr = np.array(all_y_pred, dtype=np.int8) # Cast from list to np.ndarray
    metrics = performance_metrics(y_true_arr, y_pred_arr, labels_h, filter_normal)

    # Log performance Metrics
    log_performance(
        detection_times_by_class,
        missed_detections_by_class,
        metrics,
        logger
    )   

if __name__ == "__main__":
    main()
