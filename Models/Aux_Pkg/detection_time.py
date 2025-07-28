"""
detection_time.py

This class allows to evaluate the detection time (in timestamps) since a fault occurs and the model declares it.

Author: Miguel Marques
Date: 15-04-2025
"""

import torch
import numpy as np
from tqdm import tqdm

class DetectionTime:
    """
    Params:
        model: Trained model
        settings: settings from which to retrieve extra attributes
        npy_dataset: from which to retrieve the validation set
  
    """
    def __init__(
        self, 
        model, 
        settings, 
        npy_dataset, 
    ) -> None:
        self.model           = model
        self.val_cache       = npy_dataset.val_cache
        self.window_size     = settings.window_size
        self.scaler_type     = settings.scaler_type.lower()
        self.flatten_window  = settings.flatten_window
        self.global_mean     = npy_dataset.global_mean
        self.global_std      = npy_dataset.global_std
        self.global_min      = npy_dataset.global_min
        self.global_max      = npy_dataset.global_max
        self.total_labels    = npy_dataset.selected_labels

    def compute_detection_times(self) -> dict:
        """
        For each simulation in self.val_cache:
          - finds true fault‐onset true_onset_t
          - builds all sliding windows
          - normalizeS (zscore or minmax) each window
          - flatten data only if self.flatten_window = True
          - calculates the identification delay = (window_end) - true_onset_t
          - record result as None if never detected
        """

        # Getting the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare storage
        fault_delays = {lbl: [] for lbl in self.total_labels if lbl != "Normal"}
        total = 0
        missed = 0

        # Loop with progress bar
        for data in tqdm(self.val_cache, desc="Evaluating Detection Time", unit="simulation"):
            features   = data["features"]            # [T, F]
            labels_arr = data["labels"]              # [T, C]
            class_idx  = np.argmax(labels_arr, axis=1)

            # Skip all‑normal or too short
            if np.all(class_idx == 0) or features.shape[0] < self.window_size:
                continue
            
            # Valid Simulation
            total += 1

             # 1) True‐fault onset via diff
            time_axis = np.arange(len(class_idx))
            onsets = np.where(np.diff(class_idx) > 0)[0] + 1
            if len(onsets) > 0:
                onset = onsets[0]
            else:
                # if fault starts at t=0
                if class_idx[0] != 0:
                    onset = 0
                else:
                    # fallback to first non-zero
                    onset = np.where(class_idx != 0)[0][0]

            true_fault = class_idx[onset]
            fault_name  = self.total_labels[true_fault]

            # 2) Build all windows
            num_windows = len(class_idx) - self.window_size + 1
            raw_windows = np.stack([
                features[i : i + self.window_size]
                for i in range(num_windows)
            ], axis=0)  # [W, window_size, F]

            # 3) Normalize
            if self.scaler_type == "zscore":
                norm_windows = (raw_windows - self.global_mean) / self.global_std
            elif self.scaler_type == "minmax":
                denom = self.global_max - self.global_min
                denom[denom == 0] = 1.0
                norm_windows = (raw_windows - self.global_min) / denom
            else:
                raise ValueError(f"Unsupported scaler type: {self.scaler_type}")

            # 4) Flatten if requested
            if self.flatten_window:
                batch_data = norm_windows.reshape(num_windows, -1)  # [W, window_size*F]
            else:
                batch_data = norm_windows                           # For LSTM [W, window_size, F]

            # 5) Batched inference
            batch_tensor = torch.tensor(batch_data, dtype=torch.float32, device=device)
            with torch.no_grad():
                outputs = self.model(batch_tensor)                     # [W, C]
                preds   = torch.argmax(outputs, dim=1).cpu().numpy()   # [W]

            # 6) Find first correct prediction via time alignment
            pred_time = np.arange(self.window_size - 1, len(class_idx))
            match_idx = None
            for j, t_pred in enumerate(pred_time):
                # locate which class_idx is "in effect" at t_pred
                k = np.searchsorted(time_axis, t_pred, side='right') - 1
                if k < 0:
                    continue
                if preds[j] == class_idx[k] != 0:
                    match_idx = j
                    break

            if match_idx is not None:
                detected_ts = pred_time[match_idx]
                delay = detected_ts - onset
                fault_delays[fault_name].append(delay)
            else:
                missed += 1

        # 7) Aggregate into result dict
        results = {}
        all_delays = []
        for fault, delays in fault_delays.items():
            if delays:
                avg = sum(delays) / len(delays)
                results[fault] = avg
                all_delays.extend(delays)
            else:
                results[fault] = None

        results["Global Average"]= (sum(all_delays) / len(all_delays)) if all_delays else None
        results["Total Sims"]    = total
        results["Missed Sims"]   = missed

        return results
