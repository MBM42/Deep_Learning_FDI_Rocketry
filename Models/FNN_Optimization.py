"""
FNN_Optimization.py

Module to optimize the FNN Hyperparameters with Optuna.

Author: Miguel Marques
Date: 22-04-2025
"""

import os
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from sklearn.metrics import *

# FNN Architecture
from FNN_main import FNN
# Import custom packages
from Preprocessing_Pkg import *
from Aux_Pkg import *
from Custom_NN_Pkg import *
from FNN_settings import *

# Silence verbose logging during hyperparameter search - Only WARNING and above would be logged
logging.getLogger().setLevel(logging.WARNING)

def train_and_evaluate(
    settings: FNNSettings,
    hyperparams: FNNHyperparameters,
    debug_mode: bool = False,
) -> dict:
    """
    Train the FNN for a fixed number of epochs and evaluate on the validation set.
    
    If debug_mode is True, uses debug dataset and fewer epochs.
    
    Returns a dict with the following metrics:
      'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'avg_detection_time'.
    """
    # 0) Debug overrides
    from Headers_Pkg import (
        features_selected, total_labels,
        debug_features, debug_labels,
        weights_manual, debug_weights,
    )
    if debug_mode:
        sel_features = debug_features
        sel_labels   = debug_labels
        weights      = debug_weights
        settings.data_folder = "../Data/Debug_Data_npy"
        hyperparams.num_epochs = 4
    else:
        sel_features = features_selected
        sel_labels   = total_labels
        weights      = weights_manual


    # Directory handling
    current_dir = os.path.dirname(os.path.abspath(__file__))    # Get the current directory path
    data_dir = os.path.join(current_dir, settings.data_folder)  # Path to data directory

    # 1) Reproducibility
    if settings.fixed_seed:
        torch.manual_seed(settings.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 2) Prepare data
    npy_data = NPYDataset(
        data_dir,
        logger = logging.getLogger(__name__),
        settings = settings,
        selected_features = sel_features,
        selected_labels = sel_labels,
    )

    # NPY Dataset
    if settings.fixed_seed:
        npy_data.set_seed()
    npy_data.count_npy_files()
    npy_data.gather_data()
    npy_data.cache_npy_files()
    npy_data.count_class_instances(weights_manual = weights)
    npy_data.compute_normalization_stats()

    # SlidingWindow datasets
    train_ds = SlidingWindow(npy_data.train_cache, settings, npy_data)
    val_ds   = SlidingWindow(npy_data.val_cache,   settings, npy_data)
    
    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size = hyperparams.batch_size,
        shuffle = settings.shuffle_samples,
        num_workers = settings.loader_workers,
        pin_memory = settings.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size = hyperparams.batch_size,
        shuffle = False,
        num_workers = settings.loader_workers,
        pin_memory = settings.pin_memory,
    )

    # 3) Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = settings.window_size * len(npy_data.selected_features)
    model = FNN(
        input_dim,
        hyperparams.hidden_dim,
        len(npy_data.selected_labels),
        hyperparams.drop_prob,
    ).to(device)

    # Loss function
    if settings.use_focal_loss:
        if settings.apply_class_weights:
            cw = torch.FloatTensor(npy_data.class_weights).to(device)
            loss_fn = FocalLoss(settings.gamma_fl, cw)
        else:
            loss_fn = FocalLoss(settings.gamma_fl)
    else:
        if settings.apply_class_weights:
            cw = torch.FloatTensor(npy_data.class_weights).to(device)
            loss_fn = nn.CrossEntropyLoss(weight=cw)
        else:
            loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams.learning_rate,
        weight_decay=settings.decay,
    )

    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=settings.scheduler_mode,
        factor=settings.scheduler_factor,
        patience=settings.scheduler_patience,
    )

    # Ealy Stopping
    early_stopper = EarlyStopping(settings.stop_patience, settings.stop_min_delta, settings.stop_mode)

    # 4) Training and validation loop
    metrics = {}
    for epoch in range(hyperparams.num_epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                out = model(Xv)
                val_loss += loss_fn(out, yv).item()
                preds = torch.argmax(out, dim=1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(yv.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)

        # Saving loss
        metrics['val_loss'] = avg_val_loss

        # Computing Accuracy
        metrics['val_accuracy'] =  accuracy_score(y_true, y_pred)

        # Filter out normal class if desired
        if settings.filter_normal:
            mask = y_true != 0
            y_true_m, y_pred_m = y_true[mask], y_pred[mask]
        else:
            y_true_m, y_pred_m = y_true, y_pred

        # Computing remaining metrics
        metrics['val_precision'] = precision_score(y_true_m, y_pred_m, average='macro', zero_division=0.0)
        metrics['val_recall'] = recall_score(y_true_m,    y_pred_m, average='macro', zero_division=0.0)
        metrics['val_f1'] = f1_score(y_true_m,        y_pred_m, average='macro', zero_division=0.0)

        # Early stopping on validation accuracy
        early_stopper.step(metrics['val_accuracy'])
        if early_stopper.should_stop:
            break

        # Scheduler
        scheduler.step(avg_val_loss)

    # 5) Detection time evaluation on final model
    detection_eval = DetectionTime(model, settings, npy_data)
    det_results = detection_eval.compute_detection_times()
    metrics['avg_detection_time'] = det_results.get('Global Average')

    return metrics
