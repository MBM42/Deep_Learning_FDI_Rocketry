"""
AD_LSTM_Optuna.py

This script provides a framework for finding optimal hyperparameters for AD_LSTM_main.py

Author: Miguel Marques
Date: 05-05-2025
"""

import joblib
import json
import logging
import os
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import optuna.visualization as vis

# Import custom packages
from Preprocessing_Pkg import *
from AD_LSTM_settings import *
from Headers_Pkg import *
from Custom_NN_Pkg import *

# ─── OUTPUT FILE NAME ───
current_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR  = os.path.join(current_dir, "AD_LSTM_Optimizations", f"AD_LSTM_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"AD_LSTM_Opt.txt")

#  ─── DATA PATH ───
DATA_DIR = os.path.join(current_dir, "../Data/New_Data_npy/Normal")  # Path to data directory

# ─── CONFIGURATION ───
METRICS = ["val_loss"]
METRIC_DIRECTIONS = ["minimize"]
N_TRIALS = 50                                  # Total trials            
TIMEOUT = 1                                    # Total time in hours
N_WORKERS = 1

# Headers 
sel_headers = features_selected

def objective(trial):
    """
    Return an Optuna objective that trains the model and returns specified metrics.
    """

    # Settings
    settings_optuna = AD_LSTMSettings(
        window_size=trial.suggest_int("window_size", 1, 50),
        pred_length=trial.suggest_int("pred_length", 1, 20),
        decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    )
    settings_optuna.loader_workers = 8

    # Hyperparameters
    hyperparams_optuna = AD_LSTMHyperparameters(
        hidden_dim=trial.suggest_int("hidden_size", 16, 256),
        num_layers=trial.suggest_int("num_layers", 1, 3),
        drop_prob=trial.suggest_float("drop_prob", 0.0, 0.5),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        clip_grad_norm = trial.suggest_float("clip_grad_norm", 0.5, 5),
    )
    hyperparams_optuna.num_epochs = 1
    hyperparams_optuna.batch_size = 512

    # Prepare data
    npy_data = AD_NPYDataset(DATA_DIR, logging.getLogger(__name__), sel_headers)
    npy_data.gather_data()
    npy_data.cache_npy_files()
    npy_data.normalize(settings_optuna.scaler_type)

    train_ds = AD_SlidingWindow(npy_data.train_cache, settings_optuna.window_size, settings_optuna.pred_length)
    val_ds = AD_SlidingWindow(npy_data.val_cache, settings_optuna.window_size, settings_optuna.pred_length)

    train_loader = DataLoader(
        train_ds,
        batch_size = hyperparams_optuna.batch_size, 
        shuffle = settings_optuna.shuffle_samples,
        num_workers = settings_optuna.loader_workers,
        pin_memory = settings_optuna.pin_memory,
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size = hyperparams_optuna.batch_size, 
        shuffle = False,
        num_workers = settings_optuna.loader_workers,
        pin_memory = settings_optuna.pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMPredictor(
        input_size=npy_data.train_cache[0].shape[1],
        hidden_size=hyperparams_optuna.hidden_dim,
        num_layers=hyperparams_optuna.num_layers,
        pred_length=settings_optuna.pred_length,
        dropout=hyperparams_optuna.drop_prob
    ).to(device)

    # Train and retrieve losses
    _, train_losses, val_losses = train_predictor(
        model, device, train_loader, val_loader, settings_optuna, hyperparams_optuna
    )

    # Use final validation loss as objective
    return val_losses[-1]


# Anomaly Detection LSTM‐based predictor definition
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_length, dropout):
        """
        Sequence‐to‐sequence LSTM predictor for anomaly detection.

        Attributes:
            input_size (int): number of features (f)
            hidden_size (int): LSTM hidden dimension
            num_layers (int): number of stacked LSTM layers
            pred_length (int): how many steps ahead to predict (l)
        """
        super().__init__()
        self.pred_length = pred_length
        self.input_size = input_size

        # Stacked LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # Map last hidden state to l×m outputs
        self.fc = nn.Linear(hidden_size, pred_length * input_size)

    def forward(self, x):
        """
        x: [batch, p, f]
        returns: tensor [B, l, f]
        """

        out, _ = self.lstm(x)            # [batch, p, hidden]
        last = out[:, -1, :]             # [batch, hidden]
        y = self.fc(last)                # [batch, l*f]
        return y.view(-1, self.pred_length, self.input_size)

def train_predictor(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    settings: AD_LSTMSettings,
    hyper: AD_LSTMHyperparameters,
) -> nn.Module:
    """
    Train an LSTM-based predictor for a fixed number of epochs using MSE loss.
    
    Returns the model with the best validation metrics (lower loss).

    Args:
        model        : LSTMPredictor with the chosen setup.
        device       : either 'cpu' or 'gpu'
        train_loader
        val_loader
        settings     : AD_LSTMSettings.
        hyper        : AD_LSTMHyperparameters.

    Returns:
        model: best model.
        train_losses
        val_losses
    """

    # Set Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyper.learning_rate,
        weight_decay=settings.decay,
    )

    # Set Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        settings.scheduler_mode,
        settings.scheduler_factor,
        settings.scheduler_patience,
    )

    # Early Stopping
    stopper   = EarlyStopping(settings.stop_patience, settings.stop_min_delta, settings.stop_mode)
    
    # Loss function - Mean Squared Error
    criterion = nn.MSELoss()

    # Log Model
    input_dim = (settings.window_size, len(sel_headers))

    # Initialize variables for best performance
    best_val   = float('inf')       # Infinite positive   
    best_state = model.state_dict()

    # Loss objects
    train_losses = []
    val_losses   = []

    # Training Loop
    for epoch in range(1, hyper.num_epochs + 1):
        
        # Training Phase
        model.train()
        train_loss = 0.0
        for inp, tgt in tqdm(train_loader, desc=f"Epoch {epoch}/{hyper.num_epochs}", unit="batch"): 
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = model(inp)
            loss = criterion(out, tgt)
            loss.backward()
            if hyper.clip_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), hyper.clip_grad_norm)
            optimizer.step()
            train_loss += loss.item() * inp.size(0)
        avg_train = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, tgt in tqdm(val_loader, desc="Validation Step", leave=False, unit="batch"):
                inp, tgt = inp.to(device), tgt.to(device)
                out = model(inp)
                loss = criterion(out, tgt)
                val_loss += loss.item() * inp.size(0)
        avg_val = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val)

        # Log results
        print(f"Epoch [{epoch}/{hyper.num_epochs}] Train Loss = {avg_train:.6f}, Val Loss = {avg_val:.6f} "
                    f"[LR: {optimizer.param_groups[0]['lr']}]")

        # Scheduler & EarlyStop
        scheduler.step(avg_val)
        stopper.step(avg_val)

        # Save best
        if avg_val + settings.stop_min_delta < best_val:
            best_val   = avg_val
            best_state = model.state_dict()

        if stopper.should_stop:
            print("⏹ Early stopping triggered")
            break

    # Restore best weights
    model.load_state_dict(best_state)
    return model, train_losses, val_losses


def write_header():
    """Initialize the results file."""
    with open(OUTPUT_FILE, 'w') as f:
        f.write("Optuna Hyperparameter Optimization Results\n")
        f.write(f"Metrics: {METRICS}\n")
        f.write(f"Directions: {METRIC_DIRECTIONS}\n")
        f.write(f"Trials: {N_TRIALS}, Timeout: {TIMEOUT}h\n\n")

def write_summary(study):
    """Append the best trial summary to the results file."""
    with open(OUTPUT_FILE, 'a') as f:
        f.write("\n## Study Summary ##\n")
        best = study.best_trial
        f.write(f"Best {METRICS[0]} ({METRIC_DIRECTIONS[0]}): {study.best_value:.7f}\n")
        f.write("Best parameters:\n")
        for name, val in best.params.items():
            if isinstance(val, float):
                f.write(f"  {name}: {val:.7f}\n")
            else:
                f.write(f"  {name}: {val}\n")

def save_visualizations(study):
    """
    Save Optuna visualizations to the OUTPUT_DIR.
    """
    # Optimization history
    vis.plot_optimization_history(study).write_image(os.path.join(OUTPUT_DIR, "optuna_optimization_history.png"))
    # Parameter importances
    vis.plot_param_importances(study).write_image(os.path.join(OUTPUT_DIR, "optuna_param_importances.png"))
    # Slice plot
    vis.plot_slice(study).write_image(os.path.join(OUTPUT_DIR, "optuna_slice.png"))
    # Contour example for two key parameters
    try:
        vis.plot_contour(study, params=["window_size", "hidden_size"]).write_image(
            os.path.join(OUTPUT_DIR, "optuna_contour_window_hidden.png")
        )
    except Exception:
        pass

def log_trial_result(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    trial_number = trial.number
    with open(OUTPUT_FILE, "a") as f:
        f.write(f"\n==== Trial {trial_number} ===\n")
        f.write("Sampled parameters:\n")
        for name, val in trial.params.items():
            if isinstance(val, float):
                f.write(f"  {name}: {val:.6f}\n")
            else:
                f.write(f"  {name}: {val}\n")
        f.write("Results:\n")
        for i, val in enumerate(trial.values):
            f.write(f"  {METRICS[i]}: {val:.6f}\n")


def main():

    # Convert TIEMOUT to seconds
    timeout = TIMEOUT*60*60

    write_header()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, N_TRIALS, timeout, N_WORKERS, callbacks=[log_trial_result], show_progress_bar = True)

    # Save best parameters
    best_params = study.best_params
    with open(os.path.join(OUTPUT_DIR, "best_params.txt"), "w") as f:
        f.write(str(best_params))

    write_summary(study)
    save_visualizations(study)

    # Save 'study' object
    study_path = os.path.join(OUTPUT_DIR, "optuna_study.pkl")
    joblib.dump(study, study_path)

    # Save trials
    with open(os.path.join(OUTPUT_DIR,"all_trials.json"), "w") as f:
        json.dump([t.params for t in study.trials], f)

if __name__ == "__main__":
    main()
