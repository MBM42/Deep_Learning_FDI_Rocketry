"""
LSTM_Optuna.py

This script provides a framework for finding optimal hyperparameters for LSTM_main.py

Author: Miguel Marques
Date: 27-04-2025
"""

import joblib
import os
import multiprocessing
import optuna
from datetime import datetime
import optuna.visualization as vis
from dataclasses import asdict
from Preprocessing_Pkg import *
from LSTM_Optimization import *

# ─── OUTPUT FILE NAME ───
current_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR  = os.path.join(current_dir, "LSTM_Optimizations", f"LSTM_{timestamp}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"LSTM_Opt.txt")


# ─── CONFIGURATION ───
METRICS = ["val_f1", "val_accuracy"]           # Available: val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'avg_detection_time'.
METRIC_DIRECTIONS = ["maximize", "maximize"]   # Corresponding to the selected metrics
N_TRIALS = 50                                  # Total trials            
TIMEOUT = 1                                    # Total time in hours
DEBUG_MODE = True
N_WORKERS = 2


def objective_factory(metrics):
    """
    Return an Optuna objective that trains the model and returns specified metrics.
    """
    def objective(trial):
        
        # Weights
        apply_weights = trial.suggest_categorical("apply_weights", [True, False])
        # Only sample weight type if they're being applied
        if apply_weights:
            use_manual = trial.suggest_categorical("manual_weight", [True, False])
        else:
            use_manual = False
        
        # Settings
        settings = LSTMSettings(
            window_size = trial.suggest_int("window_size", 1, 50),
            gamma_fl = trial.suggest_float("gamma_fl", 0.0, 3.0),      # A gamma of 0 is equivalent to the Cross Entropy Loss
            decay = trial.suggest_float("L2", 1e-7, 1e-3, log = True),
            apply_class_weights = apply_weights,
            use_manual_weights = use_manual,
        )
        settings.loader_workers = 8

        # Hyperparameters
        hyperparams = LSTMHyperparameters(
            hidden_dim = trial.suggest_int("hidden_dim", 32, 256),
            num_layers =  trial.suggest_int("num_layers", 1, 3),
            drop_prob = trial.suggest_float("drop_prob", 0.0, 0.5),
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log = True),
            bidirectional = trial.suggest_categorical("bidirectional", [True, False]),
            clip_grad_norm = trial.suggest_float("clip_grad_norm", 0.5, 5),
            num_epochs = 20,
        )

        # Train and evaluate (returns final metrics)
        results = train_and_evaluate(settings, hyperparams, DEBUG_MODE)

        # Log completed trial
        with open(OUTPUT_FILE, 'a') as f:
            f.write(f"==== Trial {trial.number} ===\n")
            f.write("Sampled parameters:\n")
            for name, val in trial.params.items():
                if isinstance(val, float):
                    f.write(f"  {name}: {val:.6f}\n")
                else:
                    f.write(f"  {name}: {val}\n")
            f.write("Results:\n")
            for metric, value in results.items():
                if isinstance(value, float):
                    f.write(f"  {metric}: {value:.6f}\n")
                else:
                    f.write(f"  {metric}: {value}\n")
            f.write("========\n\n")

        # Return target metric(s)
        vals = tuple(results[m] for m in metrics)
        return vals[0] if len(vals) == 1 else vals

    return objective


def write_header():
    """Initialize the results file."""
    with open(OUTPUT_FILE, 'w') as f:
        f.write("Optuna Hyperparameter Optimization Results\n")
        f.write(f"Metrics: {METRICS}\n")
        f.write(f"Directions: {METRIC_DIRECTIONS}\n")
        f.write(f"Trials: {N_TRIALS}, Timeout: {TIMEOUT}h\n")
        f.write(f"Debug mode: {DEBUG_MODE}\n\n")


def write_summary(study):
    """Append the best trial summary to the results file."""
    with open(OUTPUT_FILE, 'a') as f:
        f.write("\n## Study Summary ##\n")
        if len(METRICS) == 1:
            best = study.best_trial
            f.write(f"Best {METRICS[0]} ({METRIC_DIRECTIONS[0]}): {study.best_value:.7f}\n")
            f.write("Best parameters:\n")
            for name, val in best.params.items():
                if isinstance(val, float):
                    f.write(f"  {name}: {val:.7f}\n")
                else:
                    f.write(f"  {name}: {val}\n")
        else:
            f.write("Pareto‐optimal trials:\n")
            for t in study.best_trials:
                # summary line
                metrics_summary = ", ".join(
                    f"{m}={v:.7f} ({d})"
                    for m, v, d in zip(METRICS, t.values, METRIC_DIRECTIONS)
                )
                f.write(f"Trial#{t.number}: {metrics_summary}\n")
                f.write("Params:\n")
                for name, val in t.params.items():
                    if isinstance(val, float):
                        f.write(f"  {name}: {val:.7f}\n")
                    else:
                        f.write(f"  {name}: {val}\n")
                f.write("----\n")


def save_visualizations(study):
    """
    Plots optimization summary as a function of f1 and accuracy.
    """
    def target_f1(trial):
        return trial.values[0]

    def target_accuracy(trial):
        return trial.values[1]

    # Save plots into OUTPUT_DIR
    vis.plot_optimization_history(study, target=target_f1).write_image(os.path.join(OUTPUT_DIR, "optuna_optimization_history_f1.png"))
    vis.plot_optimization_history(study, target=target_accuracy).write_image(os.path.join(OUTPUT_DIR, "optuna_optimization_history_accuracy.png"))
    
    vis.plot_param_importances(study, target=target_f1).write_image(os.path.join(OUTPUT_DIR, "optuna_param_importances_f1.png"))
    vis.plot_param_importances(study, target=target_accuracy).write_image(os.path.join(OUTPUT_DIR, "optuna_param_importances_accuracy.png"))
    
    vis.plot_slice(study, target=target_f1).write_image(os.path.join(OUTPUT_DIR, "optuna_slice_f1.png"))
    vis.plot_slice(study, target=target_accuracy).write_image(os.path.join(OUTPUT_DIR, "optuna_slice_accuracy.png"))
    
    # Contour Plots
    plots = [
        (["L2", "gamma_fl"], "L2_vs_gamma"),
        (["L2", "window_size"], "L2_vs_window"),
        (["gamma_fl", "hidden_dim"], "gamma_vs_hidden"),
        (["window_size", "hidden_dim"], "window_vs_hidden"),
        (["hidden_dim", "L2"], "hidden_vs_L2"),
    ]

    for params, suffix in plots:
        try:
            vis.plot_contour(study, params=params, target=target_f1)\
               .write_image(os.path.join(OUTPUT_DIR, f"optuna_contour_f1_{suffix}.png"))
            vis.plot_contour(study, params=params, target=target_accuracy)\
               .write_image(os.path.join(OUTPUT_DIR, f"optuna_contour_accuracy_{suffix}.png"))
        except Exception as e:
            print(f"Skipping {params}: {e}")

def main():

    # Convert TIEMOUT to seconds
    timeout = TIMEOUT*60*60

    write_header()
    assert len(METRICS) == len(METRIC_DIRECTIONS), \
        "Metrics and directions length mismatch"

    study = optuna.create_study(
        directions=METRIC_DIRECTIONS,
        sampler=optuna.samplers.TPESampler(),
    )
    objective = objective_factory(METRICS)
    study.optimize(objective, N_TRIALS, timeout, N_WORKERS, show_progress_bar = True)

    write_summary(study)
    save_visualizations(study)

    # Save 'study' object
    study_path = os.path.join(OUTPUT_DIR, "optuna_study.pkl")
    joblib.dump(study, study_path)


if __name__ == "__main__":
    main()
