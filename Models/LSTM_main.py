"""
LSTM_main.py

LSTM fault identification implementation.

Author: Miguel Marques
Date: 03-04-2025
"""

import os
import time
import shutil
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Import custom packages
from Preprocessing_Pkg import *
from Headers_Pkg import *
from Aux_Pkg import *
from LSTM_settings import *
from Custom_NN_Pkg import *

# =================== Debug Mode ===================
debug_mode = False         # Smaller dataset and reduced number of epochs
log_mode_f = logging.DEBUG # From which level the content of the config file is written (DEBUG or INFO)

# Headers
features_h = features_selected
labels_h = total_labels
weights_h =  weights_manual

# Instantiate settings and hyperparameters for LSTM
settings = LSTMSettings()
hyperparams = LSTMHyperparameters()

# Redefining variables for debug mode
if debug_mode:
    # Headers
    features_h = debug_features
    labels_h   = debug_labels
    weights_h  = debug_weights
    settings.data_folder = "../Data/Debug_Data_npy"
    settings.model_suffix = settings.model_suffix + "_Debug"
    hyperparams.num_epochs = 2

# Creating loggers and respective handlers
logger = logging.getLogger("logger")            # Logger object for console and model file
logger.setLevel(logging.DEBUG)                  # DEBUG and above
model_logger = logging.getLogger("model")       # Logger object for model file only
model_logger.setLevel(logging.DEBUG)            # DEBUG and above
console_logger = logging.getLogger("console")   # Logger object for console only
console_logger.setLevel(logging.DEBUG)          # DEBUG and above
console_handler = logging.StreamHandler()       # Console handler
console_handler.setLevel(logging.DEBUG)         # DEBUG and above
logger.addHandler(console_handler)
console_logger.addHandler(console_handler)


# LSTM Model definition
class LSTMModel(nn.Module):
    """
    LSTM-based model for fault identification.
    
    Attributes:
        - input_dim(int) : Dimensionality of the input features at each time step.
        - hidden_dim(int) : Number of hidden neurons per LSTM layer (per direction).
        - num_layers(int) : Number of stacked LSTM layers. Typically between 1 and 3. If deeper (>3) can help on long and complex sequences but requires
                            more careful tuning (learning rate, regularization).
        - output_dim(int) : Number of fault classes to predict
        - drop_prob(float) : Dropout probability applied between LSTM layers and in the projection head to reduce overfitting.
        - bidirectional(bool) : If True, runs two LSTMs (forward and backward) and concatenates their hidden states at each step.
                                This gives each output access to both past and future context, often boosting accuracy.
        - clip_grad_norm(float or None) : Maximum L2-norm for gradient clipping. During training, gradients whose norm exceeds this value will be scaled 
                                          down to prevent exploding gradients. A value between 0.5 and 5.0 is typical. Default == None -> Disabled
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 output_dim: int,
                 drop_prob: float,
                 bidirectional: bool = False,
                 clip_grad_norm: float = None):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        self.clip_grad_norm = clip_grad_norm

        # Core LSTM
        self.lstm = nn.LSTM(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,          # Input dimensions: [batch, window_size, features]
            dropout       = drop_prob,
            bidirectional = bidirectional
        )

        # Regularization layers
        self.dropout = nn.Dropout(drop_prob)
        self.layernorm = nn.LayerNorm(hidden_dim * self.num_directions)

        # Non-linear projection head
        self.fc1 = nn.Linear(hidden_dim * self.num_directions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_tensor):
        # input_tensor: [batch, window_size, features]
        lstm_out, _ = self.lstm(input_tensor)   # [batch, window_size, features * num_directions]
        h_last = lstm_out[:, -1, :]             # grab last time-step

        # Additional regularization
        h = self.dropout(h_last)
        h = self.layernorm(h)

        # non-linear projection
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        logits = self.fc2(h)
        return logits

    def train_step(self, batch, optimizer, loss_fn, device):
        features, labels = batch
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = self(features)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Gradient clipping
        if self.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)

        optimizer.step()
        return loss.item(), outputs, labels

    def val_step(self, batch, loss_fn, device):
        features, labels = batch
        features, labels = features.to(device), labels.to(device)
        outputs = self(features)
        loss = loss_fn(outputs, labels)
        return loss.item(), outputs, labels


def main():

    # Ensuring Reproducibility
    if settings.fixed_seed:
        torch.manual_seed(settings.seed)            # Ensures reproducibility for Dataloader and Dropout
        torch.backends.cudnn.deterministic = True   # Forces CuDNN to use deterministic algorithms
        torch.backends.cudnn.benchmark = False      # Prevents CuDNN from dynamically selecting different algorithms for different batch sizes
    
    # Directory handling
    current_dir = os.path.dirname(os.path.abspath(__file__))    # Get the current directory path
    data_dir = os.path.join(current_dir, settings.data_folder)  # Path to data directory

    # Define the model directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = f"LSTM_{settings.model_suffix}_{timestamp}"
    model_dir_path = Path(current_dir) / "LSTM_Trained_Models" / model_dir
    model_dir_path.mkdir(parents=True, exist_ok=True)           # Create folder if it doesn't exist

    # Define "plot_results.py" path
    plot_results_path = Path(current_dir) / "PostprocessingScripts" / "plot_results.py"
    
    # Define "plot_fault.py" path
    plot_fault_path = Path(current_dir) / "PostprocessingScripts"  / "plot_fault.py"

    # Define "FNN_inference.py" path
    inference_path = Path(current_dir) / "PostprocessingScripts" / "LSTM_inference.py"

    # Config file handler
    conf_handler = logging.FileHandler(os.path.join(model_dir_path, "Model_Config.txt"))
    conf_handler.setLevel(log_mode_f)
    logger.addHandler(conf_handler)
    model_logger.addHandler(conf_handler)

    # Log settings
    log_settings(logger, timestamp, debug_mode, settings)
    
    # Log Dataset Memory Info
    log_dataset_mem_size(logger, data_dir)

    # NPY Files Manipulation
    npy_dataset = NPYDataset(data_dir, logger, settings, features_h, labels_h)
    if settings.fixed_seed:
        npy_dataset.set_seed()                          # Set fixed seed for reproducibility
    npy_dataset.count_npy_files()                       # Count .npy files per type ['Normal', 'Valve', ...])
    npy_dataset.gather_data()                           # Gather paths to npy_dataset.npy_set Dict: {'Train': [...], 'Validation': [...], (...)}
    npy_dataset.cache_npy_files()
    npy_dataset.count_class_instances(weights_h)        # Count the number of timestamps and files for each class and compute their weight (normalized or not)
    npy_dataset.check_labels("val_cache")               # Check Validation Set for the presence of all labels
    npy_dataset.count_class_files("val_cache")          # Count the number of files for each class
    if settings.create_test_set:
        npy_dataset.check_labels("test_cache")          # Check Test Set for the presence of all labels
        npy_dataset.count_class_files("test_cache")     # Count the number of files for each class
    npy_dataset.compute_normalization_stats()           # Global statistics to later apply normalization on the fly
    if settings.save_datasets:                          # Saves the validation and test (if created) dataset
        npy_dataset.save_data(model_dir_path / "Data")

    # Build Datasets
    train_dataset = SlidingWindow(npy_dataset.train_cache, settings, npy_dataset)
    val_dataset   = SlidingWindow(npy_dataset.val_cache,   settings, npy_dataset)
    test_dataset  = SlidingWindow(npy_dataset.test_cache,  settings, npy_dataset) if settings.create_test_set else None

    if settings.save_debug_sample:
        train_dataset.save_debug_samples(settings.debug_type, settings.debug_windows, model_dir_path)

    # DataLoaders
    train_loader = DataLoader(train_dataset, hyperparams.batch_size, shuffle = settings.shuffle_samples, num_workers=settings.loader_workers,
                              pin_memory = settings.pin_memory)
    val_loader = DataLoader(val_dataset, hyperparams.batch_size, shuffle = False, num_workers = settings.loader_workers, 
                            pin_memory = settings.pin_memory)
    if settings.create_test_set:
        test_loader = DataLoader(test_dataset, hyperparams.batch_size, shuffle = False, num_workers = settings.loader_workers, 
                                 pin_memory = settings.pin_memory)

    # Model
    device = get_device(logger) # Check if CUDA is available and define the device for training
    input_dim = len(features_h)
    model = LSTMModel(input_dim, hyperparams.hidden_dim, hyperparams.num_layers,
                      len(labels_h), hyperparams.drop_prob, hyperparams.bidirectional, hyperparams.clip_grad_norm).to(device) # Initialize model
    logger.info(f"Confirm model parameters are on GPU:: {next(model.parameters()).device}")

    # Set Loss Function
    if settings.use_focal_loss:
        if settings.apply_class_weights:                                                          # Focal Loss
            class_weights_t = torch.FloatTensor(npy_dataset.class_weights).to(device)             # Class weights as a tensor
            loss_function = FocalLoss(settings.gamma_fl, class_weights_t)
        else:
            loss_function = FocalLoss(settings.gamma_fl)
    else:
        if settings.apply_class_weights:                                                          # Cross Entropy Loss for multi-class classification (applies Softmax)
            class_weights_t = torch.FloatTensor(npy_dataset.class_weights).to(device)             # Class weights as a tensor
            loss_function = nn.CrossEntropyLoss(weight=class_weights_t)                           
        else:
            loss_function = nn.CrossEntropyLoss()

    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), hyperparams.learning_rate, 
                                 weight_decay = settings.decay) 
    
    # Set Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, settings.scheduler_mode, settings.scheduler_factor, 
                                  settings.scheduler_patience)

    # Logs:
    log_hyperparameters_LSTM(logger, hyperparams, loss_function, optimizer)                                             # Hyperparameters
    log_model(model_logger, model, loss_function, optimizer, hyperparams.batch_size, (settings.window_size, input_dim)) # Log architecture setup
    log_dataset(logger, npy_dataset, train_dataset, val_dataset, test_dataset)                                          # Log dataset stats
    
    # Initializing PerformanceMetrics instances
    train_metrics = PerformanceMetrics(npy_dataset.selected_labels, "Train")
    val_metrics   = PerformanceMetrics(npy_dataset.selected_labels, "Validation")

    # Initializing MaxPerformance instance
    max_val_p = MaxPerformance(val_metrics)

    # Plots
    plots = PlotManager(model_dir_path, train_metrics, val_metrics)

    # Early Stopping
    early_stopper = EarlyStopping(settings.stop_patience, settings.stop_min_delta, settings.stop_mode)


    # Training loop
    logger.info("\n\n===================== Training ======================")
    for epoch in range(hyperparams.num_epochs):
        epoch_start_t = time.time() 
        model.train() 
        train_loss = 0.0
        train_labels, train_preds = [], []

        # Training Phase
        for train_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{hyperparams.num_epochs}", unit="batch"):
            loss, outputs, labels = model.train_step(train_batch, optimizer, loss_function, device)
            train_loss += loss                           # Accumulate train loss
            fault_indices = torch.argmax(outputs, dim=1) # Converts logits to the predicted class
            train_labels.append(labels.cpu().numpy())                           
            train_preds.append(fault_indices.cpu().numpy())
        
        # Convert lists to NumPy arrays
        train_labels = np.concatenate(train_labels)
        train_preds = np.concatenate(train_preds)

        # Train Performance Metrics
        train_metrics.epoch_update(train_labels, train_preds)
        train_metrics.accuracy()
        train_metrics.filter_normal(settings.filter_normal) # Remove normal class before main metrics computation
        train_metrics.compute_metrics()
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_labels, val_preds = [], []
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Validation Step", unit="batch"):
                loss, outputs, labels = model.val_step(val_batch, loss_function, device)
                val_loss += loss# Accumulate validation loss
                fault_indices = torch.argmax(outputs, dim=1) # Converts logits to the predicted class
                val_labels.append(labels.cpu().numpy())
                val_preds.append(fault_indices.cpu().numpy())

        # Convert lists to NumPy arrays
        val_labels = np.concatenate(val_labels)
        val_preds = np.concatenate(val_preds)

        # Validation Metrics
        val_metrics.epoch_update(val_labels, val_preds)
        val_metrics.accuracy()
        val_metrics.filter_normal(settings.filter_normal) # Remove normal class before main metrics computation
        val_metrics.compute_metrics()
        if max_val_p.update_on_all():
            # Save Best Model
            best_model_path = model_dir_path / "best_model.pth"
            torch.save(model.state_dict(), best_model_path) 

        # Timer
        epoch_end_t = time.time()
        epoch_duration = (epoch_end_t - epoch_start_t) / 60   # In minutes

        # Log results
        logger.info(f"Epoch [{epoch + 1}/{hyperparams.num_epochs}] Train Loss = {train_loss / len(train_loader):.4f}, Val Loss = {val_loss / len(val_loader):.4f} | "
                    f"Train Acc. = {train_metrics.acc:.4f}, Val Acc. = {val_metrics.acc:.4f} | Val F1 = {val_metrics.f1_macro:.4f} "
                    f"[LR: {optimizer.param_groups[0]['lr']}] [{epoch_duration:.2f} min]")
        max_val_p.print_best(console_logger)                  # Best Metrics
        log_main_p_metrics(console_logger, val_metrics)       # Validation Metrics
        if settings.log_train_extra_metrics:
            log_main_p_metrics(console_logger, train_metrics) # Train Metrics
        
        # Append Epoch results to Plots' attributes
        plots.update(train_loss/len(train_loader), val_loss/len(val_loader))

        # Early Stopping
        early_stopper.step(val_metrics.acc)
        if early_stopper.should_stop:
            logger.info("Early Stopping")
            break
        
        # Adjust learning rate based on validation loss
        scheduler.step(val_loss/len(val_loader))

    # Log final metrics
    model_logger.info("=====================================================")
    model_logger.info("\n\n================ Performance Metrics ================")
    model_logger.info(f"Loss: Train = {train_loss / len(train_loader):.4f}, Val = {val_loss / len(val_loader):.4f}")
    model_logger.info(f"Accuracy: Train = {train_metrics.acc:.4f}, Val = {val_metrics.acc:.4f}\n")
    max_val_p.print_best(model_logger)            # Best Validation Metrics Registered 
    log_main_p_metrics(model_logger, val_metrics) # Validation Metrics
    model_logger.info("======================================================")

    # Detection Time Evaluation
    detection_time_eval = DetectionTime(model, settings, npy_dataset) # Initiliaze DetectionTime instance
    detection_t_results = detection_time_eval.compute_detection_times()
    log_detection_time(logger, detection_t_results)

    # Plots
    plots.compute_plots()  # Produce and Save plots
    plots.save_plot_data() # Save the data to reproduce the plots

    # Copy scripts for postprocessing to the model directory
    shutil.copy(plot_results_path, model_dir_path / "Plots") # "plot_results.py"
    shutil.copy(plot_fault_path, model_dir_path / "Plots")   # "plot_fault.py"
    shutil.copy(inference_path, model_dir_path)              # "LSTM_inference.py"

    # Save final model after training
    final_model_path = model_dir_path / "final_model.pth"
    torch.save(model.state_dict(), final_model_path) 

    # Remove and close all handlers associated with the logger
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    for handler in model_logger.handlers[:]:
        handler.close()
        model_logger.removeHandler(handler)
    for handler in console_logger.handlers[:]:
        handler.close()
        console_logger.removeHandler(handler)

    # Rename the model directory to include metrics
    new_model_dir = f"LSTM_{settings.model_suffix}_{timestamp}_TrainAcc_{train_metrics.acc:.4f}_ValAcc_{val_metrics.acc:.4f}_F1_{val_metrics.f1_macro:.4f}_BestF1_{max_val_p.f1:.4f}"
    new_model_dir_path = Path(current_dir) / "LSTM_Trained_Models" / new_model_dir
    os.rename(model_dir_path, new_model_dir_path)

if __name__ == "__main__":
    main()
