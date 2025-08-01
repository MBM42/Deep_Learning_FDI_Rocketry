"""
LSTM_settings.py

This module defines the LSTMSettings and LSTMHyperparameters classes, which consolidate all configuration parameters to train an LSTM.

Author: Miguel Marques
Date: 07-04-2025
"""

from dataclasses import dataclass, field

@dataclass
class LSTMHyperparameters:
    hidden_dim: int = 256         # Neurons per hidden Layer
    num_layers: int = 2           # Number of LSTM layers
    drop_prob: float = 0.24       # Dropout probability
    learning_rate: float = 0.001  # Learning rate for the optimizer
    batch_size: int = 512         # Batch size for training
    num_epochs: int = 60          # Number of training epochs
    bidirectional: bool = False   # Bidirectionality
    clip_grad_norm: float = None  # Maximum L2 gradient for gradient clipping

@dataclass
class LSTMSettings:
    # Data Settings:
    data_types: list = field(default_factory=lambda: ['Normal', 'Valve', 'Block', 'Block_Leak', 'Sensor_Fault'])
    #data_types: list = field(default_factory=lambda: ['.'])
    data_folder: str = "../Data/New_Data_clip_npy"
    
    # Model Definitions:
    model_suffix: str = "Identification_Trial1_Trial1"

    # Seed:
    fixed_seed: bool = True            # True: Set fixed seed for reproducibility
    seed: int = 42

    # NPYDataset (class):
    gather_all: bool = True            # True: Gather all data; False: Gather an even amount of each data type ['Normal', 'Valve', ...]
    create_test_set: bool = False      # True: Creates train, validation and test sets; False: Only creates train and validation sets
    save_datasets: bool = True         # True: Saves the Validation (and Test Datasets if created)
    normalize_weights: bool = True     # True: Normalize calculated class weights (doesn't apply to the manual weights)
    use_manual_weights: bool = True    # True: Uses Manual Weights defined in "weights.py"

    # SlidingWindow (class):
    window_size: int = 11              # Number of consecutive timestamps to include in each sliding window transformation of the data.
                                       # - If set to "1", no sliding window transformation is applied.
    scaler_type: str = 'zscore'        # Data normalization:
                                       # - 'zscore': Removes the mean to center the data and divides by the standard deviation to scale it to a variance of 1.
                                       # - 'minmax': Scales the data to a fixed range [0, 1].
    flatten_window: bool = False       # True: Flatten Window Input to 1D (FNN); False: Input Window is kept as a 2D tensor (LSTM)
    save_debug_sample: bool = False    # Call ".save_debug_samples" method from "SlidingWindow" Class: saves window samples from a .npy file for debug
    debug_type: str = 'Block'          # Type of file to debug e.g.: 'Normal', 'Valve', etc.
    debug_windows: int = 10            # Number of windows to collect

    # Loss Function:
    use_focal_loss: bool = True        # True: Custom Focal Loss (focal_loss.py); False: CrossEntropyLoss
    gamma_fl: int = 0.0002             # Scaling factor (gamma) for the focal loss
    apply_class_weights: bool = True   # True: Apply class weights to the loss function

    # Optimizer:
    decay: float = 0.0002              # L2 Regularization Parameter (float). If set to 0.0 L2 regularization is not performed

    # Learning Rate Scheduler:
    scheduler_mode: str = 'min'        # Mode for ReduceLROnPlateau scheduler: 'min': (lr reduced when quantity monitored has stopped decreasing)
    scheduler_patience: int = 4        # Number of epochs with no improvement after which learning rate will be reduced
    scheduler_factor: float = 0.5      # Factor by which the learning rate will be reduced

    # Early Stopping Settings:
    stop_patience: int  = 60           # Number of epochs with no improvement after which training will be stopped 
    stop_min_delta: float = 0.0        # Minimum improvement value
    stop_mode: str = 'max'             # Mode: If 'max' a greater value is considered an improvement

    # DataLoader (class):
    shuffle_samples: bool = True       # True: Shuffle samples; False: Keep loading order
    loader_workers: int = 8            # Number of workers for data loader (rule of thumb: num_workers = number of CPU cores / 2)
    pin_memory: bool = True            # True: DataLoader allocates page-locked memory for batches (Faster transfer to GPU)

    # PerformanceMetrics (class):
    filter_normal: bool = True         # True: Exclude class 0 ("Normal") from performance metrics

    # Console Logging:
    log_train_extra_metrics: bool = False
