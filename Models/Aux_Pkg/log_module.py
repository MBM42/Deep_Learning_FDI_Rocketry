"""
log_module.py 

Module to help with logging tasks.

Functions:
    log_model(): logs the model architecture. [INFO]
    log_settings(): logs the settings of the model and data processing. [INFO]
    log_dataset_mem_size(): logs the total memory size of a dataset folder and available system RAM. [INFO]
    log_hyperparameters(): logs the hyperparameters of the FNN model. [INFO]
    log_hyperparameters_LSTM(): logs the hyperparameters of the LSTM model. [INFO]
    log_main_p_metrics(): logs the main performance metrics. [INFO]
    log_dataset(): logs metrics about the dataset. [DEBUG]
    log_dataset_AE(): logs metrics about the datasets for AE models. [DEBUG]
    log_detection_time(): logs the dection time results. [INFO]

Author: Miguel Marques
Date: 22-03-2025
"""

from pathlib import Path
from torchinfo import summary
import re
import psutil

def log_model(logger, model, loss_func, optimizer, batch_size, input_dim):
    """
    Logs the model architecture. [INFO]

    Params:
        logger: logger object.
        model
        loss_func: loss function.
        optimizer
        batch_size
        input_dim
    """

    logger.info("================== Architecture ==================")
    logger.info("Model architecture:")
    logger.info(f"- {model}")
    logger.info("\nLoss function:")
    logger.info(f"- {loss_func}")
    
    if hasattr(loss_func, 'weight'):
        logger.info("\nLoss function weights:")
        logger.info(f"- {loss_func.weight}")
    
    logger.info("\nOptimizer:")
    logger.info(f"- {optimizer}")
    logger.info("\n")

    # figure out the shape torchinfo needs:
    if isinstance(input_dim, tuple):
        inp = (batch_size, * input_dim)
    else:
        inp = (batch_size, input_dim)
    
    # Model Summary from torchinfo
    model_summary = str(summary(model, input_size = inp, col_names=["input_size", "output_size", "num_params"], verbose=0))
    # Replacing column headers with more inteligible description
    model_summary = model_summary.replace("Input Shape", "Input [batch, input]")
    model_summary = model_summary.replace("Output Shape", "Output [batch, output]")
    # Flushing Column headers to the left by removing extra spaces
    model_summary = re.sub(r'(Input \[batch, input\]) {9}', r'\1', model_summary)
    model_summary = re.sub(r'(Output \[batch, output\]) {}', r'\1', model_summary)
    logger.info(model_summary)


def log_settings(logger, timestamp, debug_mode, settings):
    """
    Logs multiple settings for preprocessing and the model. [INFO]

    Params:
        logger (logging.Logger): Logger object.
        timestamp (str): timestamp of the experiment.
        debug_mode (bool)
        settings (FNNSettings): instance of FNNSettings with attributes to self explanatory settings.
    """

    logger.info("==================== Settings ====================")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Debug Mode: {debug_mode}\n")
    
    logger.info(f"Fixed Seed: {settings.fixed_seed}\n")
    
    logger.info(f"NPYDataset (class):")
    logger.info(f"- Data Types = {settings.data_types}")
    logger.info(f"- Gather All Data: {settings.gather_all}")
    logger.info(f"- Create Test Set: {settings.create_test_set}")
    logger.info(f"- Save Datasets: {settings.save_datasets}")
    logger.info(f"- Normalize Class Weights: {settings.normalize_weights}")
    logger.info(f"- Use Manual Weights: {settings.use_manual_weights}\n")
    
    logger.info(f"SlidingWindow (class):")
    logger.info(f"- Sequence Window: {settings.window_size}")
    logger.info(f"- Scaler Type: {settings.scaler_type}")
    logger.info(f"- Flatten Window: {settings.flatten_window}\n")

    logger.info(f"Loss function:")
    logger.info(f"- Use Focal Loss: {settings.use_focal_loss}")
    logger.info(f"- Focal Loss Gamma: {settings.gamma_fl}") 
    logger.info(f"- Apply Class Weights: {settings.apply_class_weights}\n")

    logger.info(f"Optimizer:")
    logger.info(f"- L2 Regularization (Weight Decay): {settings.decay}\n")

    logger.info(f"Learning Rate Scheduler:")
    logger.info(f"- Scheduler Mode: {settings.scheduler_mode}")
    logger.info(f"- Scheduler Patience: {settings.scheduler_patience}")
    logger.info(f"- Scheduler Factor: {settings.scheduler_factor}\n")

    logger.info(f"Early Stop:")
    logger.info(f"- Patience: {settings.stop_patience}")
    logger.info(f"- Minimum delta: {settings.stop_min_delta}")
    logger.info(f"- Mode: {settings.stop_mode}\n")

    logger.info(f"DataLoader (class):")
    logger.info(f"- Shuffle Samples: {settings.shuffle_samples}")
    logger.info(f"- Number of Loader Workers: {settings.loader_workers}")
    logger.info(f"- Pin Memory: {settings.pin_memory}\n")

    logger.info(f"Performance Metrics:")
    logger.info(f"- Exclude Normal Class: {settings.filter_normal}")
    logger.info("==================================================\n\n")


def log_settings_ad(logger, timestamp, debug_mode, settings):
    """
    Logs all AD‐LSTM training and data settings. [INFO]

    Params:
        logger (logging.Logger): logger to write to
        timestamp (str): timestamp of the experiment.
        debug_mode (bool)
        settings (AD_LSTMSettings): instance holding all config
    """
    logger.info("==================== Settings ====================")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Debug Mode: {debug_mode}\n")

    logger.info(f"Fixed Seed: {settings.fixed_seed}\n")

    logger.info("Sliding window / forecast:")
    logger.info(f"- window_size (p): {settings.window_size}")
    logger.info(f"- pred_length (l): {settings.pred_length}")
    logger.info(f"- scaler_type: {settings.scaler_type}\n")

    logger.info("Optimizer:")
    logger.info(f"- weight_decay (L2): {settings.decay}\n")

    logger.info("LR Scheduler (ReduceLROnPlateau):")
    logger.info(f"- mode: {settings.scheduler_mode}")
    logger.info(f"- patience: {settings.scheduler_patience}")
    logger.info(f"- factor: {settings.scheduler_factor}\n")

    logger.info("Early stopping:")
    logger.info(f"- mode: {settings.stop_mode}")
    logger.info(f"- patience: {settings.stop_patience}")
    logger.info(f"- min_delta: {settings.stop_min_delta}\n")

    logger.info("DataLoader (class):")
    logger.info(f"- Shuffle Samples: {settings.shuffle_samples}")
    logger.info(f"- Number of Loader Workers: {settings.loader_workers}")
    logger.info(f"- Pin Memory: {settings.pin_memory}\n")

    logger.info("Anomaly Detection Threshold:")
    logger.info(f"- Threshold Percentile: {settings.threshold_percentile}\n")

    logger.info("Performance metrics:")
    logger.info(f"- Exclude Normal Class: {settings.filter_normal}")
    logger.info("=================================================\n")


def log_dataset_mem_size(logger, dataset_path) -> None:
    """
    Logs the total memory size of a dataset folder and available system RAM. [INFO]
    Warns if dataset size exceeds available RAM.

    For NumPy files, the folder size should be approximately the same as the amount of 
    memory RAM it takes to cache it.

    Params:
        logger (logging.Logger): logger object.
        dataset_path (str or Path): path to dataset folder.
    """

    dataset_path = Path(dataset_path)

    # Compute dataset size in bytes
    total_bytes = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file())
    total_gb = total_bytes / (1024 ** 3)

    # Get available RAM in GB
    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)

    logger.info("================ Dataset Size =====================")
    logger.info(f"- Dataset location: {dataset_path.resolve()}")
    logger.info(f"- Dataset size: {total_gb:.2f} GB")
    logger.info(f"- Available RAM: {available_ram_gb:.2f} GB")

    if total_gb > available_ram_gb:
        logger.warning(f"⚠️  Dataset is larger than available RAM — you may run into Out-Of-Memory issues during caching!")
    else:
        logger.info(f"✓ Dataset fits in available RAM.")
    logger.info("==================================================\n")


def log_hyperparameters(logger, hyperparams, loss_function, optimizer):
    """
    Logs the hyperparameters of the FNN model. [INFO]

    Params:
        logger (logging.Logger): logger object.
        hyperparams (FNNHyperparameters)
        loss_function (str)
        optimizer (str)
    """
    logger.info("================= Hyperparameters ================")
    logger.info(f"Neurons per Hidden Layer: {hyperparams.hidden_dim}")
    logger.info(f"Dropout Rate: {hyperparams.drop_prob}")
    logger.info(f"Batch Size: {hyperparams.batch_size}")
    logger.info(f"Number of Epochs: {hyperparams.num_epochs}")
    logger.info(f"Initial learning Rate: {hyperparams.learning_rate}")
    logger.info(f"Loss Function: {loss_function}")
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info("==================================================\n\n")


def log_hyperparameters_AD_LSTM(logger, hyperparams):
    """
    Logs the hyperparameters of the AD_LSTM model. [INFO]

    Params:
        logger (logging.Logger): logger object.
        hyperparams (LSTMHyperparameters)
    """
    logger.info("================= Hyperparameters ================")
    logger.info(f"Neurons per Hidden Layer: {hyperparams.hidden_dim}")
    logger.info(f"Number of LSTM Layers: {hyperparams.num_layers}")
    logger.info(f"Dropout Rate: {hyperparams.drop_prob}")
    logger.info(f"Batch Size: {hyperparams.batch_size}")
    logger.info(f"Number of Epochs: {hyperparams.num_epochs}")
    logger.info(f"Clip L2 Gradient: {hyperparams.clip_grad_norm}")
    logger.info(f"Initial learning Rate: {hyperparams.learning_rate}")
    logger.info("==================================================\n")


def log_hyperparameters_LSTM(logger, hyperparams, loss_function, optimizer):
    """
    Logs the hyperparameters of the LSTM model. [INFO]

    Params:
        logger (logging.Logger): logger object.
        hyperparams (LSTMHyperparameters)
        loss_function (str)
        optimizer (str)
    """
    logger.info("================= Hyperparameters ================")
    logger.info(f"Neurons per Hidden Layer: {hyperparams.hidden_dim}")
    logger.info(f"Number of LSTM Layers: {hyperparams.num_layers}")
    logger.info(f"Dropout Rate: {hyperparams.drop_prob}")
    logger.info(f"Batch Size: {hyperparams.batch_size}")
    logger.info(f"Number of Epochs: {hyperparams.num_epochs}")
    logger.info(f"Bidirectional: {hyperparams.bidirectional}")
    logger.info(f"Clip L2 Gradient: {hyperparams.clip_grad_norm}")
    logger.info(f"Initial learning Rate: {hyperparams.learning_rate}")
    logger.info(f"Loss Function: {loss_function}")
    logger.info(f"Optimizer: {optimizer.__class__.__name__}")
    logger.info("==================================================\n\n")


def log_main_p_metrics(logger, metrics):
    """
    Logs the main performance metrics. [INFO]

    Params:
        logger (logging.Logger): Logger object.
        metrics (<class 'p_metrics.PerformanceMetrics'>): PerformanceMetrics object.
    """
      
    logger.info(f"{metrics.name} Metrics:")
    logger.info(f"- Precision: {metrics.precision_macro:.4f}")
    logger.info(f"- Recall: {metrics.recall_macro:.4f}")
    logger.info(f"- F1 Score: {metrics.f1_macro:.4f}")
    logger.info(f"- MCC: {metrics.mcc:.4f}")
    # Precision & recall per class
    for cls, name in metrics.class_labels.items():
        logger.info(f"- Class {name}: Val Precision = {metrics.precision_per_class[cls]:.4f}, Val Recall = {metrics.recall_per_class[cls]:.4f}")


def log_dataset(logger, dataset, train_dataset, val_dataset, test_dataset):
    """
    Logs metrics of the npy_dataset, including balance ratio. [DEBUG]
    
    Params:
        logger (logging.Logger): logger.object.
        dataset (NPYDataset)
        train_dataset (SlidingWindow)
        val_dataset (SlidingWindow)
        test_dataset (SlidingWindow)
    """

    logger.debug("\n\n====================== Data ======================")
    logger.debug(f"Selected features: {dataset.selected_features}\n")
    logger.debug(f"Selected labels: {dataset.selected_labels}")
    logger.debug("")

    logger.debug("Data type balance, files per data type:")
    for category, count in dataset.category_counts.items():
        logger.debug(f"- {category}: {count} files")

    logger.debug("\nClass balance in training data:")
    for label, count in dataset.class_instances.items():
        file_count = dataset.class_f_count_train[label]
        logger.debug(f"- {label}: {count:,} instances, present in {file_count} files")    
    # Compute total, normal, and fault instances on the fly
    total_instances = sum(dataset.class_instances.values())
    normal_instances = dataset.class_instances.get("Normal", 0)
    fault_instances = total_instances - normal_instances
    logger.debug(f"- Total Data points: {total_instances:,}")
    logger.debug(f"- Total Normal instances: {normal_instances:,}")
    logger.debug(f"- Total Fault instances: {fault_instances:,}")
    logger.debug(f"- Imbalance ratio: {dataset.imbalance_metric:.2f}%")

    if getattr(dataset, 'class_f_count_val', None):
        logger.debug("\nClass file count in validation data:")
        for label, count in dataset.class_f_count_val.items():
            logger.debug(f"- {label}: present in {count} files")

    if getattr(dataset, 'class_f_count_test', None):
        logger.debug("\nClass file count in test data:")
        for label, count in dataset.class_f_count_val.items():
            logger.debug(f"- {label}: present in {count} files")

    logger.debug("\nLoss Function Weights:")
    for label_index, weight in enumerate(dataset.class_weights):
        logger.debug(f"- {dataset.selected_labels[label_index]}: {weight:.6f}")

    logger.debug("\nGlobal Normalization Statistics:")
    logger.debug(f"- Global Mean: {dataset.global_mean}")
    logger.debug(f"- Global Std: {dataset.global_std}")
    logger.debug(f"- Global Min: {dataset.global_min}")
    logger.debug(f"- Global Max: {dataset.global_max}")

    logger.debug("\nFeatures and labels set - torch:")
    # Train
    logger.debug(f"- Total training windows: {len(train_dataset)}")
    sample, _ = train_dataset[0]
    logger.debug(f"- Shape of one training sample: {sample.shape}")
    # Validation
    logger.debug(f"- Total validation windows: {len(val_dataset)}")
    sample_val, _ = val_dataset[0]
    logger.debug(f"- Shape of one validation sample: {sample_val.shape}")
    # Test
    if dataset.create_test_set:
        logger.debug(f"Total test windows: {len(test_dataset)}")
        sample_test, _ = test_dataset[0]
        logger.debug(f"Shape of one test sample: {sample_test.shape}")

    logger.debug("==================================================")


def log_dataset_AE(logger, dataset):
    """
    Logs metrics about the datasets for AE models. [DEBUG]

    Params:
        logger (logging.Logger): logger.object.
        dataset (Dataset): dataset object.
    """

    logger.debug("\n\n====================== Data ======================")
    logger.debug("Features and labels set - numpy:")
    logger.debug(f"- Train features shape: {dataset.features_train.shape}")
    logger.debug(f"- Train labels shape: {dataset.labels_train.shape}")
    logger.debug(f"- Validation features shape: {dataset.features_validate.shape}")
    logger.debug(f"- Validation labels shape: {dataset.labels_validate.shape}")
    if dataset.create_test_set:
        logger.debug(f"- Test features shape: {dataset.features_test.shape}")
        logger.debug(f"- Test labels shape: {dataset.labels_test.shape}")

    logger.debug("==================================================")


def log_detection_time(logger, detection_t_results):
    """
    Logs the detection time results. [INFO]

    Params:
        logger (logging.Logger): Logger object.
        metrics (<class 'p_metrics.PerformanceMetrics'>): PerformanceMetrics object.
    """
    logger.info("\n\n================= Detection Time =================")
    for key, value in detection_t_results.items():
        if key in ("Missed Sims", "Total Sims"):
            logger.info(f"{key}: {int(value)}")
        elif isinstance(value, (float, int)):
            logger.info(f"{key}: {value:.1f} t_stamps")
        else:
            logger.info(f"{key}: missed")
    logger.info("==================================================")

def log_data_stats_ad(
    instance_counts,
    file_counts,
    total_points,
    total_normal,
    total_faults,
    imbalance,
    num_train_files,
    total_train_timestamps,
    num_val_files,
    total_val_timestamps,
    selected_features,
    selected_labels,
    logger=None
):
    """
    Log dataset statistics for anomaly detection using precomputed metrics. [INFO]

    Args:
        instance_counts (Counter):
            Mapping from each class label to the total number of label occurrences
            across all processed test files.
        file_counts (dict):
            Mapping from each class label to the number of distinct test files in which
            that label appears at least once.
        total_points (int):
            Total number of label instances across all test files.
        total_normal (int):
            Number of instances labeled as 'Normal' (or label 0 if numeric) in the test set.
        total_faults (int):
            Number of non-normal (fault) instances in the test set.
        imbalance (float):
            Ratio of fault instances to normal instances in the test set.
        num_train_files (int):
            Number of simulation files in the training cache.
        total_train_timestamps (int):
            Sum of timestamps across all training cache files.
        num_val_files (int):
            Number of simulation files in the validation cache.
        total_val_timestamps (int):
            Sum of timestamps across all validation cache files.
        selected_features (List[str])
        selected_labels (List[str])
        logger (logging.Logger, optional):
            Logger instance to use.
    """
    # Log cache statistics
    logger.info("\n====================== Data ======================")
    logger.info(f"Selected features: {selected_features}\n")
    logger.info(f"Selected labels: {selected_labels}\n")
    logger.info(f"Train simulations : {num_train_files} files, {total_train_timestamps} timestamps")
    logger.info(f"Val simulations : {num_val_files} files, {total_val_timestamps} timestamps\n")

    logger.info(f"Test Dataset:")
    logger.info(f"- Total data points: {total_points}")
    logger.info(f"- Normal: {total_normal}")
    logger.info(f"- Faults: {total_faults}")
    logger.info(f"- Imbalance ratio: {imbalance:.2f}%\n")
    for cls_name, cnt in instance_counts.items():
        logger.info(f"- {cls_name}: {cnt} instances, in {file_counts[cls_name]} files")
    logger.info("==================================================\n")

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
