"""
npy_preprocessing.py

This module implements the NPYDataset class to preprocess the training data at the "*.npy" level.

It provides methods for:
- Setting a fixed seed for reproducibility of randomization steps: set_seed()
- Counting the number of .npy files per data category: count_npy_files()
- Gathering .npy file paths into a dictionary whose keys are the type of set ('Train', 'Validation', etc): gather_data()
- Caches the content of the .npy files from each data type for faster processing:cache_npy_files()
- Saving the validation and test (if created) datasets: save_data()
- Counting instances and files per class across train .npy files, computing class weights and imbalance metrics: count_class_instances()
- Counting files per class for a given dataset (validation or test): count_class_files()
- Cheks if all labels are presents in a given Dataset: check_labels()
- Computing global normalization statistics from training .npy files: compute_normalization_stats()

Author: Miguel Marques
Date: 20-03-2025
"""

import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
from FNN_settings import *

class NPYDataset:
    """   
    Params:
        root_dir (str): Root directory containing the data
        logger (logging.Logger): Logger object
        settings (FNNSettings)
        selected_features (List[str])
        selected_labels (List[str])

    Extra Attributes:
        root_dir (str): Root directory containing the data
        logger (logging.Logger): Logger object
        settings (FNNSettings)
        selected_features (List[str])
        selected_labels (List[str])
        categories (List[str]): List of data categories (e.g.: ['Normal', 'Valve', ...])
        create_test_set (bool): If True, a test set is created
        seed (int): Fixed Seed
        gather_all(bool): If True, gathers all .npy files for each category. Otherwise, gathers an even number per category (for data imbalance countermeasures)
        normalize_weights (bool): If True, normalize the computed class weights so that they sum to 1
        manual_weights (bool): True: Uses Manual Weights defined in "weights.py"
        category_counts (Dict[str, int]): Number of .npy files per category
        npy_set (Dict[str, List[Path]]): Dictionary of .npy file paths split by dataset type (Train, Validation, Test)
        train_cache (List[Dict[str, np.ndarray]]): List of training file data dicts with 'features' and 'labels'
        val_cache (List[Dict[str, np.ndarray]]): List of validation file data dicts with 'features' and 'labels'
        test_cache (List[Dict[str, np.ndarray]]): List of test file data dicts with 'features' and 'labels'
        class_instances (Dict[str, int]): Count of timestamps for each class (Train Dataset)
        class_f_count_train (Dict[str, int]): Count of files for each class (Train Dataset)
        class_f_count_val (Dict[str, int]): Count of files for each class (Val Dataset)
        class_f_count_test (Dict[str, int]): Count of files for each class (Test Dataset)
        class_weights (np.ndarray): Computed class weights for balancing training
        imbalance_metric (float): Metric quantifying class imbalance (normal vs. fault)
        global_mean (np.ndarray): Global mean of each feature from the training set
        global_std (np.ndarray): Global standard deviation of each feature
        global_min (np.ndarray): Global minimum value of each feature
        global_max (np.ndarray): Global maximum value of each feature
    """

    def __init__(
        self,
        root_dir: str,
        logger: logging.Logger,
        settings,
        selected_features: list[str],
        selected_labels : list[str],
    ) -> None:
        self.root_dir = Path(root_dir)
        self.logger = logger
        self.settings = settings
        self.selected_features = selected_features
        self.selected_labels = selected_labels
        self.categories = settings.data_types
        self.create_test_set = settings.create_test_set
        self.seed = settings.seed
        self.gather_all = settings.gather_all
        self.normalize_weights = settings.normalize_weights
        self.manual_weights = settings.use_manual_weights

        # Attributes that will be computed
        self.category_counts: Dict[str, int] = None     # Store the number of .npy files per data category e.g.: {'Normal': n1, 'Valve': n2, 'Block': n2, (...)}
        self.npy_set: Dict[str, list[Path]] = {}        # Dict containing a list of paths for each set e.g.: {'Train': [...], 'Validation': [...], (...)}
        self.train_cache: list[Dict[str, np.ndarray]]   # List containing a Dict for each Train file: {'features': [np.array], 'labels': [np.array]}
        self.val_cache: list[Dict[str, np.ndarray]]     # List containing a Dict for each Validation file: {'features': [np.array], 'labels': [np.array]}
        self.test_cache: list[Dict[str, np.ndarray]]    # List containing a Dict for each Test file: {'features': [np.array], 'labels': [np.array]}
        self.class_instances: Dict[str, int] = {}       # Stores the number of timestamps of each class
        self.class_f_count_train: Dict[str, int] = {}   # Stores the number of files of each class (train)
        self.class_f_count_val: Dict[str, int] = {}     # Stores the number of files of each class (val)
        self.class_f_count_test: Dict[str, int] = {}    # Stores the number of files of each class (test)
        self.class_weights: np.ndarray = None           # Stores the class weights of each class
        self.imbalance_metric: float = None
        self.global_mean: np.ndarray = None             # Array with for the global mean of each feature
        self.global_std: np.ndarray = None              # Array with for the global standard deviation of each feature
        self.global_min: np.ndarray = None              # Array with for the global minimum value of each feature
        self.global_max: np.ndarray = None              # Array with for the global maximum value of each feature


    def set_seed(self) -> None:
        """
        Sets a fixed seed for reproducibility.
        """

        if not isinstance(self.seed, int):
            raise TypeError("Seed must be an integer.")
        random.seed(self.seed)
        np.random.seed(self.seed)

    
    def count_npy_files(self) -> None:
        """
        Recursively counts .npy files for each data category and logs the counts.
        The resulting counts are stored as self.category_counts.
        """

        # Reset category counts
        self.category_counts = {category: 0 for category in self.categories}

        for category in self.categories:
            category_path = self.root_dir / category
            if category_path.exists():
                npy_files = list(category_path.rglob("*.npy"))
                self.category_counts[category] = len(npy_files)
            else:
                self.logger.debug(f"Category '{category}' not found in '{self.root_dir}'.")


    def gather_data(self) -> None:
        """
        Gathers .npy file paths for each data category and splits them into Train, Validation, and (if applicable) Test sets.
        The resulting dictionary is stored as self.npy_set.

        If self.create_test_set:
        - True: 70% for training, 20% for validation, remaining (~10%) for testing.
        - False: 75% for training, remaining (~25%) for validation. Test set is empty.
        """

        # Initializing dictionaries to store the .npy files per set type
        if self.create_test_set:
            npy_set = {key: [] for key in ['Train', 'Validation', 'Test']}
        else:
            npy_set = {key: [] for key in ['Train', 'Validation']}

        # Gather .npy file paths per category
        cat_files = {}
        for category in self.categories:
            cat_path = self.root_dir / category
            if cat_path.exists():
                files = list(cat_path.rglob("*.npy"))
                if len(files) == 0:
                    raise ValueError(f"No .npy files found for category '{category}' in '{cat_path}'.")
                cat_files[category] = files
            else:
                raise FileNotFoundError(f"Category '{category}' not found in '{self.root_dir}'.")

        # Shuffle files (deterministically if seed is set)
        for files in cat_files.values():
            if isinstance(self.seed, int):
                rng = random.Random(self.seed)
                rng.shuffle(files)
            else:
                random.shuffle(files)

        # Split the data
        if self.gather_all:
            for category, paths in cat_files.items():
                n_files = len(paths)
                if self.create_test_set:
                    n_train = int(0.7 * n_files)
                    n_val = int(0.2 * n_files)
                else:
                    n_train = int(0.75 * n_files)
                    n_val = n_files - n_train
                npy_set['Train'].extend(paths[:n_train])
                npy_set['Validation'].extend(paths[n_train:n_train+n_val])
                if self.create_test_set:
                    npy_set['Test'].extend(paths[n_train+n_val:])
        
        else: # Use even number of files from each category (based on the minimum count across categories)
            
            # Count the .npy files in each category if not already counted
            if not getattr(self, 'category_counts', None):
                self.count_npy_files()
            
            # Find the limiting category    
            min_count = min(self.category_counts.values())

            for category, paths in cat_files.items():
                if self.create_test_set:
                    n_train = int(0.7 * min_count)
                    n_val = int(0.2 * min_count)
                else:
                    n_train = int(0.75 * min_count)
                    n_val = min_count - n_train
                npy_set['Train'].extend(paths[:n_train])
                npy_set['Validation'].extend(paths[n_train:n_train+n_val])
                if self.create_test_set:
                    npy_set['Test'].extend(paths[n_train+n_val:])
        
        self.npy_set = npy_set

    def cache_npy_files(self) -> None:
        """
        Caches the content of the .npy files from each data type in the npy_set dictionary.
        Each file is loaded using np.load() and stored in separate caches : self.train_cache, self.val_cache,
        and self.test_cache (if the Test set exists).

        Filters both 'features' and 'labels' down to only the columns in self.selected_features / self.selected_labels.
        """

        def _filter_data(data: dict):
            # 'data' must have 'features', 'labels', 'feature_names', 'label_names'
            feats = data["features"]  # shape (timestamps, n_features)
            labs  = data["labels"]    # shape (timestamps, n_classes)

            # Get the name lists
            fnames = data.get("feature_names", None)
            lnames = data.get("labels_names",   None)

            # --- Features ---
            if fnames is not None:
                # Find indices of the selected features
                idxs = [fnames.index(f) for f in self.selected_features]
                feats = feats[:, idxs]
            else:
                raise ValueError("Cannot select features by name: 'fnames' is missing.")

            # --- Labels ---
            if lnames is not None:
                idxs = [lnames.index(l) for l in self.selected_labels]
                labs = labs[:, idxs]
            else:
                raise ValueError("Cannot select labels by name: 'lnames' is missing.")

            return {"features": feats, "labels": labs}

        self.train_cache = []
        self.val_cache = []
        if 'Test' in self.npy_set:
            self.test_cache = []
        
        # Train Cache
        for path in tqdm(self.npy_set.get('Train', []), desc="Caching Train Files", unit="file"):
            try:
                raw_data = np.load(path, allow_pickle=True).item()
                if not isinstance(raw_data, dict) or "features" not in raw_data or "labels" not in raw_data:
                    raise ValueError(f"File {path} does not contain the expected keys 'features' and 'labels'.")
            except Exception as e:
                self.logger.error(f"Error loading file {path}: {e}")
                raise
            filtered = _filter_data(raw_data)
            self.train_cache.append(filtered)
        
        # Validation Cache
        for path in tqdm(self.npy_set.get('Validation', []), desc="Caching Validation Files", unit="file"):
            try:
                raw_data = np.load(path, allow_pickle=True).item()
                if not isinstance(raw_data, dict) or "features" not in raw_data or "labels" not in raw_data:
                    raise ValueError(f"File {path} does not contain the expected keys 'features' and 'labels'.")
            except Exception as e:
                self.logger.error(f"Error loading file {path}: {e}")
                raise
            filtered = _filter_data(raw_data)
            self.val_cache.append(filtered)
        
        # Test Cache
        if 'Test' in self.npy_set:
            for path in tqdm(self.npy_set.get('Test', []), desc="Caching Test Files", unit="file"):
                try:
                    raw_data = np.load(path, allow_pickle=True).item()
                    if not isinstance(raw_data, dict) or "features" not in raw_data or "labels" not in raw_data:
                        raise ValueError(f"File {path} does not contain the expected keys 'features' and 'labels'.")
                except Exception as e:
                    self.logger.error(f"Error loading file {path}: {e}")
                    raise
                filtered = _filter_data(raw_data)
                self.test_cache.append(filtered)


    def save_data(self, dir_path: Path) -> None:
        """
        Saves the validation and test datasets (if it has been created) in "dir_path". In both Raw and Normalized state.

        Additionaly saves the normalization statistics (global_mean, global_std, global_min, global_max)
        into a JSON file within the same directory.
        
        Parameters:
            dir_path (Path): Directory where the datasets will be saved.
        """

        # Ensure the target directory exists.
        dir_path.mkdir(parents=True, exist_ok=True)

        ## Raw Data
        # Create subdirectory for raw validation data
        raw_dir = dir_path / "Raw"
        raw_val_dir = raw_dir / "Validation"
        raw_val_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw validation dataset: one file per simulation
        for i, data in enumerate(self.val_cache):
            file_path = raw_val_dir / f"val_dataset_{i}.npy"
            try:
                np.save(file_path, data)
            except Exception as e:
                raise IOError(f"Error saving raw validation simulation {i} to {file_path}: {e}") from e

        # Save raw test dataset if available     
        if self.create_test_set and hasattr(self, "test_cache") and self.test_cache:
            raw_test_dir = raw_dir / "Test"
            raw_test_dir.mkdir(parents=True, exist_ok=True)
            for i, data in enumerate(self.test_cache):
                file_path = raw_test_dir / f"test_dataset_{i}.npy"
                try:
                    np.save(file_path, data)
                except Exception as e:
                    raise IOError(f"Error saving raw test simulation {i} to {file_path}: {e}") from e

        ## Normalized Data
        # Check for Normalization stats
        if self.global_mean is None or self.global_std is None or self.global_min is None or self.global_max is None:
            raise ValueError("Normalization statistics have not been computed. Please run compute_normalization_stats() first.")

        # Create subdirectory for normalized validation data
        norm_dir = dir_path / "Normalized"
        norm_val_dir = norm_dir / "Validation"
        norm_val_dir.mkdir(parents=True, exist_ok=True)
        
        # Save normalized validation dataset: one file per simulation
        for i, data in enumerate(self.val_cache):
            features = data["features"]
            # Normalize features based on the scaler type:
            if self.settings.scaler_type == "zscore":
                norm_features = (features - self.global_mean) / self.global_std
            elif self.settings.scaler_type == "minmax":
                denom = self.global_max - self.global_min
                # Avoid division by zero:
                denom[denom == 0] = 1.0
                norm_features = (features - self.global_min) / denom
            else:
                raise ValueError(f"Unknown scaler type: {self.settings.scaler_type}.")
            # Create a new dictionary with normalized features
            norm_data = {
                "features": norm_features,
                "labels": data["labels"]
            }
            file_path = norm_val_dir / f"val_dataset_{i}.npy"
            try:
                np.save(file_path, norm_data)
            except Exception as e:
                raise IOError(f"Error saving normalized validation simulation {i} to {file_path}: {e}") from e

        # Save normalized test dataset if available
        if self.create_test_set and hasattr(self, "test_cache") and self.test_cache:
            norm_test_dir = norm_dir / "Test"
            norm_test_dir.mkdir(parents=True, exist_ok=True)
            for i, data in enumerate(self.test_cache):
                features = data["features"]
                if self.settings.scaler_type == "zscore":
                    norm_features = (features - self.global_mean) / self.global_std
                elif self.settings.scaler_type == "minmax":
                    denom = self.global_max - self.global_min
                    denom[denom == 0] = 1.0
                    norm_features = (features - self.global_min) / denom
                else:
                    raise ValueError(f"Unknown scaler type: {self.settings.scaler_type}.")
                norm_data = {
                    "features": norm_features,
                    "labels": data["labels"]
                }
                file_path = norm_test_dir / f"test_dataset_{i}.npy"
                try:
                    np.save(file_path, norm_data)
                except Exception as e:
                    raise IOError(f"Error saving normalized test simulation {i} to {file_path}: {e}") from e


    def count_class_instances(self, weights_manual) -> None:
        """
        Computes class counts (intances and files), class weights, and the imbalance metric (from the training set).
        The results are stored in self.class_instances, self.class_f_count_train, self.class_weights, and self.imbalance_metric.
    
        If "self.manual_weights" is True, "self.class_weights" is set to weights_manual.

        Params:
            weights_manual (np.ndarray): manually defined weights.
        """
       
        if not getattr(self, 'train_cache', None):
            raise AttributeError("Train cache not available. Please run cache_npy_files() first.")

        # Initialize counts
        instance_counts = {header: 0 for header in self.selected_labels}
        file_counts = {header: 0 for header in self.selected_labels}
        
        for data in tqdm(self.train_cache, desc="Computing Train Set Statistics", unit="file"):
            
            # 'labels' is assumed to be a numpy array of shape (num_timestamps, num_label_columns)
            labels = data["labels"]
            
            # Timestamps per label
            label_counts = np.sum(labels, axis=0)                       # label_counts is a 1D array where each element corresponds to a label’s count
            for header, cnt in zip(self.selected_labels, label_counts): # Zip pairs each label with the corresponding count
                instance_counts[header] += cnt
                if cnt > 0:
                    file_counts[header] += 1
        
        self.class_instances = instance_counts
        self.class_f_count_train = file_counts

        total_instances = sum(instance_counts.values())
        normal_instances = instance_counts.get("Normal", 0)
        fault_instances = total_instances - normal_instances
        
        # Calculate imbalance metric
        self.imbalance_metric = (abs(normal_instances - fault_instances) / total_instances * 100) if total_instances > 0 else 0
        
        # Class Weights
        if self.manual_weights is True:
            # Assign the manually defined weights
            self.class_weights = weights_manual
        else:
            # Compute class weights as the inverse of counts
            class_counts_array = np.array([instance_counts[header] for header in self.selected_labels], dtype=float)
            for header, cnt in zip(self.selected_labels, class_counts_array):
                if cnt == 0:
                    raise ValueError(f"Division by zero encountered: class '{header}' has zero instances.")
            self.class_weights = 1. / class_counts_array
            # Normalize class weights
            if self.normalize_weights:
                self.class_weights = self.class_weights / np.sum(self.class_weights)

    def count_class_files(self, cached_list: str) -> None:
        """
        Checks if a given cached Dataset (validation or test) exists and counts the number of files (simulations) per label.

        The results are stored either in self.class_f_count_val or self.class_f_count_test.
        
        Parameters:
            cached_list (str): "val_cache" OR "test_cache" - Name of the cahed list.
        """

        # Confirm Cached List Exists
        cache = getattr(self, cached_list, None)
        if not cache:
            raise  AttributeError(f"Cached attribute '{cached_list}' does not exist.")
        
        # Dict to save file counts per label
        file_counts = {label: 0 for label in self.selected_labels}
        
        # Search for labels in each file
        for data in cache:
            labels = data["labels"]             # 'labels' is a numpy array of shape (num_timestamps, num_label_columns)
            presence = np.any(labels, axis=0)   # Check if any label is present

            # Update Dict
            for label, present in zip(self.selected_labels, presence):
                if present:
                    file_counts[label] += 1
            
        # Select the attribute Dict in which to save file_counts
        if cached_list.find("val") != -1:
            self.class_f_count_val = file_counts    # Validation
        elif cached_list.find("test") != -1:
            self.class_f_count_test = file_counts   # Test
        else:
            raise AttributeError (f"Cached attribute - '{cached_list}' - doesn't have a correspoding dict for file counts per label.")


    def check_labels(self, cached_list: str) -> None:
        """
        Checks if a given cached Dataset exists and contains all labels specified in self.selected_labels.

        The method checks if for each label there is at least one instance (a value equal to 1) in the corresponding column. 
        For efficiency reasons, it only checks for labels it hasn't found yet, i.e.: doesn't iterate over all labels of each of 
        the files present in the cached list.
        
        If a label is not found (i.e. not present in any file's labels), it logs a DEBUG message.

        Parameters:
            cached_list (str): Name of the cahed list.
        """

        # Confirm Cached List Exists
        cache = getattr(self, cached_list, None)
        if not cache:
            raise  AttributeError(f"Cached attribute '{cached_list}' does not exist.")


        num_labels = len(self.selected_labels)
        found_labels = [False] * num_labels

        # Iterate over each cached file and update found flags.
        for data in cache:
            labels_array = data["labels"]  # Assumes 'labels' is present and is a 2D numpy array.
            for i, label in enumerate(self.selected_labels):
                # Only checks for labels that haven't been found yet
                if not found_labels[i] and np.any(labels_array[:, i] == 1):
                    found_labels[i] = True

        # Log missing labels in DEBUG mode.
        for i, label in enumerate(self.selected_labels):
            if not found_labels[i]:
                self.logger.warning(f"⚠️  Label '{label}' not found in {cached_list}.")
        self.logger.info("")

        
    def compute_normalization_stats(self) -> None:
            """
            Computes global normalization statistics (mean, standard deviation, min, and max) 
            using the training set.
            These statistics will be stored in:
            - self.global_mean
            - self.global_std
            - self.global_min
            - self.global_max
            """

            if not getattr(self, 'train_cache', None):
                raise AttributeError("Train cache not available. Please run cache_npy_files() first.")

            total_sum = None
            total_sq_sum = None
            global_min = None
            global_max = None
            total_count = 0

            for data in tqdm(self.train_cache, desc="Computing Normalization Statistics", unit="file"):
                
                # Convert features to float64 for precision in accumulation for stats
                features = data["features"].astype(np.float64)
                if total_sum is None:
                    total_sum = np.sum(features, axis=0)
                    total_sq_sum = np.sum(features**2, axis=0)
                    global_min = np.min(features, axis=0)
                    global_max = np.max(features, axis=0)
                else:
                    total_sum += np.sum(features, axis=0)
                    total_sq_sum += np.sum(features**2, axis=0)
                    global_min = np.minimum(global_min, np.min(features, axis=0))
                    global_max = np.maximum(global_max, np.max(features, axis=0))
                total_count += features.shape[0]
            
            self.global_mean = total_sum / total_count
            variance = total_sq_sum / total_count - self.global_mean**2
            self.global_std = np.sqrt(variance)
            self.global_min = global_min
            self.global_max = global_max
