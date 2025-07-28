"""
p_metrics.py 

This module implements the PerformanceMetrics class, which computes performance metrics for a ML classification task.

It provides methods for:
- Computing the accuracy score: accuracy()
- Removing the normal class from the labels and predictions: filter_normal() -> this allows to compute performance metrics solely for the fault classes
- Computing precision, recall, F1 score, MCC, and per-class precision and recall: compute_metrics()

Author: Miguel Marques
Date: 22-03-2025
"""

import numpy as np
from sklearn.metrics import *


class PerformanceMetrics:
    """
    PerformanceMetrics class to compute performance metrics for a ML classification task.

    Params:
        label_headers (list[str]): list of labels for which to compute performance metrics
        name (str): type of instance, e.g.: "Validation"; "Test"

    Extra Attributes:
        name (str): type of instance
        labels (np.array): true labels
        preds (np.array): predicted labels
        class_labels (dict): dictionary mapping class indices to class names. e.g. {0: "Normal", 1: "Valve", ...}
        acc (float): accuracy.
        precision_macro (float): macro-averaged precision.
        recall_macro (float): macro-averaged recall.
        f1_macro (float): macro-averaged F1 score.
        mcc (float): Matthews Correlation Coefficient.
        precision_per_class (dict): precision per class.
        recall_per_class (dict): recall per class.
    """

    def __init__(self, label_headers: list[str], name: str) -> None:
        self.class_labels = {i: label for i, label in enumerate(label_headers)}
        self.name = name
        self.labels = None
        self.preds = None
        self.acc: float = 0.0
        self.precision_macro: float = 0.0
        self.recall_macro: float = 0.0
        self.f1_macro: float = 0.0
        self.mcc: float = 0.0
        self.precision_per_class = {}
        self.recall_per_class = {}


    def epoch_update(self, labels: np.ndarray, preds: np.ndarray) -> None:
        """
        Updates the PerformanceMetrics instance with a new set of labels and predictions from which the metrics will be computed.
        
        Params:
            labels (np.array)
            preds (np.array)
        
        """
        self.labels = labels
        self.preds = preds


    def accuracy(self) -> None:
        """
        Computes the model accuracy.
        """
        self.acc = accuracy_score(self.labels, self.preds)


    def compute_metrics(self) -> None:
        """
        Computes main performance metrics: 
            precision; 
            recall; 
            F1 score; 
            MCC; 
            per-class precision and recall.
        """

        self.precision_macro = precision_score(self.labels, self.preds, average='macro', zero_division=0.0)
        self.recall_macro = recall_score(self.labels, self.preds, average='macro', zero_division=0.0)
        self.f1_macro = f1_score(self.labels, self.preds, average='macro', zero_division=0.0)
        self.mcc = matthews_corrcoef(self.labels, self.preds)

        for cls in self.class_labels.keys():
            TP = np.sum((self.preds == cls) & (self.labels == cls))  # True Positives
            FP = np.sum((self.preds == cls) & (self.labels != cls))  # False Positives
            FN = np.sum((self.preds != cls) & (self.labels == cls))  # False Negatives

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

            self.precision_per_class[cls] = precision
            self.recall_per_class[cls] = recall


    def filter_normal(self, filter_flag: bool = True) -> None:
        """
        Removes the normal class from the labels and predictions if filter_flag = True.
        This allows to compute performance metrics solely for the fault classes.

        Params:
            filter_flag (bool, optional): Default is True
        """

        if filter_flag:
            fault_mask = self.labels != 0
            self.labels, self.preds = self.labels[fault_mask], self.preds[fault_mask]
            self.class_labels.pop(0, None) # Remove normal class
        else:
            return
