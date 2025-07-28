"""
early_stop.py

This module provides an `EarlyStopping` class that can monitor any scalar
metric (e.g., validation loss, accuracy, F1 score) and halt training when
that metric has stopped improving for a specified number of epochs.

Features:
  - mode='min': consider lower values as improvements (e.g. loss)
  - mode='max': consider higher values as improvements (e.g. accuracy)
  - adjustable `patience` (number of epochs to wait for improvement)
  - adjustable `min_delta` (minimum change to qualify as an improvement)

Author: Miguel Marques
Date: 21-04-2025
"""
import math

class EarlyStopping:
    """
    Params:
        patience (int): number of epochs to wait for improvement
        min_delta (float): minimum change to qualify as an improvement
        mode (str): 'min' lower values are improvements; 
                    'max' consider higher values as improvements
    """
    def __init__(self, patience: int = 8, min_delta: float = 0.0, mode: str = 'min'):
        if mode not in ('min', 'max'):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience    = patience
        self.min_delta   = min_delta
        self.mode        = mode
        self.best        = math.inf if mode == 'min' else -math.inf
        self.counter     = 0
        self.should_stop = False

    def step(self, metric: float):
        improved = (
            (self.mode == 'min' and metric <  self.best - self.min_delta) or
            (self.mode == 'max' and metric >  self.best + self.min_delta)
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
