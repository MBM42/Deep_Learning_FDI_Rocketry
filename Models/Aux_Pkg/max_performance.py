"""
max_performance.py

This class stores the best overall performance for a given PerformanceMetrics' instance.

If provides methods for:
- Updating performance metrics if the f1 score has improved: update_on_f1()
- Updating performance metrics if precision, recall and f1 are all improved: update_on_all()
- Logging the current maximum performance metrics: print_best()

Author: Miguel Marques
Date: 08-04-2025
"""

import logging
from Aux_Pkg.p_metrics import *

class MaxPerformance:
    """
    Params:
        metrics (PerformanceMetrics): PerformanceMetrics' instance of which to keep track of best perforamnce
    
    Extra Attributes:
        metrics (PerformanceMetrics): PerformanceMetrics' instance
        precision (float): Best precision recorded
        recall (float): Best recall recorded
        f1 (float): Best F1 score recorded
        mcc (float): Best Matthews Correlation Coefficient recorded
    """
    
    def __init__(
        self, 
        metrics: PerformanceMetrics,
    ) -> None:
        self.metrics: PerformanceMetrics = metrics
        self.precision: float = metrics.precision_macro
        self.recall: float = metrics.recall_macro
        self.f1: float = metrics.f1_macro
        self.mcc: float = metrics.mcc


    def update_on_f1(self) -> bool:
        """
        Update stored performance metrics if the current f1 score was improveded.

        Returns:
            bool: True if the metrics were updated, False otherwise.
        """

        if self.metrics.f1_macro > self.f1:
            self.precision = self.metrics.precision_macro
            self.recall = self.metrics.recall_macro
            self.f1 = self.metrics.f1_macro
            self.mcc = self.metrics.mcc
            return True
        return False
    
    def update_on_all(self) -> bool:
        """
        Update stored performance metrics only if the current F1 score, precision, and recall are all improved.

        Returns:
            bool: True if all three metrics were improved and updated, False otherwise.
        """
        if (self.metrics.f1_macro > self.f1 and self.metrics.precision_macro > self.precision and self.metrics.recall_macro > self.recall):
            self.f1 = self.metrics.f1_macro
            self.precision = self.metrics.precision_macro
            self.recall = self.metrics.recall_macro
            self.mcc = self.metrics.mcc
            return True
        return False



    def print_best(self, logger: logging.Logger) -> None:
        """
        Logs the current best perforamnce metrics [INFO].

        Params:
            logger (logging.Logger): Logger object.
        """
        logger.info(f"Best Metrics:")
        logger.info(f"- Precision: {self.precision:.4f}")
        logger.info(f"- Recall: {self.recall:.4f}")
        logger.info(f"- F1 Score: {self.f1:.4f}")
        logger.info(f"- MCC: {self.mcc:.4f}")
