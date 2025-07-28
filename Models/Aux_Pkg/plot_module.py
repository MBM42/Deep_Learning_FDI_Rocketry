"""
plot_module.py

Module to produce plots of the performance metrics for the trained models.

Provides methods for:
- Updating the class attributes with the current epoch's metrics: update()
- Plot overall performance metrics: compute_plots()
- Saving plot data, to allow for reload and recreation and customization of the plots: save_plot_data()

Author: Miguel Marques
Date: 25-03-2025
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use('TkAgg')
from pathlib import Path
import matplotlib.colors as mcolors
import json

from Aux_Pkg.p_metrics import *

class PlotManager:
    """ 
    Parameters:
        model_dir (Path): Model path where the plots will be saved.
        train_metrics (PerfromanceMetrics): PerformanceMetrics' instance for train metrics.
        val_metrics (PerformanceMetrics): PerformanceMetrics' instance for validation metrics.

    Extra Attributes:
        plots_dir (Path): Model path.
        train_metrics (PerfromanceMetrics): PerformanceMetrics' instance for train metrics.
        val_metrics (PerformanceMetrics): PerformanceMetrics' instance for validation metrics.
        train_loss (list[float])
        val_loss (list[float])
        train_acc (list[float])
        val_acc (list[float])
        val_precision (list[float])
        val_recall (list[float])
        val_f1 (list[float])
        val_precision_per_class (list[dict])
        val_recall_per_clss (list[dict])
    """ 
    def __init__(
        self, 
        model_dir: Path,
        train_metrics: PerformanceMetrics,
        val_metrics: PerformanceMetrics,
    ) -> None: 
        self.plots_dir: Path = model_dir / "Plots"
        self.val_metrics: PerformanceMetrics = val_metrics
        self.train_metrics: PerformanceMetrics = train_metrics
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []
        self.train_acc: list[float] = []
        self.val_acc: list[float] = []
        self.val_precision: list[float] = []
        self.val_recall: list[float] = []
        self.val_f1: list[float] = []
        self.val_precision_per_class: list[dict] = []
        self.val_recall_per_class: list[dict] = []


    def update(self, train_loss: float, val_loss: float) -> None:
        """
        Updates the stored lists with the metrics for the current epoch.

        Parameters:
            train_loss (float): Average training loss for the epoch.
            val_loss (float): Average validation loss for the epoch.
        """
        
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(self.train_metrics.acc)
        self.val_acc.append(self.val_metrics.acc)
        self.val_precision.append(self.val_metrics.precision_macro)
        self.val_recall.append(self.val_metrics.recall_macro)
        self.val_f1.append(self.val_metrics.f1_macro)
        self.val_precision_per_class.append(self.val_metrics.precision_per_class.copy())
        self.val_recall_per_class.append(self.val_metrics.recall_per_class.copy())


    def compute_plots(self):
        """
        Computes and saves the following plots:
        - Loss Plot: plots train and validation losses, as a function of the number of epochs.
        - Accuracy plot: plots train and validation accuracy, as a function of the number of epochs.
        - Validation metrics plot: plots precision, recall and f1 score for fault detection (excluding 'Normal' labes), as a function of the
        number of epochs.
        """

        # Generate epochs based on the length of the train_loss list
        epochs = list(range(1, len(self.train_loss) + 1))

        # Ticks starting at 1 and multiples of 5
        ticks = [1] + list(range(5, len(epochs) + 1, 5))

        # Loss plot
        fig, ax = plt.subplots()
        plt.plot(epochs, self.train_loss, '-', color = mcolors.TABLEAU_COLORS['tab:orange'], label="Train")
        plt.plot(epochs, self.val_loss, '-', color = mcolors.TABLEAU_COLORS['tab:green'], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))         # y-ticks spaced by 0.1
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f')) # y-ticks with 1 decimal places
        ax.set_xticks(ticks)                                            # Custom x-ticks
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))   # x-ticks as integers
        ax.set_xlim(left = 0, right = len(epochs) + 1)                  # Setting x-axis limits
        self._save_plot(fig, "LossPlot")

        # Accuracy plot
        fig, ax = plt.subplots()
        plt.plot(epochs, self.train_acc, '-', color = mcolors.TABLEAU_COLORS['tab:orange'], label="Train")
        plt.plot(epochs, self.val_acc, '-', color = mcolors.TABLEAU_COLORS['tab:green'], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        ax.set_xticks(ticks)                                            # Custom x-ticks
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))   # x-ticks as integers
        ax.set_xlim(left = 0, right = len(epochs) + 1)                  # Setting x-axis limits
        self._save_plot(fig, "AccPlot")

        # Validation metrics plot - precision, recall, F1
        fig, ax = plt.subplots()
        plt.plot(epochs, self.val_precision, '-', color = mcolors.CSS4_COLORS['dodgerblue'], label="Precision")
        plt.plot(epochs, self.val_recall, '-', color = mcolors.CSS4_COLORS['gold'], label="Recall")
        plt.plot(epochs, self.val_f1, '-', color = mcolors.CSS4_COLORS['magenta'], label="F1 Score")
        plt.xlabel("Epoch")
        plt.title("Validation Metrics")
        plt.legend()
        plt.grid(True)
        ax.set_xticks(ticks)                                            # Custom x-ticks
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))   # x-ticks as integers
        ax.set_xlim(left = 0, right = len(epochs) + 1)                  # Setting x-axis limits
        self._save_plot(fig, "ValMetrics")


    def _save_plot(self, fig: plt.Figure, filename: str):
            """
            Saves a given figure in PNG and EPS formats.

            Parameters:
                fig (matplotlib.figure.Figure): Plot to save.
                filename (str): File name (without extension) for the saved plots.
            """
            # Ensure target directory exists
            self.plots_dir.mkdir(parents=True, exist_ok=True)

            png_path = self.plots_dir / (filename + ".png")
            eps_path = self.plots_dir / (filename + ".eps")
            fig.savefig(png_path, bbox_inches="tight")
            #fig.savefig(eps_path, format="eps", bbox_inches="tight")
            plt.close(fig)


    def save_plot_data(self, filename="plot_data.json"):
        """
        Compiles the accumulated metrics recorded during training and validation
        across epochs into a dictionary and writes it to a JSON file. 
        
        The data saved includes:
        - Epoch numbers.
        - Training and validation loss values.
        - Training and validation accuracy.
        - Macro-averaged precision, recall, and F1 score for validation.
        - Per-class precision and recall for validation.

        Parameters:
            filename (str): The name of the JSON file to save the plot data, defaulting to "plot_data.json".
        """
        
        data = {
            "epochs": list(range(1, len(self.train_loss) + 1)),
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "val_precision": self.val_precision,
            "val_recall": self.val_recall,
            "val_f1": self.val_f1,
            "val_precision_per_class": self.val_precision_per_class,
            "val_recall_per_class": self.val_recall_per_class
        }
        file_path = self.plots_dir / filename
        with open(file_path, "w") as f:
            json.dump(data, f)
