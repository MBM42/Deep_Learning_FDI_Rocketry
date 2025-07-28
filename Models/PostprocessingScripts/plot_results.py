"""
plot_results.py

Standalone script to plot multiple metrics/results of the trained model.

Allows for an easier and faster approach into re-creation and customization of the plots.

Author: Miguel Marques
Date: 12-04-2025
"""

import os
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from pathlib import Path


def load_plot_data(filename): 
    """
    Loads plot metrics from the .json file.
    """ 
    with open(filename, 'r') as f: 
        data = json.load(f) 
        return data


def plot_loss(data):
    """
    Generates and saves the Loss Plot (training vs. validation loss).
    """ 
    epochs = data['epochs'] 
    train_loss = data['train_loss'] 
    val_loss = data['val_loss']

    # Define custom x-ticks: starting with 1 and then every 5 epochs
    ticks = [1] + list(range(5, len(epochs) + 1, 5))

    fig, ax = plt.subplots()
    ax.plot(epochs, train_loss, '-', color = mcolors.TABLEAU_COLORS['tab:orange'], label="Train")
    ax.plot(epochs, val_loss, '-', color = mcolors.TABLEAU_COLORS['tab:green'], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))         # y-ticks spaced by 0.1
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f')) # y-ticks with 1 decimal places
    ax.set_xticks(ticks)                                            # Custom x-ticks
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))   # x-ticks as integers
    ax.set_xlim(left=0, right=len(epochs) + 1)                      # Setting x-axis limits


def plot_accuracy(data):
    """
    Generates and saves the Accuracy Plot (training and validation accuracy).
    """ 
    epochs = data['epochs'] 
    train_acc = data['train_acc'] 
    val_acc = data['val_acc']

    # Define custom x-ticks: starting with 1 and then every 5 epochs
    ticks = [1] + list(range(5, len(epochs) + 1, 5))

    fig, ax = plt.subplots()
    ax.plot(epochs, train_acc, '-', color = mcolors.TABLEAU_COLORS['tab:orange'], label="Train")
    ax.plot(epochs, val_acc, '-', color = mcolors.TABLEAU_COLORS['tab:green'], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)
    ax.set_xticks(ticks)                                            # Custom x-ticks
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))   # x-ticks as integers
    ax.set_xlim(left=0, right=len(epochs) + 1)                      # Setting x-axis limits


def plot_val_metrics(data): 
    """
    Generates and saves the Validation Metrics Plot (precision, recall, and F1 score).
    """ 
    epochs = data['epochs'] 
    val_precision = data['val_precision'] 
    val_recall = data['val_recall'] 
    val_f1 = data['val_f1']

    # Define custom x-ticks: starting with 1 and then every 5 epochs
    ticks = [1] + list(range(5, len(epochs) + 1, 5))

    fig, ax = plt.subplots()
    ax.plot(epochs, val_precision, '-', color = mcolors.CSS4_COLORS['dodgerblue'], label="Precision")
    ax.plot(epochs, val_recall, '-', color = mcolors.CSS4_COLORS['gold'], label="Recall")
    ax.plot(epochs, val_f1, '-', color = mcolors.CSS4_COLORS['magenta'], label="F1 Score")
    ax.set_xlabel("Epoch")
    ax.set_title("Validation Metrics")
    ax.legend()
    ax.grid(True)
    ax.set_xticks(ticks)                                            # Custom x-ticks
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))   # x-ticks as integers
    ax.set_xlim(left=0, right=len(epochs) + 1)                      # Setting x-axis limits


def main():
    
    # Dir handling
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plot_data_path = Path(current_dir) / "plot_data.json"
    
    # Check if data file exists
    if not plot_data_path.exists(): 
       raise FileNotFoundError("Error: 'plot_data.json' was not found in the current directory.") 

    # Load data
    data = load_plot_data(plot_data_path)

    # Create create the plots
    plot_loss(data)
    plot_accuracy(data)
    plot_val_metrics(data)
    plt.show()

if __name__ == "__main__": 
    main()
