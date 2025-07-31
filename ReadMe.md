# Deep Learning for Fault Detection Identification (FDI) in Rocketry

**Author:** Miguel Marques

This project explores the use of deep learning for fault detection and identification (FDI) for a small hopper vehicle of TUM's chair of Space Mobility and Propulsion. 

The research report is available in the respective folder.

Two neural network models—a feedforward neural network (FNN) and a long short-term memory (LSTM) network—are trained on synthetic time-series data generated using an EcosimPro-based digital twin. The objective is to evaluate whether such models can meet the accuracy, latency, and adaptability requirements of real-time FDI in aerospace systems. 

## How It's Made

**Tech used:** Python, PyTorch, EcosimPro

## EcosimPro Model

Model developed by Saravjit Singh

![Example Image](./Images/EcosimPro_model.png)

## Simulations

### data_gen.py

Produces a CSV file where the faults are labelled with "1" and normal behavior with "0".

To use this script the current OS directory should be the "Simulations" folder. This way the script creates and saves the generated data in the corresponding folders.

## Models

To train any of models the current OS directory should be that of the corresponding model.


## Code Example

```
source myenv/bin/activate (myenv: virtual environment)
```

## Write Later

## Installation
```
pip install PyQt5
import matplotlib
matplotlib.use('TkAgg')
sudo apt-get install python3-tk
```

Explain that plot_fault.py expects a certain profile of simulations
