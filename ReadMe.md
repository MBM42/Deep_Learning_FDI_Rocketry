# Deep Learning for FDI in Rocketry

**Author:** Miguel Marques  
**Release Date:** June 2025

(Work in progress)

## 1. About The Project

This project explores the use of deep learning for fault detection and identification (FDI) for a small hopper vehicle of TUM's chair of Space Mobility and Propulsion.

This research was developed under the scope of Master's Thesis at the Technical University of Munich (TUM). The report is available [here](./Report/Msc_Thesis_Miguel_FDI_Final.pdf).

The project encompasses two main Python-based frameworks: 
- **Simulations:** Synthetic time-series data generation using an EcosimPro digital twin.
- **Models:** For model training and performance evaluation. Using an FNN and an LSTM.

Two neural network models, a feedforward neural network (FNN) and a long short-term memory (LSTM) are trained on the generated synthetic. The FNN serves as a baseline model, whereas the LSTM was chosen for its recurrent architecture and gatet memory mechanisms, allowing it to better capture dependencies in time-series data.
 
The complete workflow is depicted in the figure below:

<div align="center">
  <img src="./Images/workflow.drawio.jpg" alt="Workflow" width="40%">
</div>

## 2. Goals

- **1. Validate the suitability of deep learning models for fault identification in rocketry:** Up until this research endeavor, machine learning applications in the context of rocketry were restricted to plain fault detection.
- **2. Achieve high fault identification accuracy with low latency:** With concrete performance goals of an **F1 score > 0.9** and **Identification Delay < 50 ms**.
- **3. Confirm the expected better performance of the LSTM model**
- **4. Build a scalable and adaptable framework:** So that upon new iterations of the hopper the proposed 

## 3. Performance and Goal Assessment


## 4. Built With

- ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)  
- ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)  
- ![EcosimPro](https://img.shields.io/badge/EcosimPro-blue?style=for-the-badge)

## 5. Simulations

### 5.1 EcosimPro Model

Model developed by Saravjit Singh

![EcosimPro](./Images/EcosimPro_model.png)

### 5.2 data_gen.py

Produces a CSV file where the faults are labelled with "1" and normal behavior with "0".

To use this script the current OS directory should be the "Simulations" folder. This way the script creates and saves the generated data in the corresponding folders.


## 6. Models

## 7. Future Work

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
