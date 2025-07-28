"""
debug_data.py

Provides lists of strings (List[str]) to select the features and labels for the debug data.

Author: Miguel Marques
Date: 18-04-2025
"""

import numpy as np

# Features
debug_features = [
    "feature1",
    "feature2",
    "feature3",
    "feature4",
    "feature5",
]

# Labels
debug_labels = [
    "Normal",
    "Fault1",
    "Fault2",
    "Fault3",
    "Fault4",
]

# Weights
debug_weights = np.array([
    1.0,  # Normal
    2.0,  # Fault1
    2.0,  # Fault2
    2.0,  # Fault3
    2.0,  # Fault4
])
