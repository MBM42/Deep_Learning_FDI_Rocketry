"""
weights.py

Provides manually defined class weights as an np.ndarray.

Author: Miguel Marques
Date: 06-04-2025
"""

import numpy as np

# Manual Cross-Entropy Loss Weights
weights_manual = np.array([
    1.0,  # Normal
    2.0,  # PLB_11_Block
    2.0,  # PLB_11_Leak
    2.0,  # PLB_11_Block_Leak
    2.0,  # PLB_12_Block
    2.0,  # PLB_12_Leak
    2.0,  # PLB_12_Block_Leak
    2.0,  # PLB_21_Block
    2.0,  # PLB_21_Leak
    2.0,  # PLB_21_Block_Leak
    2.0,  # PLB_22_Block
    2.0,  # PLB_22_Leak
    2.0,  # PLB_22_Block_Leak
    2.0,  # Valve_1_Status
    2.0,  # Valve_2_Status
    2.0,  # P_ch_1_freeze
    2.0,  # P_ch_1_bias
    2.0,  # P_ch_1_drift
    2.0,  # T_ch_1_freeze
    2.0,  # T_ch_1_bias
    2.0,  # T_ch_1_drift
    2.0,  # CC_oxy_m_freeze
    2.0,  # CC_oxy_m_bias
    2.0,  # CC_oxy_m_drift
    2.0,  # CC_red_m_freeze
    2.0,  # CC_red_m_bias
    2.0,  # CC_red_m_drift
    2.0,  # PLB_11_Pin_freeze
    2.0,  # PLB_11_Pin_bias
    2.0,  # PLB_11_Pin_drift
    2.0,  # PLB_12_Pout_freeze
    2.0,  # PLB_12_Pout_bias
    2.0,  # PLB_12_Pout_drift
    2.0,  # PLB_21_Pin_freeze
    2.0,  # PLB_21_Pin_bias
    2.0,  # PLB_21_Pin_drift
    2.0,  # PLB_22_Pout_freeze
    2.0,  # PLB_22_Pout_bias
    2.0,  # PLB_22_Pout_drift
    2.0,  # PLB_11_Tin_freeze
    2.0,  # PLB_11_Tin_bias
    2.0,  # PLB_11_Tin_drift
    2.0,  # PLB_12_Tout_freeze
    2.0,  # PLB_12_Tout_bias
    2.0,  # PLB_12_Tout_drift
    2.0,  # PLB_21_Tin_freeze
    2.0,  # PLB_21_Tin_bias
    2.0,  # PLB_21_Tin_drift
    2.0,  # PLB_22_Tout_freeze
    2.0,  # PLB_22_Tout_bias
    2.0,  # PLB_22_Tout_drift
    2.0,  # PLB_11_m_in_freeze
    2.0,  # PLB_11_m_in_bias
    2.0,  # PLB_11_m_in_drift
    2.0,  # PLB_21_m_in_freeze
    2.0,  # PLB_21_m_in_bias
    2.0,  # PLB_21_m_in_drift
    2.0,  # Valve_1_apos_freeze
    2.0,  # Valve_1_apos_bias
    2.0,  # Valve_1_apos_drift
    2.0,  # Valve_2_apos_freeze
    2.0,  # Valve_2_apos_bias
    2.0   # Valve_2_apos_drift
])
