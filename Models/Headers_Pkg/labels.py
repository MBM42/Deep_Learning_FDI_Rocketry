"""
labels.py

Provides lists of strings (List[str]) to select the fault labels with which to train the model.

Author: Miguel Marques
Date: 01-04-2025
"""

# System Labels + Sensor Labels
total_labels = [
    "Normal",
    "PLB_11_Block",	
    "PLB_11_Leak",
	"PLB_11_Block_Leak",	
    "PLB_12_Block",	
    "PLB_12_Leak",
	"PLB_12_Block_Leak",
    "PLB_21_Block",
    "PLB_21_Leak",
    "PLB_21_Block_Leak",
	"PLB_22_Block",
    "PLB_22_Leak",
	"PLB_22_Block_Leak",
	"Valve_1_Status",
	"Valve_2_Status",
	"P_ch_1_freeze",
	"P_ch_1_bias",
	"P_ch_1_drift",
	"T_ch_1_freeze",
	"T_ch_1_bias",
    "T_ch_1_drift",
	"CC_oxy_m_freeze",
	"CC_oxy_m_bias",
	"CC_oxy_m_drift",
	"CC_red_m_freeze",
	"CC_red_m_bias",
	"CC_red_m_drift",
	"PLB_11_Pin_freeze",
	"PLB_11_Pin_bias",
	"PLB_11_Pin_drift",
	"PLB_12_Pout_freeze",
	"PLB_12_Pout_bias",
	"PLB_12_Pout_drift",
	"PLB_21_Pin_freeze",
	"PLB_21_Pin_bias",
	"PLB_21_Pin_drift",
	"PLB_22_Pout_freeze",	
    "PLB_22_Pout_bias",
    "PLB_22_Pout_drift",
	"PLB_11_Tin_freeze",
    "PLB_11_Tin_bias",
    "PLB_11_Tin_drift",
    "PLB_12_Tout_freeze",
    "PLB_12_Tout_bias",
    "PLB_12_Tout_drift",
    "PLB_21_Tin_freeze",
    "PLB_21_Tin_bias",
    "PLB_21_Tin_drift",
    "PLB_22_Tout_freeze",
    "PLB_22_Tout_bias",
    "PLB_22_Tout_drift",
    "PLB_11_m_in_freeze",
    "PLB_11_m_in_bias",	
    "PLB_11_m_in_drift",
    "PLB_21_m_in_freeze",
    "PLB_21_m_in_bias",
    "PLB_21_m_in_drift",
    "Valve_1_apos_freeze",
    "Valve_1_apos_bias",
    "Valve_1_apos_drift",
    "Valve_2_apos_freeze",
    "Valve_2_apos_bias",
    "Valve_2_apos_drift"
]

# System Labels
system_labels = [
    "Normal",
    "PLB_11_Block",	
    "PLB_11_Leak",
	"PLB_11_Block_Leak",	
    "PLB_12_Block",	
    "PLB_12_Leak",
	"PLB_12_Block_Leak",
    "PLB_21_Block",
    "PLB_21_Leak",
    "PLB_21_Block_Leak",
	"PLB_22_Block",
    "PLB_22_Leak",
	"PLB_22_Block_Leak",
	"Valve_1_Status",
	"Valve_2_Status"
]





