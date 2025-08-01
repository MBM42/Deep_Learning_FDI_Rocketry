"""
features.py

Provides lists of strings (List[str]) to select the features with which to train the model.

Author: Miguel Marques
Date: 30-03-2025
"""

# Total Available Features
features_total = [ 
    "Thrust",
    "P_ch_1",
    "T_ch_1",
    "CC_oxy_m",
    "CC_red_m",
    "Inj_oxy_dp",
    "Inj_red_dp",
    "J11_dp",
    "J12_dp",
    "J21_dp",
    "J22_dp",
    "Ox_Source_P",
    "Red_Source_P",
    "Ox_Source_T",
    "Red_Source_T",
    "PLB_11_Pin",
    "PLB_11_Pout",
    "PLB_12_Pin",
    "PLB_12_Pout",
    "PLB_21_Pin",
    "PLB_21_Pout",
    "PLB_22_Pin",
    "PLB_22_Pout",
    "PLB_11_Tin",
    "PLB_11_Tout",
    "PLB_12_Tin",
    "PLB_12_Tout",
    "PLB_21_Tin",
    "PLB_21_Tout",
    "PLB_22_Tin",
    "PLB_22_Tout",
    "PLB_11_m_in",
    "PLB_12_m_in",
    "PLB_21_m_in",
    "PLB_22_m_in",
    "Valve_1_cpos",
    "Valve_1_apos",
    "Valve_2_cpos",
    "Valve_2_apos",
    "Valve_1_dp",
    "Valve_2_dp"
]
# Current Selected Features
features_selected = [
    "Thrust",
    "P_ch_1",
    "T_ch_1",
    "CC_oxy_m",
    "CC_red_m",
    "Inj_oxy_dp",
    "Inj_red_dp",
    "J11_dp",
    "J12_dp",
    "J21_dp",
    "J22_dp",
    "PLB_11_Pin",
    "PLB_11_Pout",
    "PLB_12_Pin",
    "PLB_12_Pout",
    "PLB_21_Pin",
    "PLB_21_Pout",
    "PLB_22_Pin",
    "PLB_22_Pout",
    "PLB_11_Tin",
    "PLB_11_Tout",
    "PLB_12_Tin",
    "PLB_12_Tout",
    "PLB_21_Tin",
    "PLB_21_Tout",
    "PLB_22_Tin",
    "PLB_22_Tout",
    "PLB_11_m_in",
    "PLB_12_m_in",
    "PLB_21_m_in",
    "PLB_22_m_in",
    "Valve_1_cpos",
    "Valve_1_apos",
    "Valve_2_cpos",
    "Valve_2_apos",
    "Valve_1_dp",
    "Valve_2_dp"
]
# Reduced Features Proposal
features_reduced = [
    "Thrust",
    "P_ch_1",
    "T_ch_1",
    "CC_oxy_m",
    "CC_red_m",
    "J11_dp",
    "J12_dp",
    "J21_dp",
    "J22_dp",
    "PLB_11_Pin",
    "PLB_12_Pout",
    "PLB_21_Pin",
    "PLB_22_Pout", 
    "PLB_11_Tin",
    "PLB_12_Tout",
    "PLB_21_Tin",
    "PLB_22_Tout",
    "PLB_11_m_in",
    "PLB_21_m_in", 
    "Valve_1_apos",
    "Valve_2_apos", 
    "Valve_1_cpos",
    "Valve_2_cpos"
]
