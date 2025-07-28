# Labels for available data / features / faults
# TIME: [s]
# Thrust:        Engine Thrust [N]
# P_ch_1:        Combustion chamber Pressure in the first segment [Pa]
# T_ch_1:        Combustion chamber Temperature in the first segment [K]
# CC_oxy_m:      Combustion Combustion Chamber Oxidizer Entering Mass Flow Rate [kg/s]
# CC_red_m:      Combustion Combustion Chamber Reductant Entering Mass Flow Rate [kg/s]
# Inj_oxy_dp:    Oxidizer Injector Pressure Drop [Pa]
# Inj_red_dp:    Reductant Injector Pressure Drop [Pa]
# J11_dp:        Oxidizer Tank Junction Pressure Drop [Pa]
# J12_dp:        Oxidizer Combustion Chamber Junction Pressure Drop [Pa]
# J21_dp:        Reductant Tank Junction Pressure Drop [Pa]
# J22_dp:        Reductant Combustion Chamber Junction Pressure Drop [Pa]
# Ox_Source_P:   Oxidizer Source Pressure [Pa]
# Red_Source_P:  Reductant Source Pressure [Pa]
# Ox_Source_T:   Oxidizer Source Temperature [K]
# Red_Source_T:  Reductant Source Temperature [K]
## Pressures [Pa]
# PLB_11_Pin:    Oxidizer Before Valve, Inlet Pressure
# PLB_11_Pout:   Oxidizer Before Valve, Outlet Pressure
# PLB_12_Pin:    Oxidizer After Valve, Inlet Pressure
# PLB_12_Pout:   Oxidizer After Valve, Outlet Pressure
# PLB_21_Pin:    Reductant Before Valve, Inlet Pressure
# PLB_21_Pout:   Reductant Before Valve, Outlet Pressure
# PLB_22_Pin:    Reductant After Valve, Inlet Pressure
# PLB_22_Pout:   Reductant After Valve, Outlet Pressure
## Temperatures [K]
# PLB_11_Tin:    Oxidizer Before Valve, Inlet Temperature
# PLB_11_Tout:   Oxidizer Before Valve, Outlet Temperature
# PLB_12_Tin:    Oxidizer After Valve, Inlet Temperature
# PLB_12_Tout:   Oxidizer After Valve, Outlet Temperature
# PLB_21_Tin:    Reductant Before Valve, Inlet Temperature
# PLB_21_Tout:   Reductant Before Valve, Outlet Temperature
# PLB_22_Tin:    Reductant After Valve, Inlet Temperature
# PLB_22_Tout:   Reductant After Valve, Outlet Temperature
## Mass flow rates [kg/s]
# PLB_11_m_in:   Oxidizer Before Valve, Inlet Mass Flow
# PLB_12_m_in:   Oxidizer After Valve, Inlet Mass Flow
# PLB_21_m_in:   Reductant Before Valve, Inlet Mass Flow
# PLB_22_m_in:   Reductant After Valve, Inlet Mass Flow
## Valves
# Valve_1_cpos:  Oxidizer Valve Command Position
# Valve_1_apos:  Oxidizer Valve Actual Position
# Valve_1_dp:    Oxidizer Valve Pressure Drop
# Valve_2_cpos:  Reductant Valve Command Position
# Valve_2_apos:  Reductant Valve Actual Position
# Valve_2_dp:    Reductant Valve Pressure Drop
## Faults

features_dict = {
    "HFM_01_1": [
    "TIME", "Thrust", "P_ch_1", "T_ch_1", "CC_oxy_m", "CC_red_m",
    "Inj_oxy_dp", "Inj_red_dp", "J11_dp", "J12_dp", "J21_dp", "J22_dp",
    "Ox_Source_P", "Red_Source_P", "Ox_Source_T", "Red_Source_T",
    "PLB_11_Pin", "PLB_11_Pout", "PLB_12_Pin", "PLB_12_Pout",
    "PLB_21_Pin", "PLB_21_Pout", "PLB_22_Pin", "PLB_22_Pout",
    "PLB_11_Tin", "PLB_11_Tout", "PLB_12_Tin", "PLB_12_Tout",
    "PLB_21_Tin", "PLB_21_Tout", "PLB_22_Tin", "PLB_22_Tout",
    "PLB_11_m_in", "PLB_12_m_in", "PLB_21_m_in", "PLB_22_m_in",
    "Valve_1_cpos", "Valve_1_apos", "Valve_2_cpos", "Valve_2_apos",
    "Valve_1_dp", "Valve_2_dp"
],
    "HFM_01_2": [
    "TIME", "Thrust", "P_ch_1", "T_ch_1", "CC_oxy_m", "CC_red_m",
    "Inj_oxy_dp", "Inj_red_dp", "J11_dp", "J12_dp", "J21_dp", "J22_dp",
    "Ox_Source_P", "Red_Source_P", "Ox_Source_T", "Red_Source_T",
    "PLB_11_Pin", "PLB_11_Pout", "PLB_12_Pin", "PLB_12_Pout",
    "PLB_21_Pin", "PLB_21_Pout", "PLB_22_Pin", "PLB_22_Pout",
    "PLB_11_Tin", "PLB_11_Tout", "PLB_12_Tin", "PLB_12_Tout",
    "PLB_21_Tin", "PLB_21_Tout", "PLB_22_Tin", "PLB_22_Tout",
    "PLB_11_m_in", "PLB_12_m_in", "PLB_21_m_in", "PLB_22_m_in",
    "Valve_1_cpos", "Valve_1_apos", "Valve_2_cpos", "Valve_2_apos",
    "Valve_1_dp", "Valve_2_dp"
],
    "HFM_02_1": [
    "TIME", "Thrust", "P_ch_1", "T_ch_1", "CC_oxy_m", "CC_red_m",
    "Inj_oxy_dp", "Inj_red_dp", "J11_dp", "J12_dp", "J21_dp", "J22_dp", "J23_dp",
    "Ox_Source_P", "Red_Source_P", "Ox_Source_T", "Red_Source_T",
    "PLB_11_Pin", "PLB_11_Pout", "PLB_12_Pin", "PLB_12_Pout",
    "PLB_21_Pin", "PLB_21_Pout", "PLB_22_Pin", "PLB_22_Pout",
    "PLB_11_Tin", "PLB_11_Tout", "PLB_12_Tin", "PLB_12_Tout",
    "PLB_21_Tin", "PLB_21_Tout", "PLB_22_Tin", "PLB_22_Tout",
    "PLB_11_m_in", "PLB_12_m_in", "PLB_21_m_in", "PLB_22_m_in",
    "Valve_1_cpos", "Valve_1_apos", "Valve_2_cpos", "Valve_2_apos",
    "Valve_1_dp", "Valve_2_dp", "Tw_hg_in", "Tw_hg_mid", "Tw_hg_ex",
    "RC_Pin", "RC_Pout", "RC_Tin", "RC_Tout"
]}


fault_components = {
    "HFM_01_2": [
    "PLB_11_Block", "PLB_11_Leak", "PLB_11_Block_Leak", "PLB_12_Block", "PLB_12_Leak", "PLB_12_Block_Leak",  "PLB_21_Block", "PLB_21_Leak",
    "PLB_21_Block_Leak", "PLB_22_Block", "PLB_22_Leak", "PLB_22_Block_Leak", "Valve_1_Status", "Valve_2_Status"
]}

sensor_faults = {
    "HFM_01_2": ["P_ch_1", "T_ch_1", "CC_oxy_m", "CC_red_m", "PLB_11_Pin", "PLB_12_Pout", 
                 "PLB_21_Pin", "PLB_22_Pout", "PLB_11_Tin", "PLB_12_Tout", "PLB_21_Tin",
                 "PLB_22_Tout", "PLB_11_m_in", "PLB_21_m_in",  "Valve_1_apos","Valve_2_apos"
]}

getVar = {
    "HFM_01_1": [
    "CC.Nozzle.Thrust", "CC.Combustor.P[1]", "CC.Combustor.T[1]",
    "CC.Combustor.f_oxy.m", "CC.Combustor.f_red.m", "CC.Inj_oxy.dP_loss", "CC.Inj_red.dP_loss",
    "J11.dP_loss", "J12.dP_loss", "J21.dP_loss", "J22.dP_loss", "Ox_Source.s_pres.signal[1]",
    "Red_Source.s_pres.signal[1]", "Ox_Source.s_temp.signal[1]", "Red_Source.s_temp.signal[1]",
    "PLB_11.Pipe_1.P[1]", "PLB_11.Pipe_4.P[5]", "PLB_12.Pipe_1.P[1]", "PLB_12.Pipe_4.P[5]",
    "PLB_21.Pipe_1.P[1]", "PLB_21.Pipe_4.P[5]", "PLB_22.Pipe_1.P[1]", "PLB_22.Pipe_4.P[5]",
    "PLB_11.Pipe_1.T[1]", "PLB_11.Pipe_4.T[5]", "PLB_12.Pipe_1.T[1]", "PLB_12.Pipe_4.T[5]",
    "PLB_21.Pipe_1.T[1]", "PLB_21.Pipe_4.T[5]", "PLB_22.Pipe_1.T[1]", "PLB_22.Pipe_4.T[5]",
    "PLB_11.Pipe_1.m_cel[1]", "PLB_12.Pipe_1.m_cel[1]", "PLB_21.Pipe_1.m_cel[1]", "PLB_22.Pipe_1.m_cel[1]",
    "Valve_1.pos_com", "Valve_1.pos", "Valve_2.pos_com", "Valve_2.pos",
    "Valve_1.dP_loss", "Valve_2.dP_loss"
],
    "HFM_01_2": [
    "CC.Nozzle.Thrust", "CC.Combustor.P[1]", "CC.Combustor.T[1]",
    "CC.Combustor.f_oxy.m", "CC.Combustor.f_red.m", "CC.Inj_oxy.dP_loss", "CC.Inj_red.dP_loss",
    "J11.dP_loss", "J12.dP_loss", "J21.dP_loss", "J22.dP_loss", "Ox_Source.s_pres.signal[1]",
    "Red_Source.s_pres.signal[1]", "Ox_Source.s_temp.signal[1]", "Red_Source.s_temp.signal[1]",
    "PLB_11.Pipe_1.P[1]", "PLB_11.Pipe_4.P[5]", "PLB_12.Pipe_1.P[1]", "PLB_12.Pipe_4.P[5]",
    "PLB_21.Pipe_1.P[1]", "PLB_21.Pipe_4.P[5]", "PLB_22.Pipe_1.P[1]", "PLB_22.Pipe_4.P[5]",
    "PLB_11.Pipe_1.T[1]", "PLB_11.Pipe_4.T[5]", "PLB_12.Pipe_1.T[1]", "PLB_12.Pipe_4.T[5]",
    "PLB_21.Pipe_1.T[1]", "PLB_21.Pipe_4.T[5]", "PLB_22.Pipe_1.T[1]", "PLB_22.Pipe_4.T[5]",
    "PLB_11.Pipe_1.m_cel[1]", "PLB_12.Pipe_1.m_cel[1]", "PLB_21.Pipe_1.m_cel[1]", "PLB_22.Pipe_1.m_cel[1]",
    "Valve_1.pos_com", "Valve_1.pos", "Valve_2.pos_com", "Valve_2.pos",
    "Valve_1.dP_loss", "Valve_2.dP_loss"
],
    "HFM_02_1": [
    "CC.Nozzle.Thrust", "CC.Combustor.P[01]", "CC.Combustor.T[01]",
    "CC.Combustor.f_oxy.m", "CC.Combustor.f_red.m", "CC.Inj_oxy.dP_loss", "CC.Inj_red.dP_loss",
    "J11.dP_loss", "J12.dP_loss", "J21.dP_loss", "J22.dP_loss", "J23.dP_loss", "Ox_Source.s_pres.signal[1]",
    "Red_Source.s_pres.signal[1]", "Ox_Source.s_temp.signal[1]", "Red_Source.s_temp.signal[1]",
    "PLB_11.Pipe_1.P[1]", "PLB_11.Pipe_4.P[5]", "PLB_12.Pipe_1.P[1]", "PLB_12.Pipe_4.P[5]",
    "PLB_21.Pipe_1.P[1]", "PLB_21.Pipe_4.P[5]", "PLB_22.Pipe_1.P[1]", "PLB_22.Pipe_4.P[5]",
    "PLB_11.Pipe_1.T[1]", "PLB_11.Pipe_4.T[5]", "PLB_12.Pipe_1.T[1]", "PLB_12.Pipe_4.T[5]",
    "PLB_21.Pipe_1.T[1]", "PLB_21.Pipe_4.T[5]", "PLB_22.Pipe_1.T[1]", "PLB_22.Pipe_4.T[5]",
    "PLB_11.Pipe_1.m_cel[1]", "PLB_12.Pipe_1.m_cel[1]", "PLB_21.Pipe_1.m_cel[1]", "PLB_22.Pipe_1.m_cel[1]",
    "Valve_1.pos_com", "Valve_1.pos", "Valve_2.pos_com", "Valve_2.pos",
    "Valve_1.dP_loss", "Valve_2.dP_loss", "RC.tp_ch.Tk[01]", "RC.tp_ch.Tk[11]", "RC.tp_ch.Tk[20]", "RC.Channel.P1", "RC.Channel.Pn",
    "RC.Channel.f2.T", "RC.Channel.f1.T"
]}


