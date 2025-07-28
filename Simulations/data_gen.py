import os
import csv
import sys
import time
import random
import tkinter as tk
from scipy.interpolate import interp1d
import numpy as np
import shutil
from itertools import zip_longest
from datetime import datetime
import matplotlib.pyplot as plt
from TrajectoryCalc_Functions import rk4_up, landing_calculation, plot_results
from feature_fault_dict import *
from sensor_label import *

#-----------------------------------------------------------------------------------------------------
#                                           FUNCTIONS
#-----------------------------------------------------------------------------------------------------

## GUI to configure the simulation's settings
# OUTPUT:
# -tuple: A tuple containing the selected model (str), simulation time (int), number of simulations (int), 
# trajectory print option (bool), thrust & position plot option (bool), data type (str), and selected fault type (str or None).
def gui():
    # Needed to perform actions after the user has selected an option (in this case, clikced the submit button)
    def on_select():
        root.destroy()  # Kill the window after selection

    # Show/Hide fault options based on data type selection
    def update_fault_options():
        if data_var.get() == "Fault":
            fault_frame.pack(pady=10, padx=20, fill='x')  # Show fault selection
            submit_btn.pack_forget()  # Remove submit button from original position
            submit_btn.pack(pady=10)  # Move submit button inside fault_frame
        else:
            fault_frame.pack_forget()  # Hide fault selection
            submit_btn.pack_forget()  # Remove submit button from fault_frame
            submit_btn.pack(pady=10)  # Restore submit button at original position

    # Update simulation time based selections
    def update_sim_time():
        if fault_var.get() == "Block_Leak":
            sim_time_var.set(500)
        else:
            sim_time_var.set(500)

    # Command update for data type selection
    def update_both():
        update_fault_options()
        update_sim_time()      
        
    # Validation function for positive integers
    def validate_positive_integer(value):
        if value == "" or value.isdigit() and int(value) > 0:
            return True
        return False
    
    # Toggle the 'Trajectory Print' button appearance and update the state
    def toggle_trajectory_plots():
        trajectory_var.set(not trajectory_var.get())
        if trajectory_var.get():
            trajectory_btn.config(relief="sunken", text="Trajectory Print ON  ")
        else:
            trajectory_btn.config(relief="raised", text="Trajectory Print OFF")

    # Toggle the 'Thrust & Position Plot' button appearance and update the state
    def toggle_thrust_position():
        thrust_var.set(not thrust_var.get())
        if thrust_var.get():
            thrust_btn.config(relief="sunken", text="Thrust/Position Plots ON  ")
        else:
            thrust_btn.config(relief="raised", text="Thrust/Position Plots OFF")

    # Create main window
    root = tk.Tk()
    root.title("Hopper - DataGen")

    # Label for Model Selection
    label_model = tk.Label(root, text="Model:", font=("Arial", 12), anchor="w")
    label_model.pack(pady=10, padx=20, fill='x')

    # Model options with descriptions
    models = {
        "HFM_01_1": "Hopper Fault Model - no controller",
        "HFM_01_2": "Hopper Fault Model - with controller (Default)",
        "HFM_02_1": "Hopper Fault Model with Regenerative Cooling - no controller",
        "HFM_02_2": "Hopper Fault Model with Regenerative Cooling - with controller"
    }

    # Variable to store selected model
    model_var = tk.StringVar(value="HFM_01_2")

    # Creating radio buttons for each model
    for model, description in models.items():
        rb = tk.Radiobutton(root, text=f"{model}: {description}", variable=model_var, value=model, font=("Arial", 10))
        rb.pack(anchor="w", padx=20, pady=2)

    # Frame for the Simulatuion Time Selection
    sim_time_frame = tk.Frame(root)
    sim_time_frame.pack(pady=10, padx=15, fill='x')

    # Label for Simulation Time Selection
    label_sim_time = tk.Label(sim_time_frame, text="Simulation Time (s): (Recommendation by Default)", font=("Arial", 12), anchor="w")
    label_sim_time.pack(side="left", padx=5)

    # Variable to store simulation time
    sim_time_var = tk.StringVar(value="500")

    # Entry widget for simulation time with validation
    sim_time_entry = tk.Entry(sim_time_frame, textvariable=sim_time_var, font=("Arial", 10), width=7, validate="key", validatecommand=(root.register(validate_positive_integer), "%P"))
    sim_time_entry.pack(side="left", padx=5)

    # Frame for the number of simulations
    sim_n_frame = tk.Frame(root)
    sim_n_frame.pack(pady=10, padx=15, fill='x')

    # Label for the number of simulations
    label_sim_n = tk.Label(sim_n_frame, text="Number of simulations:", font=("Arial", 12), anchor="w")
    label_sim_n.pack(side="left", padx=5)

    # Variable to store the number of simulations
    sim_n_var = tk.StringVar(value="1")

    # Entry widget for simulation number with validation
    sim_n_entry = tk.Entry(sim_n_frame, textvariable=sim_n_var, font=("Arial", 10), width=5, validate="key", validatecommand=(root.register(validate_positive_integer), "%P"))
    sim_n_entry.pack(side="left", padx=5)

    # Frame for plotting options (side-by-side buttons)
    plot_frame = tk.Frame(root)
    plot_frame.pack(pady=10, padx=20)

    # Trajectory Print selection
    trajectory_var = tk.BooleanVar(value=False)  # Default to not selected (False)
    trajectory_btn = tk.Button(plot_frame, text="Trajectory Print OFF", font=("Arial", 12), relief="raised", width=20, command=toggle_trajectory_plots)
    trajectory_btn.pack(side="left", padx=5)  # Side-by-side with padding

    # Thrust & Position Plot button
    thrust_var = tk.BooleanVar(value=False)  # Default to not selected (False)
    thrust_btn = tk.Button(plot_frame, text="Thrust/Position Plots OFF", font=("Arial", 12), relief="raised", width=20, command=toggle_thrust_position)
    thrust_btn.pack(side="left", padx=5)  # Side-by-side with padding


    # Label for Data Type Selection
    label_data_type = tk.Label(root, text="Data Type:", font=("Arial", 12), anchor="w")
    label_data_type.pack(pady=10, padx=20, fill='x')

    # Data Type options
    data_types = {
        "Normal": "Normal",
        "Fault": "Fault"
    }

    # Variable to store selected data type
    data_var = tk.StringVar(value="Normal")

    # Creating radio buttons for each data type (with command to update fault options)
    for data_type, description in data_types.items():
        rb = tk.Radiobutton(root, text=description, variable=data_var, value=data_type, font=("Arial", 10), command=update_both)
        rb.pack(anchor="w", padx=20, pady=2)

    # Frame for Fault Type Selection (Initially Hidden)
    fault_frame = tk.Frame(root)

    label_fault_type = tk.Label(fault_frame, text="Fault Type:", font=("Arial", 12), anchor="w")
    label_fault_type.pack(pady=5, padx=0, fill='x')

    # Fault Type options
    fault_types = {
        "Valve": "Valve Fault",
        "Block": "Pipe Blockage",
        "Block_Leak": "Blockage Followed by Leak",
        "Sensor_Fault": "Sensor Fault"
    }

    # Variable to store selected fault type
    fault_var = tk.StringVar(value="Valve")

    # Creating radio buttons for fault types
    for fault, description in fault_types.items():
        rb = tk.Radiobutton(fault_frame, text=description, variable=fault_var, value=fault, font=("Arial", 10), command=update_sim_time)
        rb.pack(anchor="w", padx=0, pady=2)

    # Initially hide fault selection section
    fault_frame.pack_forget()

    # Submit button (initially at the bottom)
    submit_btn = tk.Button(root, text="Submit", command=on_select, font=("Arial", 12))
    submit_btn.pack(pady=10)

    # Wait for the window to be closed before continuing
    root.wait_window()

    # Return the selected model, selected data, selected fault (if selected), and simulation time
    selected_fault = fault_var.get() if data_var.get() == "Fault" else None
    sim_time = sim_time_var.get() if sim_time_var.get() != "" else 500 # Default to 500 sec if empty
    sim_n = sim_n_var.get() if sim_n_var.get() != "" else 1 # Default to 1 if empty
    return model_var.get(), np.int64(sim_time), np.int64(sim_n), trajectory_var.get(), thrust_var.get(), data_var.get(), selected_fault


## Function "folder_create"
# DESCRIPTION: Checks if the specified folder exists. If it doesn't, creates it.
# INPUT:
# -folder_name(str): The name of the folder to check or create.
# OUTPUT:
# -str: The absolute path of the folder.
def folder_create(folder_name):
    if  not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    # Return the absolute path of the folder
    return os.path.abspath(folder_name)


## Function "folder_exists"
# DESCRIPTION: Checks if the specified folder exists. If it doesn't, the program exits.
# INPUT:
# -folder_name(str): The name of the folder to check.
# OUTPUT:
# -str: The absolute path of the folder if it exists.
def folder_exists(folder_name):
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        return os.path.abspath(folder_name)
    else:
        print(f"Folder '{folder_name}' does not exist. Program exits.")
        sys.exit(1)


## Function "find_compiled_model_path"
# DESCRIPTION: Searches for the target folder where the compiled deck is located within the specified model directory.
# INPUT:
# -decks_path(str): The base path where the model directories are located.
# -model(str): The name of the model to search for.
# -target_folder(str): The name of the target folder to find within the model directory.
# OUTPUT:
# -str or None: The full path to the target folder if found, otherwise None.
def find_compiled_model_path(decks_path, model, target_folder):
    base_search_path = os.path.join(decks_path, model) # Model path
    # Search for the folder
    for root, dirs, files in os.walk(base_search_path):
        if target_folder in dirs:
            return os.path.join(root, target_folder)
    return None


## Function "wait_for_file"
# DESCRIPTION: Waits for the specified file to appear in the directory. Throws an error if the file is not found within the timeout period.
# INPUT:
# -file_path(str): The path of the file to wait for.
# -timeout(int): The maximum time to wait for the file in seconds (default is 10 seconds).
def wait_for_file(file_path, timeout=10):
    start_time = time.monotonic()
    while True:
        if os.path.exists(file_path):
            return
        elif time.monotonic() - start_time > timeout:
            raise FileNotFoundError(f"File '{file_path}' not found within {timeout} seconds.")
        time.sleep(1)  # Wait for 1 second before checking again

#-----------------------------------------------------------------------------------------------------
#                      FUNCTIONS FOR BLOCK/LEAK PROFILE GENERATION
#-----------------------------------------------------------------------------------------------------

## Function "block_profile_func"
# DESCRIPTION: generates a time and respective blockage severity (0: complete blockage; 1: completely unblocked) for
# either the oxidizer or reductant feeding. Due to EcosimPro simulation limitations the blockage for the reductant 
# only goes as low as 0.06. Complete blockage in the reductant side would lead to combustion extinction and no re-ignition.
# INPUT:
# -total_sim_time(float)
# -oxi(bool): True for oxidizer, False for Reductant 
# OUTPUT:
# -time_list(list of floats): a list of time points at which the position of the blockage changes
# -block_list(list of floats): list of corresponding blockage severity at the specified time points
# -block_time(float): the randomly chosen time value at which the blockage starts
# -profile type: 
#  -Severe blockage and recovery (1 and 2)
#  -Severe blockage and partial blockage for the remaining of the simulation (3)
# POSSIBLE COMBINATIONS:
# -oxi: 2
# -profile_type: 3
# -position_s: 4
# -position_p: 4
# -total: 96
def block_profile_func(total_sim_time, oxi):
    if oxi: # Block in oxidizer side
        block_time  = random.uniform(0.1 * total_sim_time, total_sim_time) # Random time value from 10 percent of simulation time to end
        block_s = np.random.choice(np.arange(0, 0.21, 0.05)) # For severe blocking
        block_p = np.random.choice(np.arange(0.4, 0.6, 0.05)) # For partial blocking
    else: # Block in reductant side
        block_time  = random.uniform(0.1 * total_sim_time, total_sim_time) # From 10 percent of simulation time to end
        block_s = np.random.choice(np.arange(0.06, 0.21, 0.05)) # For severe blocking
        block_p = np.random.choice(np.arange(0.4, 0.6, 0.05))  # For partial blocking
    
    profile_type = random.randint(1,3)

    match profile_type:
        case 1: # Severe blockage and recovery
            time_list = [0, block_time, block_time + 0.1, block_time + 0.5, block_time + 0.55, total_sim_time]
            block_list = [1, 1, block_s, block_s, 1, 1]
        case 2: # Severe blockage and recovery
            time_list = [0, block_time, block_time + 0.1, block_time + 1, block_time + 1.05, total_sim_time]
            block_list = [1, 1, block_s, block_s, 1, 1]
        case 3: # Severe blockage and partial blockage for the remaining of the simulation
            time_list = [0, block_time, block_time + 0.1, block_time + 0.2, block_time + 0.25, total_sim_time]
            block_list = [1, 1, block_s, block_s, block_p, block_p]

    return time_list, block_list, block_time, profile_type


## Function "leak_profile_func"
# DESCRIPTION: generates a time and respective leak severity after the onset of a blockage.
# The leak starts within 0.5 seconds after the blokcing occurs.
# INPUT:
# -block_t(float): the start time of the blockage
# -total_sim_time(float): the stop time of the simulation.
# OUTPUT:
# -time_list(list of floats): a list of time points at which the leak occurs
# -leak_list(list of floats): a list of corresponding leak severity at the specified time points.
def leak_profiles_func(block_t, total_sim_time):
    leak_time = random.uniform(block_t, block_t + 0.5)
    leak = np.random.choice(np.arange(0.4, 0.6, 0.05))
    time_list = [0, leak_time, leak_time + 0.05, total_sim_time]
    leak_list = [0, 0, leak, leak]
    return time_list, leak_list


#-----------------------------------------------------------------------------------------------------
#                       GUI INTERFACE FOR MODEL, DATA TYPE AND FAULT SELECTION
#-----------------------------------------------------------------------------------------------------

# Model Selection ("selected_model")
# -Model: HFM_01_1 = Hopper Fault Model with no controllers --> Valves need to be controlled by laws --> for testing purposespip
# -Model: HFM_01_2 = Hopper Fault Model with controllers
# -Model: HFM_02_1 = Hopper Fault Model with Regenerative Cooling and no Controllers
# -Model: HFM_02_1 = Hopper Fault Model with Regenerative Cooling and Controllers


# Simulation Time ("sim_time")
# -Default: 500 seconds
# -If blockage and leak, default recommendation is set to 500 seconds

# Number of Simulations ("simulations")
# -Default: 1

# Print Trajectory ("print_trajectory") if selected, prints the trajectory results to the console

# Plot Thrust & Position ("plot_thrust"): if selected, plots the thrust and position

# Data Type ("selected_data")
# -Normal: Normal
# -Fault: Fault
# -Sensor_Fault": Sensor Fault (Normal data produced for sensor fault pre-processing)

# Fault type ("selected_fault")
# -Valve: Valve
# -Block: Pipe Blockage
# -Block_Leak: Blockage Followed by Leak

model, sim_time, n_simulations, print_trajectory, plot_thrust, data_type, fault_type = gui()
print(f"Model selected: {model}")
print(f"Simulation Time: {sim_time} s")
print(f"Number of Simulations: {n_simulations}")
print(f"Trajectory Plots Selected: {print_trajectory}")
print(f"Thrust & Position Plots Selected: {plot_thrust}")
print(f"Data type selected: {data_type}")
print(f"Fault selected: {fault_type}")

# Note to be e appended to the file name for better organization (Optional)
note = ""

#-----------------------------------------------------------------------------------------------------
#                                   REQUIREMENTS AND DEFINITIONS FOR TRAJECTORY CALCULATIONS
#-----------------------------------------------------------------------------------------------------

## Fixed variables
# System and environmental parameters
g           = 9.81              # Gravitational acceleration [m/s^2]
I_sp        = 200               # Specific Impulse [s] --> Felix in "2D_LQR_rotation_matrix.py"
I_sp_g      = I_sp*g            # Effective exhaust velocity [m/s]
y0_asc      = 0                 # Starting heigt (ascent phase) [m] --> leave 0 
v0_asc      = 0                 # Starting velocity (ascent phase) [m/s] --> leave 0
m_dry       = 150               # Dry mass [kg ] --> Felix in "2D_LQR_rotation_matrix.py"
height      = 3                 # Height of the hopper [m]
radius      = 0.5               # Radius of the hopper [m]
throttle    = 0.4               # Maximum allowed throttle during descent (clipped to comply with the physical limits of the engine)      
# Integration Parameters
dt          = 1/30              # Step size [s] --> Felix in "2D_LQR_rotation_matrix.py" (Values lower than 1/100 the rk4_up gives error: problem at the turning point)
t0          = 0                 # Starting time [s] --> Felix in "2D_LQR_rotation_matrix.py" 
tn          = 100               # Integration time [s] --> Felix in "2D_LQR_rotation_matrix.py"
# Initial Conditions
x0          = 0                 # Starting position x [m] --> Felix in "2D_LQR_rotation_matrix.py"
y0          = 0                 # Starting position y [m] --> Felix in "2D_LQR_rotation_matrix.py"
velx0       = 0                 # Starting velocity x [m/s] --> Felix in "2D_LQR_rotation_matrix.py"
vely0       = 0                 # Starting velocity y [m/s] --> Felix in "2D_LQR_rotation_matrix.py"
velphi0     = 0                 # Starting angular velocity [rad/s] --> Felix in "2D_LQR_rotation_matrix.py"
phi0        = 0 / 180 * np.pi   # Starting pitch angle [rad] --> Felix in "2D_LQR_rotation_matrix.py"
alpha0      = 0                 # Angle between the thrust vector and the hopper's long. axis [rad]

# Array for time steps
time_land   = np.linspace(t0, tn, int((tn-t0)/dt)+1)

# R = R11, R12
#     R21, R22
R11_0        = np.cos(phi0)
R12_0        = np.sin(phi0)
R21_0        = -np.sin(phi0)
R22_0        = np.cos(phi0)

y_init       = np.array([x0, y0, phi0, velx0, vely0, velphi0, R11_0, R12_0, R21_0, R22_0])
y_desired    = np.array([0, 0, 0, 0, 0, 0])

states          = np.zeros((len(y_init), len(time_land)))
controls        = np.zeros((4, len(time_land)))


# Data logging parameters
sampling_freq = 100                    # [Hz] --> If 100 it will load 100 samples/timestamps each second
sampling_interval = 1 / sampling_freq  # [s]


#-----------------------------------------------------------------------------------------------------
#                        REQUIREMENTS AND DEFINITIONS FOR ECOSIM SIMULATION
#-----------------------------------------------------------------------------------------------------

TIME        = 0                 # Initial time [s]
CINT        = sampling_interval # Comunication interval, to save the data [s] --> should be equal to the sampling interval

# Environmental parameters
P_ext       = 100000            # External pressure [Pa]
T_ext       = 300               # External temperature [K]

# System parameters
T_ox        = 110               # Oxidizer temperature [K] --> Felix in "v2.0_07_11_2024_system.json"
T_red       = 293               # Reductant temperature [K] --> Felix in "v2.0_07_11_2024_system.json"
P_ox        = 3200000           # Oxidizer pressure [Pa] --> Felix in "v2.0_07_11_2024_system.json"
P_red       = 4000000           # Reductant presure [Pa] --> Felix in "v2.0_07_11_2024_system.json"
MR_min      = -1                # Minimum mixture ratio
MR_max      = 1e20              # Maximum mixture ratio

# Valve time constants
tau_1       = 0.5               # Oxidizer valve opening time constant [s]
tau_c_1     = 0                 # Oxidizer valve closing time constant [s] --> if set to "0" during the Ecosim it assumes the opening constant value
tau_2       = 0.5               # Reductant valve opening time constant [s]
tau_c_2     = 0                 # Reductant valve closing time constant [s] --> if set to "0" during the Ecosim it assumes the opening constant value


#-----------------------------------------------------------------------------------------------------
#                                TRAJECTORY CALCULATIONS
#-----------------------------------------------------------------------------------------------------

# Data variables used for the deck simulatin in Ecosim
thrust_profiles = []            # List with "n_simulations" arrays containing the thrust profile throughout each simulation
timestamps      = []            # List with "n_simulations" arrays containing the timestamps throughout each simulation
thrust_interp   = []            # List of "n_simulations" interpolation functions created from the thrust/time profiles. Provides continuos data for Ecosim simulation
sim_total_time  = []            # List of "n_simulations" doubles containing the total time of each simulation. Needed to define TSTOP, which defines when Ecosim stops the simulation

# Arrays with random values for the following simulation variables. Provides simulation variability
random_T_asc  = np.random.choice(np.arange(2500, 2900, 5), size=n_simulations)  # T_asc: Thrust limit [N] During the simulation random value between 2.4-2.9 kN (80 possibilities)
random_y_targ = np.random.choice(np.arange(1, 100, 1), size=n_simulations)      # y_targ: Target height [m] During the simulation random value between 1-100m (99 possibilities)
random_m_prop = np.random.choice(np.arange(66, 88, 0.2), size=n_simulations)    # m_prop: Propellant mass [kg] During simulation random value between 66-88kg (110 possibilities)


print("-" * 80 + f"\n{'Starting trajectory calculations'.center(80)}\n" + "-" * 80)
# Creating trajectory data for each simulation
for i in range(n_simulations):
    # Collecting the random simulation variables for the current simulation
    T_asc       = random_T_asc[i]
    y_targ      = random_y_targ[i]
    m_prop      = random_m_prop[i]

    m_wet       = m_dry + m_prop      # Total mass of the hopper [kg]
    T_coast     = T_asc * throttle    # Minimum thrust during coast phase [N]
    m_dot       = T_asc/(I_sp_g )     # Propellant mass flow rate [kg/s]
    m_dot_coast = T_coast / (I_sp_g)  # Propelland mass flow rate during coast phase [N]

    ## Ascent Phase
    # "rk4_up(ascent, pos, vel, time, init_mass, massflow, thrust, grav, timestep, m_dry, target_pos=None)"
    # rk4_asc will store a tuple contining the retur of the rk4_up function: time, mass, vel, tp_array, pos, counter
    # -[0]time: total time elapsed during the simulation
    # -[1]mass: remaining mass at the y_targ
    # -[2]vel: final velocity at y_targ
    # -[3]tp_array: 2D array containing thrust values in the first row and position (height) in the second row
    # -[4]pos: final position (should be the ~ y_targ)
    # -[5]counter: number of iterations
    rk4_asc = rk4_up(True, y0_asc, v0_asc, t0, m_wet, m_dot, T_asc, g, dt, m_dry, y_targ)

    ## Coast phase
    # "rk4_up(ascent, pos, vel, time, init_mass, massflow, thrust, grav, timestep, m_dry, target_pos=None)"
    # rk4_asc will store a tuple contining the return of the rk4_up function: time, mass, vel, tp_array, pos, counter
    # -[0]time: total time elapsed during the simulation
    # -[1]mass: remaining mass at "pos"
    # -[2]vel: final velocity at "pos"
    # -[3]tp_array: 2D array containing thrust values in the first row and position (height) in the second row
    # -[4]pos: final position
    # -[5]counter: number of iterations
    rk4_cst = rk4_up(False, y0_asc, rk4_asc[2], t0, rk4_asc[1], m_dot_coast, T_coast, g, dt, m_dry)

    # Guarantees that there is still fuel available after completing ascent phase
    if rk4_cst[1] > m_dry:
        ## Ascent
        thrust_Pos_Array = np.hstack((rk4_asc[3], rk4_cst[3]))  # 2D Array that has thrust (1) and y-position (2) logged over postion

        t_tot_asc = rk4_asc[0] + rk4_cst[0]  # total ascent time
        y_tot_asc = rk4_asc[4] + rk4_cst[4]  # total ascent height
        
        num_samples = int(((t_tot_asc - t0) / dt) + 1)
        timestamps_ascent = np.linspace(t0, t_tot_asc, num_samples)

        # Making sure that 
        if len(thrust_Pos_Array[0]) != len(timestamps_ascent):   
            timestamps_ascent = timestamps_ascent[:-1]

        ## Landing 
        m_wet_landing = rk4_cst[1]  # Wet mass at start of landing phase
        mass = np.ones(len(time_land)) # Initializing mass array with ones
        mass[0] = m_wet_landing

        y_init = np.array([x0, y_tot_asc, phi0, velx0, vely0, velphi0, R11_0, R12_0, R21_0, R22_0])

        # Landing trajectory calculation
        # landing_profile contains the following variables:
        # -[0]t_landed(float): time of landing, equivalent to landing duration
        # -[1]control_matrix(4xN matrix): 
        #   -[0] thrust(N); 
        #   -[1] tilt angle(rad); 
        #   -[2] saturated thrust(N); 
        #   -[3] saturated tilt angle (rad)
        # -[2]state_matrix(10xN matrix)
        #   -[0] position in x
        #   -[1] position in y (alttitude)
        #   -[2] phi (rad)
        #   -[3] velocity in x
        #   -[4] velocity in y
        #   -[5] angular velocity - phi_dot (rad)
        #   -[6] R11
        #   -[7] R12
        #   -[8] R21
        #   -[9] R22
        # -[3]mass_array_2(1xarray)
        # -[4]counter(int)
        # -[5]tank_empty(boolean): returns True or False
        landing_profile = landing_calculation(mass, states, y_desired, time_land, controls, radius, height, g,
                                              y_init, alpha0, m_dry, m_wet_landing, I_sp_g, dt)

        landing_duration = landing_profile[0]
        control_matrix = landing_profile[1]
        state_matrix = landing_profile[2]
        mass_array = landing_profile[3]
        counter = landing_profile[4]
        tank_empty = landing_profile[5]

        m_touchdown_prop = mass_array[-1] - m_dry  # Remaining prop mass in tanks at touchdown

        timestamps_land = time_land[:counter] + t_tot_asc
        timestamps_total = np.hstack((timestamps_ascent, timestamps_land))

        thrust_Pos_Array_dem = np.hstack((thrust_Pos_Array, [control_matrix[0, :counter], state_matrix[1, :counter]])) # Demanded thurst-position ascent + land concatenation 
        thrust_Pos_Array_sat = np.hstack((thrust_Pos_Array, [control_matrix[2, :counter], state_matrix[1, :counter]])) # Saturated thurst-position ascent + land concatenation 
        
        # Prints for debug below
        #print(f"Timestamps lenght: {len(timestamps_total)}")
        #print(f"Thrust array length: {len(thrust_Pos_Array_sat[0, :])}")
        # Interpolation of the thrust to create continuos data distribution of thrust over time to be fed to EcosimPro
        interp = interp1d(timestamps_total, thrust_Pos_Array_sat[0, :], kind='linear', fill_value="extrapolate")

        # Appeding
        thrust_profiles.append(thrust_Pos_Array_sat[0, :]) # Appending the current trajectory's thrust profile to the list of "n_simulations" thrust profiles
        timestamps.append(timestamps_total) # Appending the current trajectory's timestamps 
        thrust_interp.append(interp) # Appending the current trajectory's interpolation furnction
        sim_total_time.append(timestamps_total[-1]) # Appending the current trajectory's total time

        # Printing trajectory info the the console
        if print_trajectory:
            print(f"Total duration of the ascent phase = {t_tot_asc:.4f} seconds.")
            print(f"Total ascent duration at max thrust = {rk4_asc[0]:.4f} seconds.")
            print(f"Total covered height during ascent = {y_tot_asc:.4f} m.")
            print(f"Remaining propellant mass in tanks = {rk4_asc[1] - m_dry:.4f} kg.")
            print("--------------------------------------------------------------------------------------------------------")
            if not tank_empty:
                print(f"Duration of the landing phase = {landing_duration:.3f} s.")
                print(f"Residual propellant mass in the tanks = {m_touchdown_prop:.3f} kg.")
                print("--------------------------------------------------------------------------------------------------------")
                print(f"Total duration of the simulation = {timestamps_total[-1]:.3f} s.")
            else:
                print(f"Propellant all used up after {timestamps_total[-1]:.3f} seconds during landing. Try different Input. Program exits.")
                sys.exit(1)
        # Ploting trajectory
        if plot_thrust:
            plot_results(timestamps_total, thrust_Pos_Array_dem, thrust_Pos_Array_sat)

    # No fuel left after ascent phase
    else:
        print(f"Propellant all used up during ascent. Try different Input. Program exits.")
        sys.exit(1)

print("-" * 80 + f"\n{'Trajectory calculations finished'.center(80)}\n" + "-" * 80)

#-----------------------------------------------------------------------------------------------------
#                             HANDLING DIRECTORIES AND IMPORTING API FUNCTIONS
#-----------------------------------------------------------------------------------------------------

# Unique timestamp YYYY-MM-DD_HH-MM-SS for the log directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f"{model}_{timestamp}_N{n_simulations}"

# Decks
decks_path = folder_exists("Decks") # Checks if folder exists

# Data folder
data_path = folder_create("Data") # Checks if folder "Data" exists and creates it if it doesn't
if fault_type == None:
    data_path = folder_create(os.path.join(data_path, data_type)) # Checks if folder "Data/Normal" exists and creates it if it doesn't
else:
    data_path = folder_create(os.path.join(data_path, fault_type))
data_path = folder_create(os.path.join(data_path, log_dir))

# Configs folder
configs_path = folder_create("Configs") # Checks if folder "Configs" exists and creates it if it doesn't
if fault_type == None:
    configs_path = folder_create(os.path.join(configs_path, data_type)) # Checks if folder "Configs/Normal" exists and creates it if it doesn't
else:
    configs_path = folder_create(os.path.join(configs_path, fault_type))
configs_path = folder_create(os.path.join(configs_path, log_dir))

# Plots
plots_path = folder_create("Plots")
if fault_type == None:
    plots_path = folder_create(os.path.join(plots_path, data_type)) # Checks if folder "Plots/Normal" exists and creates it if it doesn't
else:
    plots_path = folder_create(os.path.join(plots_path, fault_type))
plots_path = folder_create(os.path.join(plots_path, log_dir))

# Compiled model path retrieval
compiled_model_folder = "win64_gcc64v4_9"
compiled_model_path = find_compiled_model_path(decks_path, model, compiled_model_folder)
# If the model isn't found, exit the program
if compiled_model_path is None:
    print(f"Model: {model} doesn't exist. Program exits.")
    sys.exit(1)

# Copying the .pyc file with metadata to the current working directory
shutil.copy2(os.path.join(compiled_model_path, "deck_python_3_11.pyc"), os.getcwd())
wait_for_file(os.path.join(os.getcwd(), "deck_python_3_11.pyc"), timeout=10) # Wait for the file to be copied

# Changing the current working directory to the compiled model path
print(f"\nModel path: {compiled_model_path}")
os.chdir(compiled_model_path)
print(f"Current working directory: {os.getcwd()}")

sys.path.append(os.getcwd()) # Add the current working directory to the list of paths
from deck_python_3_11 import DeckPython
dll_path = os.path.join(os.getcwd(), 'dkd_win64_gcc64v4_9.dll')
print(f"Dll path: {dll_path}\n")
deck = DeckPython(dll_path, False)
if not deck.ok:
    print("Problem loading deck. Program exits.")
    sys.exit(1)


#-----------------------------------------------------------------------------------------------------
#                               Set Model Parameters
#-----------------------------------------------------------------------------------------------------
# Reset the deck
deck.reset()

# Setting surrounding pipe pressures and temperatures (ambient values)
deck.setD("PLB_11.P_11_12.Amp", P_ext)
deck.setD("PLB_11.T_11_12.Amp", T_ext)
deck.setD("PLB_12.P_11_12.Amp", P_ext)
deck.setD("PLB_12.T_11_12.Amp", T_ext)
deck.setD("PLB_21.P_11_12.Amp", P_ext)
deck.setD("PLB_21.T_11_12.Amp", T_ext)
deck.setD("PLB_22.P_11_12.Amp", P_ext)
deck.setD("PLB_22.T_11_12.Amp", T_ext)
# Setting pressure/temperature for oxidizer/reductant tanks
deck.setD("Press_Sig_Ox.Amp", P_ox)
deck.setD("Press_Sig_Red.Amp", P_red)
deck.setD("Temp_Sig_Ox.Amp", T_ox)
deck.setD("Temp_Sig_Red.Amp", T_red)
# Setting surrounding pressure of the combustion chamber
deck.setD("CC.np_out.P", P_ext)
# Set mixture ratio limits
deck.setD("CC.MR_min", MR_min)
deck.setD("CC.MR_max", MR_max)
# Valves
deck.setD("Valve_1.tao", tau_1)
deck.setD("Valve_1.tao_close", tau_c_1)
deck.setD("Valve_2.tao", tau_2)
deck.setD("Valve_2.tao_close", tau_c_2)

# Additional settings for specific models
match model:
    case "HFM_01_2":
        deck.setD("normalizing_MR.s_in_2.signal[1]", 1.025) # MR ideal value of 1.025 (used for normalization)
        deck.setD("Cntrl_PID_1.vi[1]", 100)
        deck.setD("Cntrl_PID_1_1.vi[1]", 1.025)
    case "HFM_02_1":
        rc = deck.setD("ExternalTemperature.Amp", T_ext)



#-----------------------------------------------------------------------------------------------------
#                    SELECT FEATURES AND FAULTS TO BE LOGGED ACCORDING TO MODEL
#-----------------------------------------------------------------------------------------------------

## Extending the model's feature list with a columen for "Normal"
features_dict[model].append("Normal")

## Resorts to the definitions in "feature_fault_dict.py"
# Extending the model's feature list with the selected faults
for model, faults in fault_components.items():
    # Append only to selected model
    if model in features_dict: 
        features_dict[model].extend([fault for fault in faults if fault not in features_dict[model]]) # Append only unique elements

# List to hold timestamps + thrust + features + status
header_names = [header for header in features_dict[model]]

#-----------------------------------------------------------------------------------------------------
#                                       RUN SIMULATION
#-----------------------------------------------------------------------------------------------------

while True:
    for sim in range(n_simulations):
        print(f"Simulation: {sim + 1} of {n_simulations}")
        print(f"Simulation total time: {sim_total_time[sim]}")

        deck.setD("TIME", TIME) # TIME == 0
        deck.setD("CINT", CINT) # Sampling interval
        deck.setD("TSTOP", sim_total_time[sim]) # Simulation stop time

        # Reset log file and dependencies
        simulation_log = [[] for _ in range(len(header_names))] # Each header has a list
        previous_log_time = 0 # Makes sure logs are perfomed at a set sampling interval

        # Reset simulation timer
        sim_begin_t = time.monotonic()

        # Reset log condition
        log_condition = False

        # Reset conditions to stop simulation
        MR_exceeded   = False # Sim stopped after mix ratio exceeded
        time_exceeded = False # Sim stopped after total time exceeded

        # Initialize wall_time_sim
        wall_time_sim = 0

        # Reset list to store timestamps of exceed MR
        mr_exceeded_times = []

        # Pipe Blocking settings
        # Block profile generation
        block_profiles = [
            block_profile_func(sim_total_time[sim], oxi=True),  # Block profile for oxidizer side before valve
            block_profile_func(sim_total_time[sim], oxi=True),  # Block profile for oxidizer side after valve
            block_profile_func(sim_total_time[sim], oxi=False), # Block profile for reductant side before valve
            block_profile_func(sim_total_time[sim], oxi=False)  # Block profile for reductant side after valve
        ]
        # Block profile interpolation generation
        random_block_1 = interp1d(block_profiles[0][0], block_profiles[0][1], kind='linear', fill_value="extrapolate") # Oxidizer before valve
        random_block_2 = interp1d(block_profiles[1][0], block_profiles[1][1], kind='linear', fill_value="extrapolate") # Oxidizer after valve
        random_block_3 = interp1d(block_profiles[2][0], block_profiles[2][1], kind='linear', fill_value="extrapolate") # Reductant before valve
        random_block_4 = interp1d(block_profiles[3][0], block_profiles[3][1], kind='linear', fill_value="extrapolate") # Reductant after valve
        
        # Randomly choose 1 of 4 profiles for both block / block+leak
        pipe_block = random.randint(1, 4)

        # Pipe Block-Leak settings
        # Leak profile generation --> always after the onset of a block ("block_profiles[:][2]": onset time of block)
        leak_profiles = [
            leak_profiles_func(block_profiles[0][2],sim_total_time[sim]), # Leak profile for oxidizer side before valve
            leak_profiles_func(block_profiles[1][2],sim_total_time[sim]), # Leak profile for oxidizer side after valve
            leak_profiles_func(block_profiles[2][2],sim_total_time[sim]), # Leak profile for reductant side before valve
            leak_profiles_func(block_profiles[3][2],sim_total_time[sim])  # Leak profile for reductant side after valve
        ]
        # Leak profile interpolation generation
        random_leak_1 = interp1d(leak_profiles[0][0], leak_profiles[0][1], kind='linear', fill_value="extrapolate") # Oxidizer before valve
        random_leak_2 = interp1d(leak_profiles[1][0], leak_profiles[1][1], kind='linear', fill_value="extrapolate") # Oxidizer after valve
        random_leak_3 = interp1d(leak_profiles[2][0], leak_profiles[2][1], kind='linear', fill_value="extrapolate") # Reductant before valve
        random_leak_4 = interp1d(leak_profiles[3][0], leak_profiles[3][1], kind='linear', fill_value="extrapolate") # Reductant after valve

        # Set default values to no block no leak
        deck.setD("PLB_11.Block_Valve.s_pos.signal[1]", 1)
        deck.setD("PLB_12.Block_Valve.s_pos.signal[1]", 1)
        deck.setD("PLB_21.Block_Valve.s_pos.signal[1]", 1)
        deck.setD("PLB_22.Block_Valve.s_pos.signal[1]", 1)
        deck.setD("PLB_11.Leak_Valve.s_pos.signal[1]", 0)
        deck.setD("PLB_12.Leak_Valve.s_pos.signal[1]", 0)
        deck.setD("PLB_21.Leak_Valve.s_pos.signal[1]", 0)
        deck.setD("PLB_22.Leak_Valve.s_pos.signal[1]", 0)

        # Valve settings
        # Set default valve time constants
        deck.setD("Valve_1.tau", tau_1)             # Oxidizer opening
        deck.setD("Valve_1.tau_close", tau_c_1)     # Oxidizer closing
        deck.setD("Valve_2.tau", tau_2)             # Reductant opening
        deck.setD("Valve_2.tau_close", tau_c_2)     # Reductant closing
        # Flags to save which valve time constants have already been set
        valve1_tau_set = False
        valve2_tau_set = False
        # Time between 1 and sim_total_time[sim]-5 seconds from where on the valve starts responding slow
        valve_slow_start_time = random.uniform(1, sim_total_time[sim] - 5)
        # Randomly choose a valve for slow valve, can later be used to choose the valve directly
        rand_valve = random.randint(1, 2)
          
        # Flag set when simulation takes too long
        fail_flag = False

        #----------------------------BEGINNING OF ECOSIM SIMULATION-------------------------------
        while not time_exceeded and not MR_exceeded:
            # Get current simulation time
            current_time = deck.getD("TIME")[1]
            #print(current_time)

            # Setting current thrust as normalizing referece
            if model == "HFM_01_2" or model == "HFM_02_2":
                deck.setD("normalizing_target.s_in_2.signal[1]", thrust_interp[sim](current_time))

            ## Pipe block fault
            if fault_type == "Block":
                match pipe_block:
                    case 1:
                        deck.setD("PLB_11.Block_Valve.s_pos.signal[1]", random_block_1(current_time))
                        block_type = block_profiles[0][3]
                    case 2:
                        deck.setD("PLB_12.Block_Valve.s_pos.signal[1]", random_block_2(current_time))
                        block_type = block_profiles[1][3]
                    case 3:
                        deck.setD("PLB_21.Block_Valve.s_pos.signal[1]", random_block_3(current_time))
                        block_type = block_profiles[2][3]
                    case 4:
                        deck.setD("PLB_22.Block_Valve.s_pos.signal[1]", random_block_4(current_time))
                        block_type = block_profiles[3][3]

            ## Pipe Block + Leak fault
            if fault_type == "Block_Leak":
                match pipe_block:
                    case 1:
                        deck.setD("PLB_11.Block_Valve.s_pos.signal[1]", random_block_1(current_time))
                        deck.setD("PLB_11.Leak_Valve.s_pos.signal[1]", random_leak_1(current_time))
                        block_type = block_profiles[0][3]
                    case 2:
                        deck.setD("PLB_12.Block_Valve.s_pos.signal[1]", random_block_2(current_time))
                        deck.setD("PLB_12.Leak_Valve.s_pos.signal[1]", random_leak_2(current_time))
                        block_type = block_profiles[1][3]
                    case 3:
                        deck.setD("PLB_21.Block_Valve.s_pos.signal[1]", random_block_3(current_time))
                        deck.setD("PLB_21.Leak_Valve.s_pos.signal[1]", random_leak_3(current_time))
                        block_type = block_profiles[2][3]
                    case 4:
                        deck.setD("PLB_22.Block_Valve.s_pos.signal[1]", random_block_4(current_time))
                        deck.setD("PLB_22.Leak_Valve.s_pos.signal[1]", random_leak_4(current_time))
                        block_type = block_profiles[3][3]

            ## Slow valve fault generation
            if fault_type == "Valve" and current_time >= valve_slow_start_time:
                if not valve1_tau_set and rand_valve == 1: # Oxidizer
                    valve_tau_slow = random.uniform(3, 5)   # Opening delay
                    #valve1_tau_c_slow = random.uniform(3,5) # Closing delay -> if commented assumes the same tau as opening
                    deck.setD("Valve_1.tau", valve_tau_slow)
                    deck.setD("Valve_1.tau_close", valve_tau_slow)
                    valve1_tau_set = True
                elif not valve2_tau_set and rand_valve == 2: # Reductant
                    valve_tau_slow = random.uniform(3, 5)   # Opening delay
                    #valve2_tau_c_slow = random.uniform(3,5) # Closing delay -> if commented assumes the same tau as opening
                    deck.setD("Valve_2.tau", valve_tau_slow)
                    deck.setD("Valve_2.tau_close", valve_tau_slow)
                    valve2_tau_set = True

            ## Run the simulation
            deck.run()

            # If current_time >= previous_log_time:
            if (current_time-previous_log_time) >= sampling_interval:
                # Cut off the starting transient calculations
                if current_time >= 0.1:
                    # Once the targe thrust has been reached all logs with the set sampling rate will be logged:
                    if (deck.getD("CC.Nozzle.Thrust")[1] >= thrust_interp[sim](current_time)):
                        log_condition = True

                    if log_condition:
                        ## Checking Mixture Ratio
                        mix_ratio = deck.getD("CC.Combustor.f_oxy.m")[1] / deck.getD("CC.Combustor.f_red.m")[1] # Calculate the mixture ratio (MR)
                        #print("Mixture Ratio:", mix_ratio)
                        if mix_ratio >= 1.3 or mix_ratio <= 0.8:
                            mr_exceeded_times.append(current_time)
                            if (current_time - mr_exceeded_times[0]) >= 2:
                                print("Mixture Ratio too high. Engine shut off.")
                                # Stop simulation if MR is exceeded for more than 2 seconds
                                MR_exceeded = True

                        ## Log time
                        simulation_log[0].append(current_time)
                        print(f"Sim log at: {simulation_log[0][-1]} seconds")

                        ## Logging features' data
                        for n in range(0, len(getVar[model])):
                            _, value = deck.getD(getVar[model][n])
                            simulation_log[n+1].append(value)

                        ## Logging status
                        # Initialize a set to keep track of which logs have already been updated during the current iteration
                        updated_logs = set()

                        # Oxidizer valve
                        #valve_1_deviation = abs(deck.getD("Valve_1.pos")[1] - deck.getD("Valve_1.pos_com")[1]) > 0.08 # Checks if difference between the actual and commanded positions of the valve is big (= slow valve)
                        valve_1_deviation = True
                        if fault_type == "Valve" and current_time >= valve_slow_start_time and valve_1_deviation and rand_valve == 1:
                            simulation_log[-2].append(1) # "Valve_1_Status"
                            updated_logs.add(-2) # "Valve_1_Status"

                        # Reductant valve
                        #valve_2_deviation = abs(deck.getD("Valve_2.pos")[1] - deck.getD("Valve_2.pos_com")[1]) > 0.08 # Checks if difference between the actual and commanded positions of the valve is big (= slow valve)
                        valve_2_deviation = True
                        if fault_type == "Valve" and current_time >= valve_slow_start_time and valve_2_deviation and rand_valve == 2:
                            simulation_log[-1].append(1) # "Valve_2_Status"
                            updated_logs.add(-1) # "Valve_2_Status"

                        # Block Oxidizer before valve
                        if deck.getD("PLB_11.Block_Valve.s_pos.signal[1]")[1] != 1 and deck.getD("PLB_11.Leak_Valve.s_pos.signal[1]")[1] == 0:
                            simulation_log[-14].append(1) # "PLB_11_Block"
                            updated_logs.add(-14) # "PLB_11_Block"
                        # Block Oxidizer after valve
                        if deck.getD("PLB_12.Block_Valve.s_pos.signal[1]")[1] != 1 and deck.getD("PLB_12.Leak_Valve.s_pos.signal[1]")[1] == 0:
                            simulation_log[-11].append(1) # "PLB_12_Block"
                            updated_logs.add(-11) # "PLB_12_Block"
                        # Block Reductant before valve
                        if deck.getD("PLB_21.Block_Valve.s_pos.signal[1]")[1] != 1 and deck.getD("PLB_21.Leak_Valve.s_pos.signal[1]")[1] == 0:
                            simulation_log[-8].append(1) # "PLB_21_Block"
                            updated_logs.add(-8) # "PLB_21_Block"
                        # Block Reductant after valve
                        if deck.getD("PLB_22.Block_Valve.s_pos.signal[1]")[1] != 1 and deck.getD("PLB_22.Leak_Valve.s_pos.signal[1]")[1] == 0:
                            simulation_log[-5].append(1) # "PLB_22_Block"
                            updated_logs.add(-5) # "PLB_22_Block"
                        # Leak Oxidizer before valve
                        if deck.getD("PLB_11.Leak_Valve.s_pos.signal[1]")[1] != 0 and deck.getD("PLB_11.Block_Valve.s_pos.signal[1]")[1] == 1:
                            simulation_log[-13].append(1) # "PLB_11_Leak"
                            updated_logs.add(-13) # "PLB_11_Leak"
                        # Leak Oxidizer after valve
                        if deck.getD("PLB_12.Leak_Valve.s_pos.signal[1]")[1] != 0 and deck.getD("PLB_12.Block_Valve.s_pos.signal[1]")[1] == 1:
                            simulation_log[-10].append(1) # "PLB_12_Leak"
                            updated_logs.add(-10) # "PLB_12_Leak"
                        # Leak Reductant before valve
                        if deck.getD("PLB_21.Leak_Valve.s_pos.signal[1]")[1] != 0 and deck.getD("PLB_21.Block_Valve.s_pos.signal[1]")[1] == 1:
                            simulation_log[-7].append(1) # "PLB_21_Leak"
                            updated_logs.add(-7) # "PLB_21_Leak"
                        # Leak Reductant after valve
                        if deck.getD("PLB_22.Leak_Valve.s_pos.signal[1]")[1] != 0 and deck.getD("PLB_22.Block_Valve.s_pos.signal[1]")[1] == 1:
                            simulation_log[-4].append(1) # "PLB_22_Leak"
                            updated_logs.add(-4) # "PLB_22_Leak"
                        
                        # Block + Leak Oxidizer before valve
                        if deck.getD("PLB_11.Leak_Valve.s_pos.signal[1]")[1] != 0 and deck.getD("PLB_11.Block_Valve.s_pos.signal[1]")[1] != 1:
                            simulation_log[-12].append(1) # "PLB_11_Block_Leak"
                            updated_logs.add(-12) # "PLB_11_Block_Leak" 

                        # Block + Leak Oxidizer after valve
                        if deck.getD("PLB_12.Leak_Valve.s_pos.signal[1]")[1] != 0 and deck.getD("PLB_12.Block_Valve.s_pos.signal[1]")[1] != 1:
                            simulation_log[-9].append(1) # "PLB_12_Block_Leak"
                            updated_logs.add(-9) # "PLB_12_Block_Leak" 

                        # Block + Leak Reductant before valve
                        if deck.getD("PLB_21.Leak_Valve.s_pos.signal[1]")[1] != 0 and deck.getD("PLB_21.Block_Valve.s_pos.signal[1]")[1] != 1:
                            simulation_log[-6].append(1) # "PLB_21_Block_Leak"
                            updated_logs.add(-6) # "PLB_21_Block_Leak"                            

                        # Block + Leak Reductant affter valve
                        if deck.getD("PLB_22.Leak_Valve.s_pos.signal[1]")[1] != 0 and deck.getD("PLB_22.Block_Valve.s_pos.signal[1]")[1] != 1:
                            simulation_log[-3].append(1) # "PLB_22_Block_Leak"
                            updated_logs.add(-3) # "PLB_22_Block_Leak"

                        # Append "0" (Normal) to logs that were not updated
                        for i in range(-14, 0):
                            if i not in updated_logs:
                                simulation_log[i].append(0) # No Fault

                        # "Normal" status update
                        if not updated_logs:
                            simulation_log[-15].append(1) # Normal
                        else:
                            simulation_log[-15].append(0) # Fault

                previous_log_time = current_time

            # Stop simulation if end of simulation time is reached
            if current_time >= sim_total_time[sim]:
                time_exceeded = True

            # Stop simulation and continue with next if it takes too long
            sim_timer = time.monotonic()
            wall_time_sim = sim_timer - sim_begin_t
            if wall_time_sim > sim_time:
                print("Simulation taking too long. Simulation FAIL.")
                fail_flag = True
                break
            
        #------------------------------END OF ECOSIM SIMULATION-------------------------------  
        print(f"Wall time of simulation: {wall_time_sim:.3f} seconds")

        ## SAVE SIM LOG AND CONFIG
        #Assign names to the data log file and corresponding configuration file
        log_file_name    = f"Sim_{sim}_T_asc_{random_T_asc[sim]}_H_{random_y_targ[sim]:.0f}_PropM_{random_m_prop[sim]:.1f}{note}"
        config_file_name = f"Sim_{sim}_T_asc_{random_T_asc[sim]}_H_{random_y_targ[sim]:.0f}_PropM_{random_m_prop[sim]:.1f}{note}"

        # Write only successful configurations
        if not fail_flag:
            # Log file
            csv_log_file_path = os.path.join(data_path, log_file_name + ".csv")
            print(f"CSV path: {csv_log_file_path}")
            with open(csv_log_file_path, "w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header_names)
                for row in zip_longest(*simulation_log, fillvalue=""):
                    formatted_row = [f"{value:.9e}" if isinstance(value, float) else str(value) for value in row]
                    csv_writer.writerow(formatted_row)

            # Apply sensor labels
            fault, start_index = apply_sensor_label(model, fault_type, csv_log_file_path)

            # Config file
            config_file_name = os.path.join(configs_path, config_file_name + ".txt")
            with open(config_file_name, 'w') as config_file:
                config_file.write(f"Model Name: {model}\n")
                config_file.write("-------------------------------------------------------------------------------------------------------\n")
                config_file.write("\n")
                # Write configuration inputs in one line
                config_file.write("-------------------------------------Trajectory Calculation Inputs-------------------------------------\n")
                config_file.write("\n")
                config_file.write(f"I_sp: {I_sp} s\nT_asc: {random_T_asc[sim]} N\nTargeted Height: {random_y_targ[sim]}\n"
                                  f"Propellant Mass: {random_m_prop[sim]:.1f} kg\nStep Size: {dt:.2f}\n"
                                  f"Dry Mass: {m_dry} kg\nThrottle: {throttle} % \n")
                config_file.write("\n")
                config_file.write("---------------------------------------Ecosim Calculation Inputs---------------------------------------\n")
                config_file.write("\n")
                config_file.write("Ecosim Calculation Inputs:\n")
                config_file.write(f"Simulated Fault: {fault_type} \nSimulation Time: {sim_total_time[sim]:.1f} s\n"
                                  f"Oxidizer Tank Pressure: {P_ox} Pa \nOxidizer Tank Temperature: {T_ox} K\n"
                                  f"Fuel Tank Pressure: {P_red} Pa \nFuel Tank Temperature: {T_red} K\n")
                
                if fault_type == "Block":
                    config_file.write(f"Faulty Pipe Number: {pipe_block} \n"
                                      f"Block type: {block_type} \n"
                                      f"Time Profile of Blocking: {[round(float(x), 2) for x in block_profiles[pipe_block - 1][0]]} \n"
                                      f"Position Profiles of Blocking: {[round(float(x), 2) for x in block_profiles[pipe_block - 1][1]]} \n")

                if fault_type == "Block_Leak":
                    config_file.write(f"Faulty Pipe Number: {pipe_block} \n"
                                      f"Block type: {block_type} \n"
                                      f"Time Profile of Blocking: {[round(float(x), 2) for x in block_profiles[pipe_block - 1][0]]} \n"
                                      f"Position Profiles of Blocking: {[round(float(x), 2) for x in block_profiles[pipe_block - 1][1]]} \n"
                                      f"Time Profile of Leaking: {[round(float(x), 2) for x in leak_profiles[pipe_block - 1][0]]} \n"
                                      f"Position Profile of Leaking: {[round(float(x), 2) for x in leak_profiles[pipe_block - 1][1]]} \n")

                if fault_type == "Valve":
                    config_file.write(f"Fault in valve: {rand_valve} \n"
                                      f"Valve fault start time: {valve_slow_start_time} s\n"
                                      f"Valve 1 Tau (o): {valve_tau_slow} \n")
                if fault_type == "Sensor_Fault":
                    config_file.write(f"Sensor fault: {fault} \n"
                                      f"Start index: {start_index} \n")
                    config_file.write("\n")
                config_file.write(f"Wall Clock Time: {wall_time_sim:.2f} s\n")
                config_file.write("\n")
                config_file.write("--------------------------------Thrust and Time for Interpolation--------------------------------------\n")
                config_file.write("\n")
                config_file.write(f"Thrust Profile: \n")
                config_file.write(", ".join(f"{value:.5e}" if isinstance(value, float) else str(value) for value in thrust_profiles[sim]) + "\n")
                config_file.write("\n")
                config_file.write(f"Time Steps: \n")
                config_file.write(", ".join(f"{value:.5e}" if isinstance(value, float) else str(value) for value in timestamps[sim]) + "\n")

            ## Plot generation
            plot_file_path = os.path.join(plots_path, log_file_name + ".png")
            fig, axs = plt.subplots(3, 2, figsize=(15, 10))   # 3 rows, 2 columns

            #convert a 2D array of subplots into a 1D array. Easier to iterate over
            axs = axs.flatten() 

            # First plot in the first subplot (axs[0])
            axs[0].plot(simulation_log[0], simulation_log[1], label='Thrust')  # Plot thrust
            axs[0].set_xlabel('Time [s]')
            axs[0].set_ylabel('Thrust [N]')
            axs[0].legend()
            axs[0].grid()

            # Second plot in the second subplot (axs[1])
            axs[1].plot(simulation_log[0], simulation_log[2], label='P_chamber')  # Plot chamber pressure
            axs[1].set_xlabel('Time [s]')
            axs[1].set_ylabel('P_chamber [Pa]')
            axs[1].legend()
            axs[1].grid()

            # Second plot in the second subplot (axs[2])
            axs[2].plot(simulation_log[0], simulation_log[3], label='T_chamber')  # Plot chamber temperature
            axs[2].set_xlabel('Time [s]')
            axs[2].set_ylabel('T_chamber [K]')
            axs[2].legend()
            axs[2].grid()

            # Second plot in the second subplot (axs[3])
            axs[3].plot(simulation_log[0], simulation_log[4], label='m_ox')  # Plot mass flow rate of oxidizer
            axs[3].set_xlabel('Time [s]')
            axs[3].set_ylabel('Massflow [kg/s]')
            axs[3].legend()
            axs[3].grid()

            # Second plot in the second subplot (axs[4])
            axs[4].plot(simulation_log[0], simulation_log[5], label='m_red')  # Plot mass flow rate of reductant
            axs[4].set_xlabel('Time [s]')
            axs[4].set_ylabel('Massflow [kg/s]')
            axs[4].legend()
            axs[4].grid()

            # Second plot in the second subplot (axs[5])
            axs[5].plot(simulation_log[0], simulation_log[6], label='Inj_dp_ox')  # Plot pressure drop oxidizer chamber inlet
            axs[5].plot(simulation_log[0], simulation_log[7], label='Inj_dp_red') # Plot pressure drop reductant chamber inlet
            axs[5].set_xlabel('Time [s]')
            axs[5].set_ylabel('Pressure Drop [Pa]')
            axs[5].legend()
            axs[5].grid()

            # Adjust layout to prevent overlap
            plt.tight_layout()
            plt.savefig(plot_file_path)
            # Show the plots
            #plt.show()
        
        print(f"Simulation: {sim + 1} of {n_simulations} ENDED.")
           
    # This will release the deck and all its output
    deck.release()
    print("Simulation ended")
    if sim == (n_simulations-1):
        break
