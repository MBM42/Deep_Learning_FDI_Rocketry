import numpy as np
import random
import csv
from feature_fault_dict import *

def apply_sensor_label(model, fault_type, csv_log_file_path):
    # Initializing variables to return
    fault = None
    start_index = None

    # Sensor fault types to append
    suffixes = ["_freeze", "_bias", "_drift"]

    # Create dictionary for the different types of sensor faults
    sensor_faults_type = {
        model: {sensor: [sensor + suffix for suffix in suffixes] for sensor in sensor_faults[model]}
    }

    # Read existing CSV
    with open(csv_log_file_path, mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        existing_headers = rows[0]  # Get the headers from the CSV

    # Get new headers from the sensor_faults_type dictionary
    new_headers = []
    for sensor in sensor_faults_type[model]:
        new_headers.extend(sensor_faults_type[model][sensor])
    final_headers = existing_headers + new_headers

    # Add a 0 (no fault) for each new header in the data rows
    for row in rows[1:]:
        row.extend([0] * len(new_headers))

    # Determine the index of the "Normal" header in final_headers, if it exists.
    normal_index = final_headers.index("Normal") if "Normal" in final_headers else None

    if fault_type == "Sensor_Fault":
        # Pick a random sensor and a random fault type
        sensor = random.choice(sensor_faults[model])
        fault = random.choice(sensor_faults_type[model][sensor])
        print(f"Sensor fault: {fault}.")

        # Find the corresponding headers in final_headers
        sensor_index = final_headers.index(sensor)
        label_index = final_headers.index(fault)
        
        if "_freeze" in fault:
            # Random starting point for the freeze, between 1 and 70% of the simulation
            start_index = random.randint(1, round(len(rows) * 0.7))
            print(f"Start index freeze: {start_index}")
            freeze_value = rows[start_index][sensor_index]
            print(f"Freeze value: {freeze_value}")

            for i in range(start_index, len(rows)):
                rows[i][sensor_index] = (
                    f"{freeze_value:.9e}" if isinstance(freeze_value, float) else str(freeze_value)
                )
                rows[i][label_index] = 1  # Mark the fault in the sensor fault column
                if normal_index is not None:
                    rows[i][normal_index] = 0  # Ensure "Normal" is turned off

        elif "_bias" in fault:
            # Random starting point for the bias, between 1 and 50% of the simulation
            start_index = random.randint(1, round(len(rows) * 0.5))
            print(f"Start index bias: {start_index}")
            col_mean = np.mean([float(row[sensor_index]) for row in rows[1:]])
            print(f"Column mean: {col_mean}")
            bias = np.random.uniform(0.1, 0.15)
            bias = col_mean * bias * np.random.choice([-1, 1])
            for i in range(start_index, len(rows)):
                current_value = float(rows[i][sensor_index])
                biased_value = current_value + bias
                rows[i][sensor_index] = (
                    f"{biased_value:.9e}" if isinstance(biased_value, float) else str(biased_value)
                )
                rows[i][label_index] = 1
                if normal_index is not None:
                    rows[i][normal_index] = 0
                if sensor in ["Valve_1_apos", "Valve_2_apos"]:
                    rows[i][sensor_index] = f"{min(max(float(rows[i][sensor_index]), 0), 1):.9e}"

        elif "_drift" in fault:
            # Random starting point for the drift, between 1 and 50% of the simulation
            start_index = np.random.randint(1, round(len(rows) * 0.5))
            print(f"Start index drift: {start_index}")
            drift_rate = np.random.uniform(0.01, 0.05)
            col_mean = np.mean([float(row[sensor_index]) for row in rows[1:]])
            print(f"Column mean: {col_mean}")
            for i in range(start_index, len(rows)):
                current_value = float(rows[i][sensor_index])
                drift_value = drift_rate * (i - start_index) * col_mean
                drifted_value = current_value + drift_value
                rows[i][sensor_index] = (
                    f"{drifted_value:.9e}" if isinstance(drifted_value, float) else str(drifted_value)
                )
                rows[i][label_index] = 1
                if normal_index is not None:
                    rows[i][normal_index] = 0
                if sensor in ["Valve_1_apos", "Valve_2_apos"]:
                    rows[i][sensor_index] = f"{min(max(float(rows[i][sensor_index]), 0), 1):.9e}"
                    
    # Save the modified CSV
    with open(csv_log_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(final_headers)
        writer.writerows(rows[1:])

    return fault, start_index
