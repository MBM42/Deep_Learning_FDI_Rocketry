import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import control as ct
from math import factorial as fak


def rk4_up(ascent, pos, vel, time, init_mass, massflow, thrust, grav, timestep, m_dry, target_pos=None):
    tp_array = None
    counter =0
    def rk4_step(pos, vel, mass, thrust1):
        """Perform a single RK4 step."""
        a = (thrust1 / mass) - grav
        k1_v = a
        k1_h = vel

        k2_v = (thrust1 / (mass - massflow * 0.5 * timestep)) - grav
        k2_h = vel + 0.5 * k1_v * timestep

        k3_v = (thrust1 / (mass - massflow * 0.5 * timestep)) - grav
        k3_h = vel + 0.5 * k2_v * timestep

        k4_v = (thrust1 / (mass - massflow * timestep)) - grav
        k4_h = vel + k3_v * timestep

        # Update velocity and position using RK4
        vel += (timestep / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        pos += (timestep / 6) * (k1_h + 2 * k2_h + 2 * k3_h + k4_h)

        return vel, pos

    T_asc = thrust
    T_cst = 0.4*thrust
    count = 1

    while True:
        # Calculate current mass
        mass = init_mass - massflow * time
        if mass <= m_dry:
            return time, mass, vel, tp_array

#-----------------------------------------------------------------------------
        if ascent and pos > 0.8 * target_pos:
            if count ==1:
                x = pos
            # Smooth transition using y = mx + c
            slope = -(T_asc - T_cst) / (target_pos-x)

            thrust = slope * (pos - 0.8 * target_pos) + T_asc
            thrust = max(T_cst, thrust)  # Ensure thrust does not go below T_cst
            count +=1
# -----------------------------------------------------------------------------
        if ascent and pos < 0.5*target_pos:
            thrust = T_asc



        # Perform RK4 step
        vel, pos = rk4_step(pos, vel, mass, thrust)

        # Log thrust and position
        array_app_1 = np.array([[thrust], [pos]])
        if tp_array is None:
            tp_array = array_app_1
        else:
            tp_array = np.hstack((tp_array, array_app_1))

        # Increment time
        time += timestep

        # Ascent phase: Check target position
        counter +=1
        if ascent and pos >= target_pos:
            return time, mass, vel, tp_array, pos, counter

        # Coast phase: Check if velocity reaches zero
        if not ascent and vel <= 0:
            return time, mass, vel, tp_array, pos, counter




def rk4_e(f, y, h, t, *args):  # runge kutta 4th order explicit
    # runge kutte 4th order explicit
    tk_05 = t + 0.5 * h
    yk_025 = y + 0.5 * h * f(t, y, *args)
    yk_05 = y + 0.5 * h * f(tk_05, yk_025, *args)
    yk_075 = y + h * f(tk_05, yk_05, *args)

    return y + h / 6 * (f(t, y, *args) + 2 * f(tk_05, yk_025, *args) + 2 * f(tk_05, yk_05, *args) + f(t + h, yk_075, *args))


def ground_truth(t, y, *args):
    '''
    no aerodynamic forces considered up to now --> Brunos Tool??
    '''
    g = args[0]
    m = args[1]
    Theta = args[2]
    height = args[3]

    T = args[4]
    alpha = args[5]

    phi = y[2]

    vel_x = y[3]
    vel_y = y[4]
    vel_phi = y[5]

    pos_x_dot = vel_x
    pos_y_dot = vel_y
    phi_dot = vel_phi

    R11 = y[6]
    R12 = y[7]
    R21 = y[8]
    R22 = y[9]

    dist = height / 2 * np.sin(alpha)  # lever of thrust vector to CoG

    vel_x_dot = -1 / m * T * np.sin(alpha + phi)
    vel_y_dot = 1 / m * (-g * m + T * np.cos(alpha + phi))

    vel_phi_dot = -1 / Theta * T * dist

    R11_dot = vel_phi * R12
    R12_dot = -vel_phi * R11
    R21_dot = vel_phi * R22
    R22_dot = -vel_phi * R21

    return np.array([pos_x_dot, pos_y_dot, phi_dot, vel_x_dot, vel_y_dot, vel_phi_dot, R11_dot, R12_dot, R21_dot, R22_dot])

# transformation body frame to inertial frame

def control_saturation(T_lim_min, T_lim_max, T_value, alpha_lim, alpha_val):
    if T_value > T_lim_max:
        T_return = T_lim_max
    elif T_value < T_lim_min:
        T_return = T_lim_min
    else:
        T_return = T_value

    if abs(alpha_val) > alpha_lim:
        alpha_return = np.sign(alpha_val) * alpha_lim
    else:
        alpha_return = alpha_val

    return T_return, alpha_return


def obtain_discrete_dyn(A, B, T, nu=50):
    def obtain_S(A, nu, T):
        S = 0
        for ii in range(nu):
            S += np.linalg.matrix_power(A, ii) * ((T ** ii) / fak(ii + 1))
        S = S * T
        return S

    S = obtain_S(A, nu, T)
    identity = np.eye(len(A))
    Ad = identity + S @ A
    Bd = S @ B
    return Ad, Bd

# linearized system dynamics with constant mass and no aerodynamic forces

def get_lqr(T0, m, height, Theta, h):
    A = np.array(
        [[0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1],
         [0, 0, -T0 / m, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]
    )

    B = np.array(
        [[0, 0],
         [0, 0],
         [0, 0],
         [0, -T0 / m],
         [1 / m, 0],
         [0, -height / 2 * T0 / Theta]]
    )

    Q = np.diag([1, 5, 1, 100, 1, 100])

    R = np.diag([1e-3, 10000])

    # K (2D array (or matrix)) – State feedback gains
    # S (2D array (or matrix)) – Solution to Riccati equation
    # E (1D array) – Eigenvalues of the closed loop system
    K, S, E = ct.lqr(A, B, Q, R)

    Ad, Bd = obtain_discrete_dyn(A, B, h)

    Qd = Q
    Rd = R

    Kd, Sd, Ed = ct.dlqr(Ad, Bd, Qd, Rd)

    return Kd




def landing_calculation(mass_array,  state_array, array_desired, time_array, control_array,
                        radius, height, grav, y, alpha0, m_dry, m_wet, I_sp_g, h):
    counter = 1
    pos_no_zero = True
    mass_array_2 = []
    tank_empty = False

    for ii in range(len(time_array)): #-1

        m = mass_array[ii]
        Theta = m / 4 * (radius ** 2 + height ** 2 / 3)

        T0 = m * grav

        Kd = get_lqr(T0, m, height, Theta, h)

        state_array[:, ii] = y

        control_input = -Kd @ (y[:6] - array_desired)
        T = control_input[0]  # actually Delta T
        alpha = control_input[1]  # actually Delta alpha
        control_array[0, ii] = T + T0
        control_array[1, ii] = alpha + alpha0
        T_lim, alpha_lim = control_saturation(T_lim_min=0.88 * m_dry * grav, T_lim_max=1.08 * m_wet * grav, T_value=T + T0,
                                              alpha_lim=10 / 180 * np.pi, alpha_val=alpha + alpha0)

        # 9. Failure recovery (PhD Felix)
        if time_array[ii] >= 25 and time_array[ii] < 27:
            T_lim = 0.88 * m_dry * grav

        control_array[2, ii] = T_lim
        control_array[3, ii] = alpha_lim

        t = time_array[ii]
        T_lim *= 1
        args = [grav, m, Theta, height, T_lim, alpha_lim]
        y = rk4_e(ground_truth, y, h, t, *args)

        m_dot_land = T_lim / I_sp_g

        mass_array[ii + 1] = mass_array[ii] - m_dot_land * h
        #print("Current Mass=", mass_array[ii])
        mass_array_2 = np.hstack((mass_array_2, mass_array[ii]))


        if mass_array[ii] <= m_dry:
            #print("Propellant all used up")
            tank_empty = True
            t_landed = time_array[ii]
            return t_landed, control_array, state_array, mass_array_2, counter, tank_empty

        if state_array[1, ii] <= 0.0 and pos_no_zero:
            pos_no_zero = False
            #print(f"Zero Position reached 1st time at {time_array[ii]}")

        elif state_array[1, ii] >= 0.0 and not pos_no_zero:
            t_landed = time_array[ii]
            return t_landed, control_array, state_array, mass_array_2, counter, tank_empty
        counter += 1
    return None, control_array, state_array, mass_array_2, counter, tank_empty




def plot_results(t_tot, tp_array_dem, tp_array_sat, ):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # First plot in the first subplot (axs[0])
    axs[0].plot(t_tot[:-1], tp_array_dem[0, :-1], label='Demanded')
    axs[0].plot(t_tot[:-1], tp_array_sat[0, :-1], label='Saturated')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Thrust [N]')
    axs[0].legend()
    axs[0].grid()

    # Second plot in the second subplot (axs[1])
    axs[1].plot(t_tot[:-1], tp_array_dem[1, :-1], label='Demanded')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Position y [m]')
    axs[1].legend()
    axs[1].grid()

    # Adjust the layout to avoid overlapping labels
    plt.tight_layout()

    # Show the plots
    plt.show()
