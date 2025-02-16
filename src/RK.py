"""
This module implements a Runge-Kutta (RK) scheme to perform a single iteration of the simulation.
It updates the state vector through a series of intermediate steps, recalculates residuals, 
and updates cell properties, ensuring that the system progresses according to the RK method.
"""

# Import modules
from constants import *
import global_vars as gv

import numpy as np

import cell
import data_io as io
from calculate_residual import update_residual

# RK-solution buffer
Y = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4], 'd')
# Residual monitoring
R_start = np.zeros([4], 'd')
R_final = np.zeros([4], 'd')

def run_iteration():
    """
    Executes a single iteration of the simulation using the four-stage Runge-Kutta (RK4) method
    to update the conserved variables. This iteration involves computing intermediate states, 
    updating cell properties, recalculating residuals, and monitoring convergence throughout.

    Key Steps:
    1. **Initial Residuals:**   Compute and store the maximum residuals for each conserved variable 
                                before the iteration begins (`R_start`).
    2. **RK4 Integration:**
        - **Time Step Calculation:**            Compute the timestep based on the maximum velocity in the domain.
        - **Substeps:**                         Calculate the intermediate states (Y) for each RK step.
        - **Update Residuals and Properties:**  After each RK substep, update the physical properties of the cells 
                                                and recalculate the residuals based on the updated state.
    3. **Final Update:**            After completing all RK steps, overwrite the conserved variable vector 
                                    with the final computed values.
    4. **Mass Flow Update:**        Update the mass flow across cell boundaries (inlet and outlet) 
                                    after the iteration.
    5. **Convergence Monitoring:**  Store the final residuals (`R_final`) and output them every 10 iterations 
                                    for convergence analysis.

    """
    global Y, R_start, R_final

    # Store the initial maximum residuals for each conserved variable before the iteration begins
    R_start = [gv.R[:, :, k].max() for k in range(4)]

    # Perform the four-step Runge-Kutta time integration
    for step in range(4):
        # Calculate the timestep based on the maximum velocity in the domain
        #_calculate_timestep()
        _calculate_local_timestep()

        # Vectorized computation of the intermediate solution Y using the RK method
        Y[:, :, :] = (gv.state_vector[:, :, :] 
                      - (gv.dt[:, :, np.newaxis] / gv.cell_area[:, :, np.newaxis] * RK_ALPHA[step] * gv.R[:, :, :]))

        # # Compute the intermediate solution Y using the RK method for each cell and conserved variable
        # for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y):
        #     for k in range(4):  # Loop over the four conserved variables
        #         Y[i, j, k] = gv.state_vector[i, j, k] - (
        #             gv.dt / gv.cell_area[i, j] * RK_ALPHA[step] * gv.R[i, j, k]
        #         )

        # Update the physical properties of each cell based on the new intermediate state vector Y
        cell.update_cell_properties(Y)

        # Recalculate residuals based on the updated intermediate conserved variables (Y)
        update_residual(Y)

    # After completing all RK steps, overwrite the state vector with the final computed values
    gv.state_vector[:, :, :] = Y[:, :, :]

    # Update the mass flow across cell boundaries after completing the iteration
    cell.update_in_out_massflow()

    # Store the final maximum residuals to monitor convergence
    R_final = [gv.R[:, :, k].max() for k in range(4)]

    # Print residual information every 10 iterations for convergence monitoring
    if gv.iteration % 10 == 0:
        io.print_iteration_residual(gv.iteration, R_start, R_final)

    return None

def _calculate_timestep():
    """
    Calculates the time step (`dt`) for the simulation based on the maximum velocity 
    in the system and the Courant-Friedrichs-Lewy (CFL) condition. The time step is used 
    to ensure the stability of the numerical method.

    In this function:
    1. The maximum value of the velocity `umax` (the maximum velocity from the sum and 
        difference of `u` and `c`) is calculated.
    2. The maximum value of the vertical velocity `vmax` (the maximum velocity from 
        the sum and difference of `u` and `c` in the vertical direction) is calculated.
    3. The time step `dt` is calculated based on the CFL condition and the maximum velocities 
        in the x and y directions.

    Returns:
    --------
    None 
        (The function updates the global variables `gv.dt` and `gv.time`).
    """
    # Calculate the maximum velocity (umax) and maximum vertical velocity (vmax)
    umax = np.max([np.max(np.abs(gv.u[:, :, 0] + gv.c[:, :])), np.max(np.abs(gv.u[:, :, 0] - gv.c[:, :]))])
    vmax = np.max([np.max(np.abs(gv.u[:, :, 1] + gv.c[:, :])), np.max(np.abs(gv.u[:, :, 1] - gv.c[:, :]))])

    # Calculate the time step (dt) based on the CFL condition
    gv.dt[:,:] = CFL / (umax / gv.cell_dx + vmax / gv.cell_dy.min())

    return None

def _calculate_local_timestep():
    """WIP"""

    dSi = 0.5 * (gv.ndS[:,:,1] + gv.ndS[:,:,3])

    u_x = gv.u[:,:,0] + gv.c[:,:]
    u_y = gv.u[:,:,1] + gv.c[:,:]

    gv.dt[:,:] = CFL * gv.cell_area[:,:] / (np.abs(u_x * dSi[:,:,0]) + np.abs(gv.ndS[:,:,2,0] + u_y * gv.ndS[:,:,2,1]))

    return None

# Select the appropriate function for calculating the time step.
# If USE_LOCAL_TIME_STEP (defined in constants) is True, use the local time step function.
# Otherwise, use the global time step function.
calculate_time_step = _calculate_local_timestep if USE_LOCAL_TIME_STEP else _calculate_timestep