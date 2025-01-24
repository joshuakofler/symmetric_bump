#%%
# This module uses a RK-scheme to calculate the next timestep

from sre_parse import State
from constants import *
import global_vars as gv

import numpy as np
import cell
import calculate_residual as cR
import data_io as io

# RK-solution buffer
Y = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4], 'd')
# Residual 
R_start = np.zeros([4], 'd')
R_final = np.zeros([4], 'd')

def run_iteration():
    global Y, R_start, R_final
    # this mehtod uses a rk 4 to calculate the new state_vector
    # between each step the cell properties have to be updated
    # as well as the residuals have to be updated
    # after that one can calculate the maxium velocity and with
    # this one can define the timestep used in the rk

    # calculate timestep
    calculate_timestep()

    # in the first step it isnt needed to update the cell_properties as well as the 
    # resiuduals bc. it was updated in the last step of the previous iteration
    # so that it wouldnt change anything bc the state vector is the same as before

    # at step = 0 Y2 is calculate
    # at setp = 1 Y3 is calculated
    # at step = 2 Y4 is calculated

    R_start[0] = gv.R[:,:,0].max()
    R_start[1] = gv.R[:,:,1].max()
    R_start[2] = gv.R[:,:,2].max()
    R_start[3] = gv.R[:,:,3].max()

    for step in range(3):
        # calculate the zwischenstep value using a RK method
        
        for i,j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y):
            Y[i,j,0] = gv.state_vector[i,j,0] - gv.dt / gv.cell_area[i,j] * RK_ALPHA[step] * gv.R[i,j,0]
            Y[i,j,1] = gv.state_vector[i,j,1] - gv.dt / gv.cell_area[i,j] * RK_ALPHA[step] * gv.R[i,j,1]
            Y[i,j,2] = gv.state_vector[i,j,2] - gv.dt / gv.cell_area[i,j] * RK_ALPHA[step] * gv.R[i,j,2]
            Y[i,j,3] = gv.state_vector[i,j,3] - gv.dt / gv.cell_area[i,j] * RK_ALPHA[step] * gv.R[i,j,3]

        cell.update_cell_properties(Y)

        cR.update_residual(Y)

        calculate_timestep()

    for i,j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y):
        gv.state_vector[i,j,0] = gv.state_vector[i,j,0] - gv.dt / gv.cell_area[i,j] * RK_ALPHA[3] * gv.R[i,j,0]
        gv.state_vector[i,j,1] = gv.state_vector[i,j,1] - gv.dt / gv.cell_area[i,j] * RK_ALPHA[3] * gv.R[i,j,1]
        gv.state_vector[i,j,2] = gv.state_vector[i,j,2] - gv.dt / gv.cell_area[i,j] * RK_ALPHA[3] * gv.R[i,j,2]
        gv.state_vector[i,j,3] = gv.state_vector[i,j,3] - gv.dt / gv.cell_area[i,j] * RK_ALPHA[3] * gv.R[i,j,3]
        
    cell.update_cell_properties(gv.state_vector)

    cell.update_in_out_massflow()

    cR.update_residual(gv.state_vector)

    R_final[0] = gv.R[:,:,0].max()
    R_final[1] = gv.R[:,:,1].max()
    R_final[2] = gv.R[:,:,2].max()
    R_final[3] = gv.R[:,:,3].max()

    if gv.iteration % 10 == 0:
        io.print_iteration_residual(gv.iteration, R_start, R_final)

    return None

def calculate_timestep():
    
    umax = np.max(np.maximum(np.abs(gv.u[:, :, 0].max() + gv.c[:,:]),
                      np.abs(gv.u[:, :, 0].max() - gv.c[:,:])))
    
    vmax = np.max(np.maximum(np.abs(gv.u[:, :, 1].max() + gv.c[:,:]),
                      np.abs(gv.u[:, :, 1].max() - gv.c[:,:])))
        
    # # Compute u_mag
    # u_mag = np.sqrt(gv.u[:, :, 0]**2 + gv.u[:, :, 1]**2)

    # # Calculate umax considering both (u + c) and (u - c)
    # umax = np.max(np.maximum(np.abs(u_mag + gv.c[:, :]), np.abs(u_mag - gv.c[:, :])))
    
    # Update time step
    gv.dt = CFL / (umax / gv.cell_dx + vmax / gv.cell_dy.min())

    # Update time
    gv.time += gv.dt

    return None
