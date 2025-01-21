#%%
# This module uses a RK-scheme to calculate the next timestep

from global_var import *
import numpy as np
import cell
import calculate_residual as cR

def run_iteration():
    global state_vector
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

    for step in range(4):
        # calculate the zwischenstep value using a RK method
        Y[:,:,0] = state_vector[:,:,0] - dt / cell_area[:,:] * RK_ALPHA[step] * R[:,:,0]
        Y[:,:,1] = state_vector[:,:,1] - dt / cell_area[:,:] * RK_ALPHA[step] * R[:,:,1]
        Y[:,:,2] = state_vector[:,:,2] - dt / cell_area[:,:] * RK_ALPHA[step] * R[:,:,2]
        Y[:,:,3] = state_vector[:,:,3] - dt / cell_area[:,:] * RK_ALPHA[step] * R[:,:,3]

        cell.update_cell_properties(Y)

        cR.update_residual()

        calculate_timestep()

    state_vector[:,:,0] = state_vector[:,:,0] - dt / cell_area[:,:] * R[:,:,0]
    state_vector[:,:,1] = state_vector[:,:,1] - dt / cell_area[:,:] * R[:,:,1]
    state_vector[:,:,2] = state_vector[:,:,2] - dt / cell_area[:,:] * R[:,:,2]
    state_vector[:,:,3] = state_vector[:,:,3] - dt / cell_area[:,:] * R[:,:,3]

    return None

def calculate_timestep():
    global time, dt
    u_mag = np.norm(u[:,:]) 
    umax = max(abs(u_mag + c[:,:]), abs(u_mag - c[:,:]))

    dt = CFL / (umax / cell_dx + umax / cell_dy)

    time += dt

    return dt