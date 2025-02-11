# WIP
# TODO: Done
from constants import *
import global_vars as gv
import numpy as np
import calculate_flux as cF

def update_residual(s_vector):
    # update flux vector using the currently saved cell properties
    cF.update_flux(s_vector)
    
    for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y):
        gv.R[i, j, 0] = np.sum(gv.F_star[i, j, :, 0])
        gv.R[i, j, 1] = np.sum(gv.F_star[i, j, :, 1])
        gv.R[i, j, 2] = np.sum(gv.F_star[i, j, :, 2])
        gv.R[i, j, 3] = np.sum(gv.F_star[i, j, :, 3])

    # gv.R[:,:,0] = np.sum(gv.F[:,:,:,0], axis=2)
    # gv.R[:,:,1] = np.sum(gv.F[:,:,:,1], axis=2)
    # gv.R[:,:,2] = np.sum(gv.F[:,:,:,2], axis=2)
    # gv.R[:,:,3] = np.sum(gv.F[:,:,:,3], axis=2)

    return None