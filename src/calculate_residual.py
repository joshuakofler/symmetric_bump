# TODO: Done

# Import global variables, and modules
import global_vars as gv  # Global variables used across the simulation
import numpy as np

from calculate_flux import update_flux

def update_residual(s_vector):
    """
    Updates the global residual array `gv.R` based on the flux calculations.

    This function updates the flux vector using the current cell properties 
    (via `calculate_flux.update_flux`) and then computes the sum of fluxes 
    along the third axis of the `gv.F_star` array to populate the residual 
    array `gv.R`.

    Parameters:
    -----------
    s_vector : numpy.ndarray
        The state vector containing the current properties of the simulation 
        (e.g., density, velocity, etc.).

    Returns:
    --------
    None
        The function directly updates the global variable `gv.R` in-place.
    """
    # Update the flux vector based on the current state vector
    update_flux(s_vector)

    # Compute the residual using a vectorized operation
    # Sum over the third axis (axis=2) of F_star to populate gv.R.
    gv.R[:] = np.sum(gv.F_star, axis=2)

    # The following is a loop-based implementation for reference, which is less efficient:
    # for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y):
    #     gv.R[i, j, 0] = np.sum(gv.F_star[i, j, :, 0])
    #     gv.R[i, j, 1] = np.sum(gv.F_star[i, j, :, 1])
    #     gv.R[i, j, 2] = np.sum(gv.F_star[i, j, :, 2])
    #     gv.R[i, j, 3] = np.sum(gv.F_star[i, j, :, 3])

    return None