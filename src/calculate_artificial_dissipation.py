# TODO 
# (?) Calculate artificial dissipation at the boundaries
#     - currently set to zero at the boundaries 
from constants import *
import global_vars as gv
import numpy as np

a_d_coefficient_gamma = np.zeros([NUM_CELLS_X, NUM_CELLS_Y, 4])

def update_artificial_dissipation():
    global a_d_coefficient_gamma
    # east/west
    for i, j in np.ndindex(NUM_CELLS_X-1, NUM_CELLS_Y):
        #artificial_dissipation[i,j,1] = a_d_coefficient_gamma[i,j,1] * (state_vector[i+1,j] - state_vector[i, j])
        # If artificial_dissipation is multi-component
        gv.artificial_dissipation[i, j, 1, :] = a_d_coefficient_gamma[i, j, 1] * (gv.state_vector[i+1, j, :] - gv.state_vector[i, j, :])

        # the sign is overseen in the calculate flux
        gv.artificial_dissipation[i+1,j,3,:] = gv.artificial_dissipation[i,j,1,:]

    # north/south
    for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y-1):
        gv.artificial_dissipation[i,j,1,:] = a_d_coefficient_gamma[i,j,2] * (gv.state_vector[i,j+1,:] - gv.state_vector[i, j,:])

        gv.artificial_dissipation[i,j+1,0,:] = gv.artificial_dissipation[i,j,2,:]

    # boundaries are still to be implemented!!!

    # south artificial dissipation at bottom boundary
    gv.artificial_dissipation[:, 0, 0] = 0
    # north artificial dissipation at top boundary
    gv.artificial_dissipation[:, -1, 2] = 0
    # west artificial dissipation at inlet boundary
    gv.artificial_dissipation[0, :, 3] = 0
    # east artificial dissipation at outlet boundary
    gv.artificial_dissipation[-1, :, 1] = 0

    return None


def calculate_coefficient():
    global a_d_coefficient_gamma
    # # Compute the dissipation coefficient for the east and west cell faces using vectorized operations
    # a_d_coefficient_gamma[:-1, :, 1] = (0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * 
    #                                     (np.abs(0.5 * (gv.u[:-1, :, 0] + gv.u[1:, :, 0]) * gv.ndS[:-1, :, 1, 0]  # Dot product: u_x * ndS_x
    #                                           + 0.5 * (gv.u[:-1, :, 1] + gv.u[1:, :, 1]) * gv.ndS[:-1, :, 1, 1])  # Dot product: u_y * ndS_y
    #                                           + 0.5 * (gv.c[:-1, :] + gv.c[1:, :]) * np.linalg.norm(gv.ndS[:-1, :, 1], axis=-1)))  # Speed of sound term
    # # Copy the east face dissipation coefficient to the west face of the neighboring cell
    # a_d_coefficient_gamma[1:, :, 3] = a_d_coefficient_gamma[:-1, :, 1]

    # -------------------------------------------------------------------------
    # Legacy explanation (loop-based version):
    # Compute the dissipation coefficient for the east and west cell faces
    for i, j in np.ndindex(NUM_CELLS_X-1, NUM_CELLS_Y):  # Loop through all cells except the last column
        # Compute average velocity (u) and speed of sound (c) for the west cell face
        u_west = (gv.u[:, i, j] + gv.u[:, i+1, j]) * 0.5  # Velocity at the west face
        c_west = (gv.c[i, j] + gv.c[i+1, j]) * 0.5        # Speed of sound at the west face
    
        # Compute the dissipation terms:
        temp1 = np.abs(np.dot(u_west, gv.ndS[i, j, 1, :]))  # Dot product: u_west · ndS
        temp2 = c_west * np.linalg.norm(gv.ndS[i, j, 1, :])  # Speed of sound term
    
        # Calculate the dissipation coefficient for the east cell face
        a_d_coefficient_gamma[i, j, 1] = 0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * (temp1 + temp2)
    
        # Copy the east face dissipation coefficient to the west face of the neighboring cell
        a_d_coefficient_gamma[i+1, j, 3] = a_d_coefficient_gamma[i, j, 1]
    # -------------------------------------------------------------------------

    # # Compute the dissipation coefficient for the north and south cell faces using vectorized operations
    # a_d_coefficient_gamma[:, :-1, 2] = (0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * 
    #                                     (np.abs(0.5 * (gv.u[:, :-1, 0] + gv.u[:, 1:, 0]) * gv.ndS[:, :-1, 2, 0]  # Dot product: u_x * ndS_x
    #                                           + 0.5 * (gv.u[:, :-1, 1] + gv.u[:, 1:, 1]) * gv.ndS[:, :-1, 2, 1])  # Dot product: u_y * ndS_y
    #                                           + 0.5 * (gv.c[:, :-1] + gv.c[:, 1:]) * np.linalg.norm(gv.ndS[:, :-1, 2], axis=-1)))  # Speed of sound term
    # # Copy the north face dissipation coefficient to the south face of the neighboring cell
    # a_d_coefficient_gamma[:, 1:, 0] = a_d_coefficient_gamma[:, :-1, 2]

    # -------------------------------------------------------------------------
    # Legacy explanation (loop-based version):
    # Compute the dissipation coefficient for the north and south cell faces
    for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y-1):  # Loop through all cells except the last row
        # Compute average velocity (u) and speed of sound (c) for the north cell face
        u_north = (gv.u[:, i, j] + gv.u[:, i, j+1]) * 0.5  # Velocity at the north face
        c_north = (gv.c[i, j] + gv.c[i, j+1]) * 0.5        # Speed of sound at the north face
    
        # Compute the dissipation terms:
        temp1 = np.abs(np.dot(u_north, gv.ndS[i, j, 2, :]))  # Dot product: u_north · ndS
        temp2 = c_north * np.linalg.norm(gv.ndS[i, j, 2, :])  # Speed of sound term
    
        # Calculate the dissipation coefficient for the north cell face
        a_d_coefficient_gamma[i, j, 2] = 0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * (temp1 + temp2)
    
        # Copy the north face dissipation coefficient to the south face of the neighboring cell
        a_d_coefficient_gamma[i, j+1, 0] = a_d_coefficient_gamma[i, j, 2]
    # -------------------------------------------------------------------------

    # now calculate the artificial dissipation coefficents at the boundaries

    # south artificial dissipation coefficent (BOTTOM boundary) 
    a_d_coefficient_gamma[:, 0, 0] = 0
    # north artificial dissipation coefficent (TOP boundary) 
    a_d_coefficient_gamma[:, -1, 2] = 0
    # west artificial dissipation coefficent (inlet boundary) 
    a_d_coefficient_gamma[0, :, 3] = 0
    # east artificial dissipation coefficent (outlet boundary) 
    a_d_coefficient_gamma[-1, :, 1] = 0

    return None