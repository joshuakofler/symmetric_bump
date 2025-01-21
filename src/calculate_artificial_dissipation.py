# TODO 
# (?) Calculate artificial dissipation at the boundaries
#     - currently set to zero at the boundaries 

from global_var import *
import numpy as np

def update_artificial_dissipation():
    # east/west
    for i, j in np.ndindex(NUM_CELLS_X-1, NUM_CELLS_Y):
        #artificial_dissipation[i,j,1] = a_d_coefficient_gamma[i,j,1] * (state_vector[i+1,j] - state_vector[i, j])
        # If artificial_dissipation is multi-component
        artificial_dissipation[i, j, 1, :] = a_d_coefficient_gamma[i, j, 1] * (state_vector[i+1, j, :] - state_vector[i, j, :])

        # the sign is overseen in the calculate flux
        artificial_dissipation[i+1,j,3,:] = artificial_dissipation[i,j,1,:]

    # north/south
    for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y-1):
        artificial_dissipation[i,j,1,:] = a_d_coefficient_gamma[i,j,2] * (state_vector[i,j+1,:] - state_vector[i, j,:])

        artificial_dissipation[i,j+1,0,:] = artificial_dissipation[i,j,2,:]

    # boundaries are still to be implemented!!!

    # south artificial dissipation at bottom boundary
    artificial_dissipation[:, 0, 0] = 0
    # north artificial dissipation at top boundary
    artificial_dissipation[:, -1, 2] = 0
    # west artificial dissipation at inlet boundary
    artificial_dissipation[0, :, 3] = 0
    # east artificial dissipation at outlet boundary
    artificial_dissipation[-1, :, 1] = 0

    return None


def calculate_coefficient():
    # Compute the dissipation coefficient for the east and west cell faces using vectorized operations
    a_d_coefficient_gamma[:-1, :, 1] = (0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * 
                                        (np.abs(0.5 * (u[:-1, :, 0] + u[1:, :, 0]) * ndS[:-1, :, 1, 0]  # Dot product: u_x * ndS_x
                                                + 0.5 * (u[:-1, :, 1] + u[1:, :, 1]) * ndS[:-1, :, 1, 1])  # Dot product: u_y * ndS_y
                                        + 0.5 * (c[:-1, :] + c[1:, :]) * np.linalg.norm(ndS[:-1, :, 1], axis=-1)))  # Speed of sound term
    # Copy the east face dissipation coefficient to the west face of the neighboring cell
    a_d_coefficient_gamma[1:, :, 3] = a_d_coefficient_gamma[:-1, :, 1]

    # -------------------------------------------------------------------------
    # Legacy explanation (loop-based version):
    # Compute the dissipation coefficient for the east and west cell faces
    # for i, j in np.ndindex(NUM_CELLS_X-1, NUM_CELLS_Y):  # Loop through all cells except the last column
    #     # Compute average velocity (u) and speed of sound (c) for the west cell face
    #     u_west = (u[:, i, j] + u[:, i+1, j]) * 0.5  # Velocity at the west face
    #     c_west = (c[i, j] + c[i+1, j]) * 0.5        # Speed of sound at the west face
    #
    #     # Compute the dissipation terms:
    #     temp1 = np.abs(np.dot(u_west, ndS[i, j, 1, :]))  # Dot product: u_west · ndS
    #     temp2 = c_west * np.linalg.norm(ndS[i, j, 1, :])  # Speed of sound term
    #
    #     # Calculate the dissipation coefficient for the east cell face
    #     a_d_coefficient_gamma[i, j, 1] = 0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * (temp1 + temp2)
    #
    #     # Copy the east face dissipation coefficient to the west face of the neighboring cell
    #     a_d_coefficient_gamma[i+1, j, 3] = a_d_coefficient_gamma[i, j, 1]
    # -------------------------------------------------------------------------

    # Compute the dissipation coefficient for the north and south cell faces using vectorized operations
    a_d_coefficient_gamma[:, :-1, 2] = (0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * 
                                        (np.abs(0.5 * (u[:, :-1, 0] + u[:, 1:, 0]) * ndS[:, :-1, 2, 0]  # Dot product: u_x * ndS_x
                                                + 0.5 * (u[:, :-1, 1] + u[:, 1:, 1]) * ndS[:, :-1, 2, 1])  # Dot product: u_y * ndS_y
                                        + 0.5 * (c[:, :-1] + c[:, 1:]) * np.linalg.norm(ndS[:, :-1, 2], axis=-1)))  # Speed of sound term
    # Copy the north face dissipation coefficient to the south face of the neighboring cell
    a_d_coefficient_gamma[:, 1:, 0] = a_d_coefficient_gamma[:, :-1, 2]

    # -------------------------------------------------------------------------
    # Legacy explanation (loop-based version):
    # Compute the dissipation coefficient for the north and south cell faces
    # for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y-1):  # Loop through all cells except the last row
    #     # Compute average velocity (u) and speed of sound (c) for the north cell face
    #     u_north = (u[:, i, j] + u[:, i, j+1]) * 0.5  # Velocity at the north face
    #     c_north = (c[i, j] + c[i, j+1]) * 0.5        # Speed of sound at the north face
    #
    #     # Compute the dissipation terms:
    #     temp1 = np.abs(np.dot(u_north, ndS[i, j, 2, :]))  # Dot product: u_north · ndS
    #     temp2 = c_north * np.linalg.norm(ndS[i, j, 2, :])  # Speed of sound term
    #
    #     # Calculate the dissipation coefficient for the north cell face
    #     a_d_coefficient_gamma[i, j, 2] = 0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * (temp1 + temp2)
    #
    #     # Copy the north face dissipation coefficient to the south face of the neighboring cell
    #     a_d_coefficient_gamma[i, j+1, 0] = a_d_coefficient_gamma[i, j, 2]
    # -------------------------------------------------------------------------

    # now calculate the artificial dissipation coefficents at the boundaries

    # south artificial dissipation coefficent (BOTTOM boundary) 
    a_d_coefficient_gamma[:,0,0] = 0
    # north artificial dissipation coefficent (TOP boundary) 
    a_d_coefficient_gamma[:,-1,2] = 0
    # west artificial dissipation coefficent (inlet boundary) 
    a_d_coefficient_gamma[0,:,3] = 0
    # east artificial dissipation coefficent (outlet boundary) 
    a_d_coefficient_gamma[-1,:,1] = 0

    return None