# TODO
# (a) Add full artificial dissipation (for now just the simple implementation) 
# (?) Calculate artificial dissipation at the boundaries
#     - currently set to zero at the boundaries 
from constants import *
import global_vars as gv
import numpy as np

def update_artificial_dissipation_no_shocks(s_vector):
    # Calculate coefficients used for artificial dissipation
    calculate_coefficient()
    #----------------------------------------------------------------------
    # Compute east/west artificial dissipation
    #----------------------------------------------------------------------
    for i in range(1, NUM_CELLS_X - 2):  # Exclude boundary cells
        for j in range(NUM_CELLS_Y):  # Loop over all cells in the y-direction
            # Compute dissipation for the east/west direction
            gv.artificial_dissipation[i, j, 1, :] = - gv.a_d_coefficient_gamma[i, j, 1] * (
                  1 * s_vector[i + 2, j, :]
                - 3 * s_vector[i + 1, j, :]
                + 3 * s_vector[i, j, :]
                - 1 * s_vector[i - 1, j, :]
            )
            # Copy the east face dissipation to the west face of the neighboring cell
            gv.artificial_dissipation[i + 1, j, 3, :] = gv.artificial_dissipation[i, j, 1, :]

    #----------------------------------------------------------------------
    # Handle the west boundary (simplified first-order approach)
    i = 0
    for j in range(NUM_CELLS_Y):
        gv.artificial_dissipation[i, j, 1, :] = gv.a_d_coefficient_gamma[i, j, 1] * (
            s_vector[i + 1, j, :] - s_vector[i, j, :]
        )
        gv.artificial_dissipation[i + 1, j, 3, :] = gv.artificial_dissipation[i, j, 1, :]

    #----------------------------------------------------------------------
    # Handle the east boundary (simplified first-order approach)
    i = NUM_CELLS_X - 2
    for j in range(NUM_CELLS_Y):
        gv.artificial_dissipation[i, j, 1, :] = gv.a_d_coefficient_gamma[i, j, 1] * (
            s_vector[i + 1, j, :] - s_vector[i, j, :]
        )
        gv.artificial_dissipation[i + 1, j, 3, :] = gv.artificial_dissipation[i, j, 1, :]

    #----------------------------------------------------------------------
    # Compute north/south artificial dissipation
    #----------------------------------------------------------------------
    for i in range(NUM_CELLS_X):  # Loop over all cells in the x-direction
        for j in range(1, NUM_CELLS_Y - 2):  # Exclude boundary cells
            # Compute dissipation for the north/south direction
            gv.artificial_dissipation[i, j, 2, :] = - gv.a_d_coefficient_gamma[i, j, 2] * (
                  1 * s_vector[i, j + 2, :]
                - 3 * s_vector[i, j + 1, :]
                + 3 * s_vector[i, j, :]
                - 1 * s_vector[i, j - 1, :]
            )
            # Copy the north face dissipation to the south face of the neighboring cell
            gv.artificial_dissipation[i, j + 1, 0, :] = gv.artificial_dissipation[i, j, 2, :]

    #----------------------------------------------------------------------
    # Handle the south boundary (simplified first-order approach)
    j = 0
    for i in range(NUM_CELLS_X):
        gv.artificial_dissipation[i, j, 2, :] = gv.a_d_coefficient_gamma[i, j, 2] * (
            s_vector[i, j + 1, :] - s_vector[i, j, :]
        )
        gv.artificial_dissipation[i, j + 1, 0, :] = gv.artificial_dissipation[i, j, 2, :]

    #----------------------------------------------------------------------
    # Handle the north boundary (simplified first-order approach)
    j = NUM_CELLS_Y - 2
    for i in range(NUM_CELLS_X):
        gv.artificial_dissipation[i, j, 2, :] = gv.a_d_coefficient_gamma[i, j, 2] * (
            s_vector[i, j + 1, :] - s_vector[i, j, :]
        )
        gv.artificial_dissipation[i, j + 1, 0, :] = gv.artificial_dissipation[i, j, 2, :]

    #----------------------------------------------------------------------
    # Set artificial dissipation at boundary edges to zero
    #----------------------------------------------------------------------
    # South boundary (bottom edge)
    gv.artificial_dissipation[:, 0, 0] = 0
    # North boundary (top edge)
    gv.artificial_dissipation[:, -1, 2] = 0
    # West boundary (left edge)
    gv.artificial_dissipation[0, :, 3] = 0
    # East boundary (right edge)
    gv.artificial_dissipation[-1, :, 1] = 0

    return None

def calculate_coefficient():
    # Compute the dissipation coefficient for the east and west cell faces using vectorized operations
    gv.a_d_coefficient_gamma[:-1, :, 1] = (0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * 
                                        (np.abs(0.5 * (gv.u[:-1, :, 0] + gv.u[1:, :, 0]) * gv.ndS[:-1, :, 1, 0]   # Dot product: u_x * ndS_x
                                              + 0.5 * (gv.u[:-1, :, 1] + gv.u[1:, :, 1]) * gv.ndS[:-1, :, 1, 1])  # Dot product: u_y * ndS_y
                                              + 0.5 * (gv.c[:-1, :] + gv.c[1:, :]) * np.linalg.norm(gv.ndS[:-1, :, 1], axis=-1)))  # Speed of sound term
    # Copy the east face dissipation coefficient to the west face of the neighboring cell
    gv.a_d_coefficient_gamma[1:, :, 3] = gv.a_d_coefficient_gamma[:-1, :, 1]

    # -------------------------------------------------------------------------
    # Legacy explanation (loop-based version):
    # Compute the dissipation coefficient for the east and west cell faces
    #
    # for i in range(NUM_CELLS_X - 1):
    #     for j in range(NUM_CELLS_Y):
    #         # calculate the west
    #         u_west = 0.5 * (gv.u[i, j] + gv.u[i + 1, j])
    #         c_west = 0.5 * (gv.c[i, j] + gv.c[i + 1, j])

    #         a_d_coefficient_gamma[i, j, 1] = 0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * (
    #             np.abs(np.dot(u_west, gv.ndS[i, j, 1]))
    #             + c_west * np.linalg.norm(gv.ndS[i, j, 1])
    #         )

    #         a_d_coefficient_gamma[i + 1, j, 3] = a_d_coefficient_gamma[i, j, 1]

    # Compute the dissipation coefficient for the north and south cell faces using vectorized operations
    gv.a_d_coefficient_gamma[:, :-1, 2] = (0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * 
                                        (np.abs(0.5 * (gv.u[:, :-1, 0] + gv.u[:, 1:, 0]) * gv.ndS[:, :-1, 2, 0]  # Dot product: u_x * ndS_x
                                              + 0.5 * (gv.u[:, :-1, 1] + gv.u[:, 1:, 1]) * gv.ndS[:, :-1, 2, 1])  # Dot product: u_y * ndS_y
                                              + 0.5 * (gv.c[:, :-1] + gv.c[:, 1:]) * np.linalg.norm(gv.ndS[:, :-1, 2], axis=-1)))  # Speed of sound term
    # Copy the north face dissipation coefficient to the south face of the neighboring cell
    gv.a_d_coefficient_gamma[:, 1:, 0] = gv.a_d_coefficient_gamma[:, :-1, 2]

    # -------------------------------------------------------------------------
    # Legacy explanation (loop-based version):
    # Compute the dissipation coefficient for the north and south cell faces
    #
    # for i in range(NUM_CELLS_X):
    #     for j in range(NUM_CELLS_Y - 1):
    #         # calculate the west
    #         u_north = 0.5 * (gv.u[i, j] + gv.u[i, j + 1])
    #         c_north = 0.5 * (gv.c[i, j] + gv.c[i, j + 1])

    #         a_d_coefficient_gamma[i, j, 2] = 0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * (
    #             np.abs(np.dot(u_north, gv.ndS[i, j, 2]))
    #             + c_north * np.linalg.norm(gv.ndS[i, j, 2])
    #         )

    #         a_d_coefficient_gamma[i, j + 1, 0] = a_d_coefficient_gamma[i, j, 2]

    # now calculate the artificial dissipation coefficents at the boundaries

    # south artificial dissipation coefficent (BOTTOM boundary) 
    gv.a_d_coefficient_gamma[:, 0, 0] = 0
    # north artificial dissipation coefficent (TOP boundary) 
    gv.a_d_coefficient_gamma[:, -1, 2] = 0
    # west artificial dissipation coefficent (inlet boundary) 
    gv.a_d_coefficient_gamma[0, :, 3] = 0
    # east artificial dissipation coefficent (outlet boundary) 
    gv.a_d_coefficient_gamma[-1, :, 1] = 0

    return None

def update_artificial_dissipation_no_shocks_simple(s_vector):

    calculate_coefficient()

    # east/west
    for i, j in np.ndindex(NUM_CELLS_X-1, NUM_CELLS_Y):
        #artificial_dissipation[i,j,1] = a_d_coefficient_gamma[i,j,1] * (s_vector[i+1,j] - s_vector[i, j])
        # If artificial_dissipation is multi-component
        gv.artificial_dissipation[i, j, 1, :] = gv.a_d_coefficient_gamma[i, j, 1] * (s_vector[i+1, j, :] - s_vector[i, j, :])

        # the sign is overseen in the calculate flux
        gv.artificial_dissipation[i+1, j, 3, :] = gv.artificial_dissipation[i, j, 1, :]

    # north/south
    for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y-1):
        # North AD 
        gv.artificial_dissipation[i, j, 2, :] = gv.a_d_coefficient_gamma[i, j, 2] * (s_vector[i, j+1, :] - s_vector[i, j, :])

        gv.artificial_dissipation[i, j+1, 0, :] = gv.artificial_dissipation[i, j, 2, :]

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

# this function also accounts for shocks
def update_artificial_dissipation(s_vector):
    
    calculate_coefficient_WIP()

    # east/west
    for i, j in np.ndindex(NUM_CELLS_X-1, NUM_CELLS_Y):
        #artificial_dissipation[i,j,1] = a_d_coefficient_gamma[i,j,1] * (s_vector[i+1,j] - s_vector[i, j])
        # If artificial_dissipation is multi-component
        gv.artificial_dissipation[i, j, 1, :] = (gv.a_d_coefficient_eta[i, j, 1] * (s_vector[i + 1, j, :] - s_vector[i, j, :])
                                                + gv.a_d_coefficient_gamma[i, j, 1] * (s_vector[i + 1, j, :] - s_vector[i, j, :]))
        # the sign is overseen in the calculate flux
        gv.artificial_dissipation[i + 1, j, 3, :] = gv.artificial_dissipation[i, j, 1, :]

    # north/south
    for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y-1):
        # North AD 
        gv.artificial_dissipation[i, j, 2, :] = (gv.a_d_coefficient_eta[i, j, 2] * (s_vector[i, j + 1, :] - s_vector[i, j, :])
                                                + gv.a_d_coefficient_gamma[i, j, 2] * (s_vector[i, j + 1, :] - s_vector[i, j, :]))

        gv.artificial_dissipation[i, j+1, 0, :] = gv.artificial_dissipation[i, j, 2, :]
    
    # south artificial dissipation at bottom boundary
    gv.artificial_dissipation[:, 0, 0] = 0
    # north artificial dissipation at top boundary
    gv.artificial_dissipation[:, -1, 2] = 0
    # west artificial dissipation at inlet boundary
    gv.artificial_dissipation[0, :, 3] = 0
    # east artificial dissipation at outlet boundary
    gv.artificial_dissipation[-1, :, 1] = 0

    return None

def calculate_coefficient_WIP():
    for i, j in np.ndindex(NUM_CELLS_X - 1, NUM_CELLS_Y - 1):
        gv.nu[i, j] = np.abs(
            (gv.p[i + 1, j] - 2 * gv.p[i, j] + gv.p[i - 1, j]) /
            (gv.p[i + 1, j] + 2 * gv.p[i, j] + gv.p[i - 1, j])
        )
    
    for i in range(NUM_CELLS_X - 1):
        for j in range(NUM_CELLS_Y):
            # calculate maximum nu for the artificial dissipation (simple)
            max_mu = max(gv.nu[i, j], gv.nu[i+1, j])
            # calculate the west
            temp = (np.abs(0.5 * (gv.u[i, j, 0] + gv.u[i + 1, j, 0]) * gv.ndS[i, j, 1, 0]
                         + 0.5 * (gv.u[i, j, 1] + gv.u[i + 1, j, 1]) * gv.ndS[i, j, 1, 1])
                         + 0.5 * (gv.c[i, j] + gv.c[i + 1, j]) * np.abs(np.sqrt(gv.ndS[i, j, 1, 0]**2 + gv.ndS[i, j, 1, 1]**2)))

            gv.a_d_coefficient_eta[i, j, 1] = 0.5 * ARTIFICIAL_DISSIPATION_KAPPA_2 * temp * max_mu
            # calculate the east
            gv.a_d_coefficient_eta[i + 1, j, 3] = gv.a_d_coefficient_eta[i, j, 1]

            # calculate the west
            gv.a_d_coefficient_gamma[i, j, 1] = max(0, 
                                                    0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * temp - gv.a_d_coefficient_eta[i, j, 1])
            # calculate the east
            gv.a_d_coefficient_gamma[i + 1, j, 3] = gv.a_d_coefficient_gamma[i, j, 1]

    for i in range(NUM_CELLS_X):
        for j in range(NUM_CELLS_Y - 1):
            # calculate maximum nu for the artificial dissipation (simple)
            max_mu = np.maximum(gv.nu[i, j], gv.nu[i, j+1])
            # calculate the north
            temp = (np.abs(0.5 * (gv.u[i, j, 0] + gv.u[i, j + 1, 0]) * gv.ndS[i, j, 2, 0]
                         + 0.5 * (gv.u[i, j, 1] + gv.u[i, j + 1, 1]) * gv.ndS[i, j, 2, 1])
                         + 0.5 * (gv.c[i, j] + gv.c[i, j + 1]) * np.abs(np.sqrt(gv.ndS[i, j, 2, 0]**2 + gv.ndS[i, j, 2, 1]**2)))

            gv.a_d_coefficient_eta[i, j, 2] = 0.5 * ARTIFICIAL_DISSIPATION_KAPPA_2 * temp * max_mu
            # calculate the south
            gv.a_d_coefficient_eta[i, j + 1, 0] = gv.a_d_coefficient_eta[i, j, 2]
            # calculate the north
            gv.a_d_coefficient_gamma[i, j, 2] = max(0, 
                                                    0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * temp - gv.a_d_coefficient_eta[i, j, 2])
            # calculate the south
            gv.a_d_coefficient_gamma[i, j + 1, 0] = gv.a_d_coefficient_gamma[i, j, 2]

    # now calculate the artificial dissipation coefficents at the boundaries

    # south artificial dissipation coefficent (BOTTOM boundary) 
    gv.a_d_coefficient_gamma[:, 0, 0] = 0
    # north artificial dissipation coefficent (TOP boundary) 
    gv.a_d_coefficient_gamma[:, -1, 2] = 0
    # west artificial dissipation coefficent (inlet boundary) 
    gv.a_d_coefficient_gamma[0, :, 3] = 0
    # east artificial dissipation coefficent (outlet boundary) 
    gv.a_d_coefficient_gamma[-1, :, 1] = 0

    return None