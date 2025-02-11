# TODO
# (?) Calculate artificial dissipation at the boundaries
#     - currently set to zero at the boundaries 
from constants import *
import global_vars as gv
import numpy as np

#######################################################################
# Function to compute artificial dissipation without considering shocks
#######################################################################

def update_artificial_dissipation_subsonic(s_vector):
    """
    This function computes the artificial dissipation for each cell in the grid
    based on the provided state vector `s_vector`. It calculates the dissipation 
    for both the east/west and north/south directions using a difference scheme 
    and updates the dissipation for the interior cells. Artificial dissipation
    at the boundaries are assumed to be zero.

    Parameters:
    -----------
    s_vector : numpy.ndarray
        A 3D array representing the state vector of the grid with shape (i, j, 4). 
        The third dimension represents the conserved variables (e.g., rho, rho*u, 
        rho*v, rho*E).

    Returns:
    --------
    None : The function directly updates the global variable `artificial_dissipation`.
    """
    # Calculate coefficients used for artificial dissipation
    _calculate_coefficient_subsonic()

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

def update_artificial_dissipation_subsonic_simple(s_vector):
    """
    This function calculates artificial dissipation using a SIMPLIFIED approach 
    for each cell in the grid, based on the provided state vector `s_vector`. 
    The dissipation is computed for both east/west and north/south directions 
    using a finite difference scheme, and the dissipation is updated for the 
    interior cells. For boundary cells, the dissipation is assumed to be zero.

    Parameters:
    -----------
    s_vector : numpy.ndarray
        A 3D array representing the state vector of the grid with shape (i, j, 4). 
        The third dimension represents the conserved variables (e.g., rho, rho*u, 
        rho*v, rho*E).

    Returns:
    --------
    None
        The function directly updates the global variable `artificial_dissipation`.
    """

    _calculate_coefficient_subsonic()

    #----------------------------------------------------------------------
    # Compute east/west artificial dissipation
    #----------------------------------------------------------------------
    for i, j in np.ndindex(NUM_CELLS_X-1, NUM_CELLS_Y):
        #artificial_dissipation[i,j,1] = a_d_coefficient_gamma[i,j,1] * (s_vector[i+1,j] - s_vector[i, j])
        # If artificial_dissipation is multi-component
        gv.artificial_dissipation[i, j, 1, :] = gv.a_d_coefficient_gamma[i, j, 1] * (s_vector[i+1, j, :] - s_vector[i, j, :])

        # the sign is overseen in the calculate flux
        gv.artificial_dissipation[i+1, j, 3, :] = gv.artificial_dissipation[i, j, 1, :]

    #----------------------------------------------------------------------
    # Compute north/south artificial dissipation
    #----------------------------------------------------------------------
    for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y-1):
        # North AD 
        gv.artificial_dissipation[i, j, 2, :] = gv.a_d_coefficient_gamma[i, j, 2] * (s_vector[i, j+1, :] - s_vector[i, j, :])

        gv.artificial_dissipation[i, j+1, 0, :] = gv.artificial_dissipation[i, j, 2, :]

    #----------------------------------------------------------------------
    # Set artificial dissipation at boundary edges to zero
    #----------------------------------------------------------------------

    # south artificial dissipation at bottom boundary
    gv.artificial_dissipation[:, 0, 0] = 0
    # north artificial dissipation at top boundary
    gv.artificial_dissipation[:, -1, 2] = 0
    # west artificial dissipation at inlet boundary
    gv.artificial_dissipation[0, :, 3] = 0
    # east artificial dissipation at outlet boundary
    gv.artificial_dissipation[-1, :, 1] = 0

    return None

def _calculate_coefficient_subsonic():
    """
    Calculates the dissipation coefficients for the east, west, north, and south cell faces 
    in a subsonic flow regime. This function updates the global variable `a_d_coefficient_gamma`.

    The dissipation coefficient is calculated for each of the four primary directions (east, west, 
    north, and south) using a finite difference scheme with vectorized operations to optimize 
    performance. The function computes the dissipation for both the interior and boundary cells, 
    with boundary dissipation assumed to be zero (though this can be modified if necessary).

    Returns:
    --------
    None : Directly updates the global variable `gv.a_d_coefficient_gamma`.
    
    Notes:
    ------
    The function relies on the global variables:
    - `gv.u`: velocity field in the x and y directions
    - `gv.c`: speed of sound field
    - `gv.ndS`: normal direction to the cell faces
    - `ARTIFICIAL_DISSIPATION_KAPPA_4`: a constant related to the strength of artificial dissipation
    """

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

    #----------------------------------------------------------------------
    # Set artificial dissipation coefficient at boundary edges to zero (can be ignored)
    #----------------------------------------------------------------------

    # # south artificial dissipation coefficent (BOTTOM boundary) 
    # gv.a_d_coefficient_gamma[:, 0, 0] = 0
    # # north artificial dissipation coefficent (TOP boundary) 
    # gv.a_d_coefficient_gamma[:, -1, 2] = 0
    # # west artificial dissipation coefficent (inlet boundary) 
    # gv.a_d_coefficient_gamma[0, :, 3] = 0
    # # east artificial dissipation coefficent (outlet boundary) 
    # gv.a_d_coefficient_gamma[-1, :, 1] = 0

    return None


#######################################################################
# Function to compute artificial dissipation considering shocks
#######################################################################

def update_artificial_dissipation_supersonic(s_vector):
    """
    This function computes the artificial dissipation for each cell in the grid
    based on the provided state vector `s_vector`. It calculates the dissipation 
    for both the east/west and north/south directions using a difference scheme 
    and updates the dissipation for the interior cells. Artificial dissipation
    at the boundaries are assumed to be zero.

    Parameters:
    -----------
    s_vector : numpy.ndarray
        A 3D array representing the state vector of the grid with shape (i, j, 4). 
        The third dimension represents the conserved variables (e.g., rho, rho*u, 
        rho*v, rho*E).

    Returns:
    --------
    None : The function directly updates the global variable `artificial_dissipation`.
    """
    # Calculate coefficients (eta and gamma) used for artificial dissipation
    _calculate_coefficient_supersonic()
    #----------------------------------------------------------------------
    # Compute east/west artificial dissipation
    #----------------------------------------------------------------------
    for i in range(1, NUM_CELLS_X - 2):  # Exclude boundary cells
        for j in range(NUM_CELLS_Y):  # Loop over all cells in the y-direction
            # Compute dissipation for the east/west direction
            gv.artificial_dissipation[i, j, 1, :] = (
                gv.a_d_coefficient_eta[i, j, 1] * (
                    s_vector[i + 1, j] - s_vector[i, j]
                ) 
                - gv.a_d_coefficient_gamma[i, j, 1] * (
                    1 * s_vector[i + 2, j, :]
                    - 3 * s_vector[i + 1, j, :]
                    + 3 * s_vector[i, j, :]
                    - 1 * s_vector[i - 1, j, :]
                ))
            # Copy the east face dissipation to the west face of the neighboring cell
            gv.artificial_dissipation[i + 1, j, 3, :] = gv.artificial_dissipation[i, j, 1, :]

    #----------------------------------------------------------------------
    # Handle the west boundary (simplified first-order approach)
    i = 0
    for j in range(NUM_CELLS_Y):
        gv.artificial_dissipation[i, j, 1, :] = (gv.a_d_coefficient_eta[i, j, 1] + gv.a_d_coefficient_gamma[i, j, 1]) * (
            s_vector[i + 1, j, :] - s_vector[i, j, :]
        )
        gv.artificial_dissipation[i + 1, j, 3, :] = gv.artificial_dissipation[i, j, 1, :]

    #----------------------------------------------------------------------
    # Handle the east boundary (simplified first-order approach)
    i = NUM_CELLS_X - 2
    for j in range(NUM_CELLS_Y):
        gv.artificial_dissipation[i, j, 1, :] = (gv.a_d_coefficient_eta[i, j, 1] + gv.a_d_coefficient_gamma[i, j, 1]) * (
            s_vector[i + 1, j, :] - s_vector[i, j, :]
        )
        gv.artificial_dissipation[i + 1, j, 3, :] = gv.artificial_dissipation[i, j, 1, :]

    #----------------------------------------------------------------------
    # Compute north/south artificial dissipation
    #----------------------------------------------------------------------
    for i in range(NUM_CELLS_X):  # Loop over all cells in the x-direction
        for j in range(1, NUM_CELLS_Y - 2):  # Exclude boundary cells
            # Compute dissipation for the north/south direction
            gv.artificial_dissipation[i, j, 2, :] = (
                gv.a_d_coefficient_eta[i, j, 2] * (
                    s_vector[i, j + 1] - s_vector[i, j]    
                )
                - gv.a_d_coefficient_gamma[i, j, 2] * (
                    1 * s_vector[i, j + 2, :]
                    - 3 * s_vector[i, j + 1, :]
                    + 3 * s_vector[i, j, :]
                    - 1 * s_vector[i, j - 1, :]
                ))
            # Copy the north face dissipation to the south face of the neighboring cell
            gv.artificial_dissipation[i, j + 1, 0, :] = gv.artificial_dissipation[i, j, 2, :]

    #----------------------------------------------------------------------
    # Handle the south boundary (simplified first-order approach)
    j = 0
    for i in range(NUM_CELLS_X):
        gv.artificial_dissipation[i, j, 2, :] = (gv.a_d_coefficient_eta[i, j, 2] + gv.a_d_coefficient_gamma[i, j, 2]) * (
            s_vector[i, j + 1, :] - s_vector[i, j, :]
        )
        gv.artificial_dissipation[i, j + 1, 0, :] = gv.artificial_dissipation[i, j, 2, :]

    #----------------------------------------------------------------------
    # Handle the north boundary (simplified first-order approach)
    j = NUM_CELLS_Y - 2
    for i in range(NUM_CELLS_X):
        gv.artificial_dissipation[i, j, 2, :] = (gv.a_d_coefficient_eta[i, j, 2] + gv.a_d_coefficient_gamma[i, j, 2]) * (
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

def update_artificial_dissipation_supersonic_simple(s_vector):
    """
    This function calculates artificial dissipation using a SIMPLIFIED approach 
    for each cell in the grid, based on the provided state vector `s_vector`. 
    The dissipation is computed for both east/west and north/south directions 
    using a finite difference scheme, and the dissipation is updated for the 
    interior cells. For boundary cells, the dissipation is assumed to be zero.

    Parameters:
    -----------
    s_vector : numpy.ndarray
        A 3D array representing the state vector of the grid with shape (i, j, 4). 
        The third dimension represents the conserved variables (e.g., rho, rho*u, 
        rho*v, rho*E).

    Returns:
    --------
    None
        The function directly updates the global variable `artificial_dissipation`.
    """

    # Calculate coefficients (eta and gamma) used for artificial dissipation
    _calculate_coefficient_supersonic()

    #----------------------------------------------------------------------
    # Compute east/west artificial dissipation
    #----------------------------------------------------------------------
    for i, j in np.ndindex(NUM_CELLS_X-1, NUM_CELLS_Y):
        #artificial_dissipation[i,j,1] = a_d_coefficient_gamma[i,j,1] * (s_vector[i+1,j] - s_vector[i, j])
        # If artificial_dissipation is multi-component
        gv.artificial_dissipation[i, j, 1, :] = (gv.a_d_coefficient_eta[i, j, 1] * (s_vector[i + 1, j, :] - s_vector[i, j, :])
                                                + gv.a_d_coefficient_gamma[i, j, 1] * (s_vector[i + 1, j, :] - s_vector[i, j, :]))
        # the sign is overseen in the calculate flux
        gv.artificial_dissipation[i + 1, j, 3, :] = gv.artificial_dissipation[i, j, 1, :]

    #----------------------------------------------------------------------
    # Compute north/south artificial dissipation
    #----------------------------------------------------------------------
    for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y-1):
        # North AD 
        gv.artificial_dissipation[i, j, 2, :] = (gv.a_d_coefficient_eta[i, j, 2] * (s_vector[i, j + 1, :] - s_vector[i, j, :])
                                                + gv.a_d_coefficient_gamma[i, j, 2] * (s_vector[i, j + 1, :] - s_vector[i, j, :]))

        gv.artificial_dissipation[i, j+1, 0, :] = gv.artificial_dissipation[i, j, 2, :]
    
    #----------------------------------------------------------------------
    # Set artificial dissipation at boundary edges to zero
    #----------------------------------------------------------------------

    # south artificial dissipation at bottom boundary
    gv.artificial_dissipation[:, 0, 0] = 0
    # north artificial dissipation at top boundary
    gv.artificial_dissipation[:, -1, 2] = 0
    # west artificial dissipation at inlet boundary
    gv.artificial_dissipation[0, :, 3] = 0
    # east artificial dissipation at outlet boundary
    gv.artificial_dissipation[-1, :, 1] = 0

    return None

def _calculate_coefficient_supersonic():
    """
    Computes the artificial dissipation coefficients for the system in a supersonic flow regime. 
    This function calculates the dissipation coefficients `eta` and `gamma` at the east and north
    face of the grid. This function updates the global variables `a_d_coefficient_eta` and 
    `a_d_coefficient_gamma`.
    
    The function first calculates the pressure sensor values using the `_calculate_pressure_sensor()` function. 

    After computing the dissipation coefficients for each face, the values are stored in `gv.a_d_coefficient_eta`
    and `gv.a_d_coefficient_gamma`. Boundary dissipation coefficients are set to zero.

    Parameters:
    -----------
    None

    Returns:
    --------
    None : This function updates the global variables `gv.a_d_coefficient_eta` and `gv.a_d_coefficient_gamma`.
    """
    # Call the function to calculate the pressure sensor (ν) values.
    _calculate_pressure_sensor()

    #----------------------------------------------------------------------
    # Calculate the artificial dissipation coefficient at the east face
    #----------------------------------------------------------------------

    # |u ⋅ ΔS| + c |ΔS| at the face (i+1/2, j)
    tmp_east = (np.abs(0.5 * (gv.u[:-1, :, 0] + gv.u[1:, :, 0]) * gv.ndS[:-1, :, 1, 0]) 
                + 0.5 * (gv.c[:-1, :] + gv.c[1:, :]) * np.abs(gv.ndS[:-1, :, 1, 0]))

    # Calculate the artificial dissipation coefficient eta for east (west) face
    gv.a_d_coefficient_eta[:-1, :, 1] = 0.5 * ARTIFICIAL_DISSIPATION_KAPPA_2 * tmp_east * gv.nu_max[:-1, :, 1]
    # gv.a_d_coefficient_eta[1:, :, 3] = gv.a_d_coefficient_eta[:-1, :, 1]

    # Calculate the artificial dissipation coefficient gamma for east (west) face
    gv.a_d_coefficient_gamma[:-1, :, 1] = np.maximum(0, 
                   0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * tmp_east - gv.a_d_coefficient_eta[:-1, :, 1])
    # gv.a_d_coefficient_gamma[1:, :, 3] = gv.a_d_coefficient_gamma[:-1, :, 1]

    # -------------------------------------------------------------------------
    # Legacy explanation (loop-based version):
    # 
    # for j in range(NUM_CELLS_Y):
    #     for i in range(NUM_CELLS_X - 1):
    #         # This expression calculates a factor based on the differences between adjacent cells.
    #         temp = (np.abs(0.5 * (gv.u[i, j, 0] + gv.u[i + 1, j, 0]) * gv.ndS[i, j, 1, 0])
    #                      + 0.5 * (gv.c[i, j] + gv.c[i + 1, j]) * np.abs(gv.ndS[i, j, 1, 0]))

    #         # Calculate the artificial dissipation coefficient for the east face (eta).
    #         # This coefficient depends on a constant (ARTIFICIAL_DISSIPATION_KAPPA_2),
    #         # the previously computed temp value, and the pressure sensor value (ν_max).
    #         gv.a_d_coefficient_eta[i, j, 1] = 0.5 * ARTIFICIAL_DISSIPATION_KAPPA_2 * temp * gv.nu_max[i, j, 1]

    #         # The west face coefficient is the same as the east face.
    #         gv.a_d_coefficient_eta[i+1, j, 3] = gv.a_d_coefficient_eta[i, j, 1]

    #         # Calculate the artificial dissipation coefficient for the east face (gamma),
    #         # which is based on a second constant (ARTIFICIAL_DISSIPATION_KAPPA_4).
    #         gv.a_d_coefficient_gamma[i, j, 1] = max(0, 
    #                                                 0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * temp - gv.a_d_coefficient_eta[i, j, 1])

    #         # The west face coefficient (gamma) is the same (for a neighboring cell) as the east face.
    #         gv.a_d_coefficient_gamma[i + 1, j, 3] = gv.a_d_coefficient_gamma[i, j, 1]

    #----------------------------------------------------------------------
    # Calculate the artificial dissipation coefficient at the north face
    #----------------------------------------------------------------------

    # |u ⋅ ΔS| + c |ΔS| at the face (i, j+1/2)
    tmp_north = (np.abs(0.5 * (gv.u[:, :-1, 0] + gv.u[:, 1:, 0]) * gv.ndS[:, :-1, 1, 0]
                      + 0.5 * (gv.u[:, :-1, 1] + gv.u[:, 1:, 1]) * gv.ndS[:, :-1, 1, 1]) 
                      + 0.5 * (gv.c[:, :-1] + gv.c[:, 1:]) * np.abs(np.linalg.norm(gv.ndS[: , :-1, 2, :], axis=-1)))
    
    # Calculate the artificial dissipation coefficient eta for north (south) face
    gv.a_d_coefficient_eta[:, :-1, 2] = 0.5 * ARTIFICIAL_DISSIPATION_KAPPA_2 * tmp_north * gv.nu_max[:, :-1, 2]
    # gv.a_d_coefficient_eta[:, 1:, 0] = gv.a_d_coefficient_eta[:, :-1, 2]

    # Calculate the artificial dissipation coefficient gamma for north (south) face
    gv.a_d_coefficient_gamma[:, :-1, 2] = np.maximum(0, 
        0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * tmp_north - gv.a_d_coefficient_eta[:, :-1, 2])
    # gv.a_d_coefficient_gamma[:, 1:, 0] = gv.a_d_coefficient_gamma[:, :-1, 2]  # South face gamma = North face gamma

    # -------------------------------------------------------------------------
    # Legacy explanation (loop-based version):
    # 
    # for i in range(NUM_CELLS_X):
    #     for j in range(NUM_CELLS_Y - 1):

    #         # This expression calculates a factor based on the differences between adjacent cells.
    #         temp = (np.abs(0.5 * (gv.u[i, j, 0] + gv.u[i, j + 1, 0]) * gv.ndS[i, j, 2, 0]
    #                      + 0.5 * (gv.u[i, j, 1] + gv.u[i, j + 1, 1]) * gv.ndS[i, j, 2, 1])
    #                      + 0.5 * (gv.c[i, j] + gv.c[i, j + 1]) * np.abs(np.sqrt(gv.ndS[i, j, 2, 0]**2 + gv.ndS[i, j, 2, 1]**2)))
            
    #         # Calculate the artificial dissipation coefficient for the north face (eta).
    #         # This coefficient depends on a constant (ARTIFICIAL_DISSIPATION_KAPPA_2),
    #         # the previously computed temp value, and the pressure sensor value (ν_max).
    #         gv.a_d_coefficient_eta[i, j, 2] = 0.5 * ARTIFICIAL_DISSIPATION_KAPPA_2 * temp * gv.nu_max[i, j, 2]
            
    #         # The south face coefficient is the same as the north face.
    #         gv.a_d_coefficient_eta[i, j + 1, 0] = gv.a_d_coefficient_eta[i, j, 2]
            
    #         # Calculate the artificial dissipation coefficient for the north face (gamma),
    #         # which is based on a second constant (ARTIFICIAL_DISSIPATION_KAPPA_4).
    #         gv.a_d_coefficient_gamma[i, j, 2] = max(0, 
    #                                                 0.5 * ARTIFICIAL_DISSIPATION_KAPPA_4 * temp - gv.a_d_coefficient_eta[i, j, 2])
            
    #         # The south face coefficient (gamma) is the same (for a neighboring cell) as the north face.
    #         gv.a_d_coefficient_gamma[i, j + 1, 0] = gv.a_d_coefficient_gamma[i, j, 2]

    #----------------------------------------------------------------------
    # Calculate the artificial dissipation coefficients at the boundaries
    #----------------------------------------------------------------------

    # # south artificial dissipation coefficent (BOTTOM boundary) 
    # gv.a_d_coefficient_gamma[:, 0, 0] = 0
    # # north artificial dissipation coefficent (TOP boundary) 
    # gv.a_d_coefficient_gamma[:, -1, 2] = 0
    # # west artificial dissipation coefficent (inlet boundary) 
    # gv.a_d_coefficient_gamma[0, :, 3] = 0
    # # east artificial dissipation coefficent (outlet boundary) 
    # gv.a_d_coefficient_gamma[-1, :, 1] = 0

    return None

def _calculate_pressure_sensor():
    """
    Compute the pressure sensor (ν) and face maximum values (ν_max).

    The pressure sensor ν is computed using a second-order finite difference 
    scheme to capture variations in pressure:

        ν_ij = | ( p_{i+1,j} - 2*p_{i,j} + p_{i-1,j} ) / ( p_{i+1,j} + 2*p_{i,j} + p_{i-1,j} ) |

    This helps identify regions of sharp pressure gradients.

    Face-based maximum values (ν_max) are computed to ensure a smooth transition 
    between cells, using a stencil of surrounding ν values. For example, 
    at an east face:

        ν_{i+1/2,j} = max( ν_{i-1,j}, ν_{i,j}, ν_{i+1,j}, ν_{i+2,j} )

    **Boundary Treatment:**
    - At the domain boundaries, the stencil is reduced from four to three points 
        since neighboring values outside the grid are unavailable.
    - Near the **west (i=0)** and **east (i=N-1)** boundaries, only valid neighboring 
        pressure values are used.
    - Near the **south (j=0)** and **north (j=M-1)** boundaries, adjustments are made 
        similarly to avoid accessing out-of-bounds indices.

    Returns:
        None : The function directly updates the global variables `nu` and `nu_max`.
    """

    # Compute ν in the interior of the domain.
    gv.nu[1:-1, :] = np.abs(
        (gv.p[2:, :] - 2 * gv.p[1:-1, :] + gv.p[:-2, :]) /
        (gv.p[2:, :] + 2 * gv.p[1:-1, :] + gv.p[:-2, :])
    )
    
    # ----- East / West Faces (index 1, 3) -----
    # Interior: using four neighboring cells.
    gv.nu_max[1:-2, :, 1] = np.maximum.reduce(
        (gv.nu[:-3, :], gv.nu[1:-2, :], gv.nu[2:-1, :], gv.nu[3:, :])
    )
    # Near east boundary: use the last three available cells.
    gv.nu_max[-2, :, 1] = np.maximum.reduce(
        (gv.nu[-3, :], gv.nu[-2, :], gv.nu[-1, :])
    )
    # Near west boundary: use the first three cells.
    gv.nu_max[0, :, 1] = np.maximum.reduce(
        (gv.nu[0, :], gv.nu[1, :], gv.nu[2, :])
    )
    
    # ----- North / South Faces (index 2, 0) -----
    # Interior: along the y-direction, use four consecutive cells.
    gv.nu_max[:, 1:-2, 2] = np.maximum.reduce(
        (gv.nu[:, :-3], gv.nu[:, 1:-2], gv.nu[:, 2:-1], gv.nu[:, 3:])
    )
    # Near north boundary: use three points.
    gv.nu_max[:, -2, 2] = np.maximum.reduce(
        (gv.nu[:, -3], gv.nu[:, -2], gv.nu[:, -1])
    )
    # Near south boundary: use three points.
    gv.nu_max[:, 0, 2] = np.maximum.reduce(
        (gv.nu[:, 0], gv.nu[:, 1], gv.nu[:, 2])
    )

    return None


#######################################################################
# Below is some old (simpler) code to calculate the coefficients
#######################################################################

def calculate_coefficient_old():
    
    _calculate_pressure_sensor()
    
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