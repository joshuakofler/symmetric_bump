# TODO: Done

from constants import *
import global_vars as gv
import numpy as np
import calculate_artificial_dissipation as ad

# Select the function for updating artificial dissipation based on the flow type.
# If IS_SUBSONIC (defined in constants), use the subsonic dissipation update function.
# Otherwise, use the supersonic dissipation update function.
update_artificial_dissipation = ad.update_artificial_dissipation_subsonic if USE_SUBSONIC_AD else ad.update_artificial_dissipation_supersonic

def update_flux(s_vector):
    # Update the flux vector components (f) in the x-direction
    gv.f[:, :, 0] = gv.rho[:, :] * gv.u[:, :, 0]
    gv.f[:, :, 1] = gv.rho[:, :] * np.power(gv.u[:, :, 0], 2) + gv.p[:, :]
    gv.f[:, :, 2] = gv.rho[:, :] * gv.u[:, :, 0] * gv.u[:, :, 1]
    gv.f[:, :, 3] = gv.rho[:, :] * gv.u[:, :, 0] * gv.H[:, :]

    # Update the flux vector components (g) in the y-direction
    gv.g[:, :, 0] = gv.rho[:, :] * gv.u[:, :, 1]
    gv.g[:, :, 1] = gv.rho[:, :] * gv.u[:, :, 0] * gv.u[:, :, 1]
    gv.g[:, :, 2] = gv.rho[:, :] * np.power(gv.u[:, :, 1], 2) + gv.p[:, :]
    gv.g[:, :, 3] = gv.rho[:, :] * gv.u[:, :, 1] * gv.H[:, :]

    #######################################################################
    # Calculate interior fluxes
    #######################################################################

    # Calculate the interior fluxes at the east face (index 1) using vectorized operations.
    # For each cell, the east face flux is computed as the average of values from 
    # the current cell and the adjacent cell to the east in the x-direction.
    # These values are scaled by the normal vector components at the east face.

    # f[:-1,:,:] selects all except the last in the x-direction (current cells).
    # f[:,1:,:] selects all except the first in the x-direction (east neighbors).
    # The two are averaged to calculate the flux contribution from f at the east face.
    # Similarly, g[:,:-1,:] and g[:,1:,:] contribute the flux from g (in x-direction equal to zero).
    # ndS[:,:-1,1,0] and ndS[:,:-1,1,1] are the normal vector components at the east face.

    # East flux (index 1)
    gv.F[:-1, :, 1, :] = (0.5 * (gv.f[:-1, :, :] + gv.f[1:, :, :]) * gv.ndS[:-1, :, 1, 0][..., np.newaxis])
                      # + 0.5 * (gv.g[:-1, :, :] + gv.g[1:, :, :]) * gv.ndS[:-1, :, 1, 1][..., np.newaxis])

    # Assign the east face flux to the west face (index 3) of the neighboring cell.
    # This ensures continuity of fluxes across the east-west interface between cells.
    # The flux is negated because the normal vectors for the east and west faces
    # point in opposite directions.

    # West flux (index 3)
    gv.F[1:, :, 3, :] = - gv.F[:-1, :, 1, :]
    
    # Calculate the interior fluxes at the north face (index 2) using vectorized operations.
    # For each cell, the north face flux is computed as the average of values from 
    # the current cell and the adjacent cell to the north in the y-direction.
    # These values are scaled by the normal vector components at the north face.

    # f[:,:-1,:] selects all rows except the last in the y-direction (current cells).
    # f[:,1:,:] selects all rows except the first in the y-direction (north neighbors).
    # The two are averaged to calculate the flux contribution from f at the north face.
    # Similarly, g[:,:-1,:] and g[:,1:,:] contribute the flux from g.
    # ndS[:,:-1,2,0] and ndS[:,:-1,2,1] are the normal vector components at the north face.

    # North flux (index 2)
    gv.F[:, :-1, 2, :] = (0.5 * (gv.f[:, :-1, :] + gv.f[:, 1:, :]) * gv.ndS[:, :-1, 2, 0][..., np.newaxis] 
                        + 0.5 * (gv.g[:, :-1, :] + gv.g[:, 1:, :]) * gv.ndS[:, :-1, 2, 1][..., np.newaxis])

    # Assign the north face flux to the south face (index 0) of the neighboring cell.
    # This ensures continuity of fluxes across the north-south interface between cells.
    # The flux is negated because the normal vectors for the north and south faces
    # point in opposite directions.

    # South flux (index 0)
    gv.F[:, 1:, 0, :] = - gv.F[:, :-1, 2, :]

    # # The code blocks below are equivalent to the above vectorized implementation,
    # # but uses explicit looping over all cells to calculate fluxes.

    # # Calculate interior fluxes at the east/west face using a loop
    # for i, j in np.ndindex(NUM_CELLS_X - 1, NUM_CELLS_Y):
    #     # East face (index 1) flux calculation
    #     # Average the values of f and g between adjacent cells and scale by
    #     # the normal vector components at the east face.
    #     gv.F[i, j, 1, :] = (0.5 * (gv.f[i, j, :] + gv.f[i + 1, j, :]) * gv.ndS[i, j, 1, 0])
    #                         #+ 0.5 * (gv.g[i, j, :] + gv.g[i + 1, j, :]) * gv.ndS[i, j, 1, 1])

    #     # Copy the calculated flux directly to the west face (index 3) of the neighbor cell.
    #     gv.F[i + 1, j, 3, :] = - gv.F[i, j, 1, :]

    # # Calculate the interior fluxes at the north face using a loop
    # for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y - 1):
    #     # Calculate the north face (index 2) flux between the current cell (i, j)
    #     # and the adjacent cell to the north (i, j+1).
    #     gv.F[i, j, 2, :] = (0.5 * (gv.f[i, j, :] + gv.f[i, j + 1, :]) * gv.ndS[i, j, 2, 0] 
    #                       + 0.5 * (gv.g[i, j, :] + gv.g[i, j + 1, :]) * gv.ndS[i, j, 2, 1])
    
    #     # Copy the calculated north face flux to the south face (index 0)
    #     # of the adjacent cell (i, j+1), with a negation to account for
    #     # the direction of the normal vector.
    #     gv.F[i, j + 1, 0, :] = - gv.F[i, j, 2, :]

    #######################################################################
    # Boundary treatment
    #######################################################################
    # Calculate the flux at the boundaries
    _calculate_boundary_flux()
    
    #######################################################################
    # Add artificial dissipation
    #######################################################################

    # Add numerical flux corrections to all faces
    update_artificial_dissipation(s_vector)

    # Compute corrected flux for the east face
    gv.F_star[:, :, 1, :] = gv.F[:, :, 1, :] - gv.artificial_dissipation[:, :, 1, :]  # East face (index 1)

    # Compute corrected flux for the west face
    gv.F_star[:, :, 3, :] = gv.F[:, :, 3, :] + gv.artificial_dissipation[:, :, 3, :]  # West face (index 3)

    # Compute corrected flux for the north face
    gv.F_star[:, :, 2, :] = gv.F[:, :, 2, :] - gv.artificial_dissipation[:, :, 2, :]  # North face (index 2)

    # Compute corrected flux for the south face
    gv.F_star[:, :, 0, :] = gv.F[:, :, 0, :] + gv.artificial_dissipation[:, :, 0, :]  # South face (index 0)

    return None

def _calculate_boundary_flux():
    
    #----------------------------------------------------------------------
    # Compute the NORTH flux at the top wall (i, j = NUM_CELLS_Y-1)
    #----------------------------------------------------------------------
    gv.F[:, -1, 2, 0] = 0.0
    gv.F[:, -1, 2, 1] = 0.0
    gv.F[:, -1, 2, 2] = gv.p[:, -1] * gv.ndS[:, -1, 2, 1]
    gv.F[:, -1, 2, 3] = 0.0

    #----------------------------------------------------------------------
    # Compute the SOUTH flux at the bottom wall (i, j = 0)
    #----------------------------------------------------------------------
    gv.F[:, 0, 0, 0] = 0.0
    gv.F[:, 0, 0, 1] = gv.p[:, 0] * gv.ndS[:, 0, 0, 0]
    gv.F[:, 0, 0, 2] = gv.p[:, 0] * gv.ndS[:, 0, 0, 1]
    gv.F[:, 0, 0, 3] = 0.0

    #----------------------------------------------------------------------
    # Compute the WEST flux at the inlet cell (i = 0, j)
    #----------------------------------------------------------------------
    # Set y-velocity component at inlet to 0
    # v_in = 0
    # Calculate speed of sound at inlet (c_in)
    gv.c_in = ((HEAT_CAPACITY_RATIO - 1) * 0.25 * (gv.u_infty - np.sqrt(np.power(gv.u[0, :, 0], 2) + np.power(gv.u[0, :, 1], 2))) 
                                                + 0.5 * (gv.c_infty + gv.c[0, :]))
    # Calculate inlet velocity (u_in)
    gv.u_in = (np.sqrt(np.power(gv.u[0,:,0], 2) + np.power(gv.u[0,:,1], 2)) + 2 / (HEAT_CAPACITY_RATIO - 1) * (gv.c_in - gv.c[0,:]))
    # Calculate inlet temperature (T_in)
    gv.T_in = np.power(gv.c_in, 2) / (HEAT_CAPACITY_RATIO * GAS_CONSTANT)
    # Calculate inlet density (rho_in)
    gv.rho_in = np.power(GAS_CONSTANT * gv.T_in * np.power(gv.rho_infty, HEAT_CAPACITY_RATIO) / gv.p_infty, 1/(HEAT_CAPACITY_RATIO-1))
    # Calculate inlet pressure (p_in)
    gv.p_in = gv.rho_in * GAS_CONSTANT * gv.T_in
    # Calculate inlet enthalpy (H_in)
    gv.H_in = SPECIFIC_HEAT_CP * gv.T_in + 0.5 * (np.power(gv.u_in, 2)) # + np.power(v_in,2))
    # Calculate inlet mach number (M_in)
    gv.M_in = gv.u_in[:] / gv.c_in[:]

    # West flux (fw) at the inlet cell (i = 0, j)
    gv.F[0, :, 3, 0] = gv.rho_in * gv.u_in * gv.ndS[0, :, 3, 0]
    gv.F[0, :, 3, 1] = (gv.rho_in * np.power(gv.u_in, 2) + gv.p_in) * gv.ndS[0, :, 3, 0]
    gv.F[0, :, 3, 2] = 0.0 # rho_in * u_in * v_in * gv.ndS[0, :, 3, 0]
    gv.F[0, :, 3, 3] = gv.rho_in * gv.u_in * gv.H_in * gv.ndS[0, :, 3, 0]

    #----------------------------------------------------------------------
    # Compute the EAST flux at the outlet cell (i = NUM_CELLS_X-1, j)
    #----------------------------------------------------------------------
    # Set outlet pressure (p_out) to atmospheric pressure
    gv.p_out = gv.p_infty * np.ones(NUM_CELLS_Y, 'd')
    # Calculate outlet density (rho_out)
    gv.rho_out = gv.rho[-1, :] * np.power((gv.p_out[:] / gv.p[-1, :]), 1/HEAT_CAPACITY_RATIO)
    # Calculate outlet temperature (T_out)
    gv.T_out = gv.p_out/(gv.rho_out * GAS_CONSTANT)
    # Calculate speed of sound at outlet (c_out)
    gv.c_out = np.sqrt(HEAT_CAPACITY_RATIO * GAS_CONSTANT * gv.T_out)
    # Calculate streamwise velocity at outlet (u_out)
    gv.u_out = gv.u[-1, :, 0] + 2 / (HEAT_CAPACITY_RATIO - 1) * (gv.c[-1, :] - gv.c_out)
    # Set y-velocity (v_out) equal to the adjacent cell's value
    v_out = gv.u[-1, :, 1]
    # Calculate outlet enthalpy (H_out)
    gv.H_out = SPECIFIC_HEAT_CP * gv.T_out + 0.5 * (np.power(gv.u_out, 2) + np.power(v_out, 2))
    # Calculate oulet mach number (M_out)
    gv.M_out = np.sqrt(np.power(gv.u_out,2) + np.power(v_out,2)) / gv.c_out

    # East flux (fe) at the outlet cell (i = NUM_CELLS_X-1, j)
    gv.F[-1, :, 1, 0] = gv.rho_out * gv.u_out * gv.ndS[-1, :, 1, 0]
    gv.F[-1, :, 1, 1] = (gv.rho_out * np.power(gv.u_out, 2) + gv.p_out) * gv.ndS[-1, :, 1, 0]
    gv.F[-1, :, 1, 2] = gv.rho_out * gv.u_out * v_out * gv.ndS[-1, :, 1, 0]
    gv.F[-1, :, 1, 3] = gv.rho_out * gv.u_out * gv.H_out * gv.ndS[-1, :, 1, 0]

    return None