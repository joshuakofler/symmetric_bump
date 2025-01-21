# TODO: Done

from global_var import *
import numpy as np
import calculate_artificial_dissipation as ad

def update_flux():
    # Update the flux vector components (f) in the x-direction
    f[:, :, 0] = rho[:,:] * u[:,:,0]                 # Mass flux (density)
    f[:, :, 1] = rho[:,:] * u[:,:,0]**2 + p[:,:]     # Momentum flux in x-direction
    f[:, :, 2] = rho[:,:] * u[:,:,0] * u[:,:,1]      # Momentum flux in y-direction
    f[:, :, 3] = rho[:,:] * u[:,:,0] * H[:,:]        # Energy flux in x-direction

    # Update the flux vector components (g) in the y-direction
    g[:, :, 0] = rho[:,:] * u[:,:,1]                 # Mass flux (density)
    g[:, :, 1] = rho[:,:] * u[:,:,0] * u[:,:,1]      # Momentum flux in x-direction
    g[:, :, 2] = rho[:,:] * u[:,:,1]**2 + p[:,:]     # Momentum flux in y-direction
    g[:, :, 3] = rho[:,:] * u[:,:,1] * H[:,:]        # Energy flux in y-direction

    # Calculate the interior fluxes at the east face (index 1) using vectorized operations.
    # For each cell, the east face flux is computed as the average of values from 
    # the current cell and the adjacent cell to the east in the x-direction.
    # These values are scaled by the normal vector components at the east face.

    # f[:-1,:,:] selects all except the last in the x-direction (current cells).
    # f[:,1:,:] selects all except the first in the x-direction (east neighbors).
    # The two are averaged to calculate the flux contribution from f at the east face.
    # Similarly, g[:,:-1,:] and g[:,1:,:] contribute the flux from g.
    # ndS[:,:-1,1,0] and ndS[:,:-1,1,1] are the normal vector components at the east face.
    F[:-1,:,1,:] = (0.5 * (f[:-1,:,:] + f[1:,:,:]) * ndS[:-1,:,1,0][..., np.newaxis]
                + 0.5 * (g[:-1,:,:] + g[1:,:,:]) * ndS[:-1,:,1,1][..., np.newaxis])

    # Assign the east face flux to the west face (index 3) of the neighboring cell.
    # This ensures continuity of fluxes across the east-west interface between cells.
    # The flux is negated because the normal vectors for the east and west faces
    # point in opposite directions.
    F[1:,:,3,:] = -F[:-1,:,1,:]
    
    # Calculate the interior fluxes at the north face (index 2) using vectorized operations.
    # For each cell, the north face flux is computed as the average of values from 
    # the current cell and the adjacent cell to the north in the y-direction.
    # These values are scaled by the normal vector components at the north face.

    # f[:,:-1,:] selects all rows except the last in the y-direction (current cells).
    # f[:,1:,:] selects all rows except the first in the y-direction (north neighbors).
    # The two are averaged to calculate the flux contribution from f at the north face.
    # Similarly, g[:,:-1,:] and g[:,1:,:] contribute the flux from g.
    # ndS[:,:-1,2,0] and ndS[:,:-1,2,1] are the normal vector components at the north face.
    F[:,:-1,2,:] = (0.5 * (f[:,:-1,:] + f[:,1:,:]) * ndS[:,:-1,2,0][..., np.newaxis] 
                    + 0.5 * (g[:,:-1,:] + g[:,1:,:]) * ndS[:,:-1,2,1][..., np.newaxis])

    # Assign the north face flux to the south face (index 0) of the neighboring cell.
    # This ensures continuity of fluxes across the north-south interface between cells.
    # The flux is negated because the normal vectors for the north and south faces
    # point in opposite directions.
    F[:,1:,0,:] = -F[:,:-1,2,:]

    # The code blocks below are equivalent to the above vectorized implementation,
    # but uses explicit looping over all cells to calculate fluxes.

    # # Calculate interior fluxes at the east/west face using a loop
    # for i, j in np.ndindex(NUM_CELLS_X-1, NUM_CELLS_Y):
    #     # East face (index 1) flux calculation
    #     # Average the values of f and g between adjacent cells and scale by
    #     # the normal vector components at the east face.
    #     F[i,j,1,:] = (0.5 * (f[i,j,:] + f[i+1,j,:]) * ndS[i,j,1][0]
    #                 + 0.5 * (g[i,j,:] + g[i+1,j,:]) * ndS[i,j,1][1])

    #     # Copy the calculated flux directly to the west face (index 3) of the neighbor cell.
    #     F[i+1,j,3,:] = -F[i,j,1,:]
    # # Calculate the interior fluxes at the north face using a loop
    # for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y-1):
    #     # Calculate the north face (index 2) flux between the current cell (i, j)
    #     # and the adjacent cell to the north (i, j+1).
    #     F[i,j,2,:] = (0.5 * (f[i,j,:] + f[i,j+1,:]) * ndS[i,j,2][0] 
    #                 + 0.5 * (g[i,j,:] + g[i,j+1,:]) * ndS[i,j,2][1])
    #
    #     # Copy the calculated north face flux to the south face (index 0)
    #     # of the adjacent cell (i, j+1), with a negation to account for
    #     # the direction of the normal vector.
    #     F[i,j+1,0,:] = -F[i,j,2,:]

    #######################################################################
    # TODO
    #######################################################################

    # calculate the fluxes at the boundarys

    # north flux at top wall
    # v = 0
    # (f[0] * (yC - yB) - g[0] * (xC - xB)) where yC-yB = 0 and g[0] = 0
    F[:, -1, 2, 0] = 0.0
    F[:, -1, 2, 1] = 0.0
    F[:, -1, 2, 2] = p[:, -1] * ndS[:, -1, 2, 1]
    F[:, -1, 2, 3] = 0.0
    
    # south flux at bottom wall
    # vn = 0

    F[:, 0, 0, 0] = 0.0
    F[:, 0, 0, 1] = p[:, 0] * ndS[:, 0, 0, 0]
    F[:, 0, 0, 2] = p[:, 0] * ndS[:, 0, 0, 1]
    F[:, 0, 0, 3] = 0.0
    
    # west flux at inlet
    v_in = 0

    c_in = (HEAT_CAPACITY_RATIO - 1) / 2 * (u_infty - u[0,:,0]) + c_infty - c[0,:]

    u_in = u[0,:,0] + 2 / (HEAT_CAPACITY_RATIO - 1) * (c[0,:] + c_infty)

    T_in = c_in**2 / (HEAT_CAPACITY_RATIO * GAS_CONSTANT)

    rho_in = np.power( rho_infty**HEAT_CAPACITY_RATIO * GAS_CONSTANT * T_in / ATMOSPHERIC_PRESSURE ,1/(HEAT_CAPACITY_RATIO-1))

    p_in = rho_in * GAS_CONSTANT * T_in

    H_in = SPECIFIC_HEAT_CP * T_in + 0.5 * (u_in**2 + v_in**2)

    F[0,:,3,0] = rho_in * u_in * ndS[0, :, 3, 0]               # Mass flux (density)
    F[0,:,3,1] = (rho_in * u_in**2 + p_in) * ndS[0, :, 3, 0]   # Momentum flux in x-direction
    F[0,:,3,2] = rho_in * u_in * v_in * ndS[0, :, 3, 0]        # Momentum flux in y-direction
    F[0,:,3,3] = rho_in * u_in * H_in * ndS[0, :, 3, 0]        # Energy flux in x-direction

    # east flux at outlet
    p_out = ATMOSPHERIC_PRESSURE * np.ones(NUM_CELLS_Y, 'd')
    
    rho_out = rho[-1, :] * np.power((p_out[:] / p[-1, :]), 1/HEAT_CAPACITY_RATIO)

    T_out = p_out/(rho_out * GAS_CONSTANT)

    c_out = np.sqrt(HEAT_CAPACITY_RATIO * GAS_CONSTANT * T_out)

    u_out = u[-1, :, 0] + 2 / (HEAT_CAPACITY_RATIO - 1) * (c[-1, :] - c_out)

    v_out = u[-1, :, 1]

    H_out = SPECIFIC_HEAT_CP * T_out + 0.5 * (u_out**2 + v_out**2)

    F[-1,:,1,0] = rho_out * u_out * ndS[-1, :, 1, 0]                # Mass flux (density)
    F[-1,:,1,1] = (rho_out * u_out**2 + p_out) * ndS[-1, :, 1, 0]   # Momentum flux in x-direction
    F[-1,:,1,2] = rho_out * u_out * v_out * ndS[-1, :, 1, 0]        # Momentum flux in y-direction
    F[-1,:,1,3] = rho_out * u_out * H_out * ndS[-1, :, 1, 0]        # Energy flux in x-direction

    #######################################################################
    # Add artificial dissipation
    #######################################################################

    # Add numerical flux corrections to all faces
    ad.update_artificial_dissipation()

    # Compute corrected flux for the east face
    F_corrected[:,:,1,:] = F[:,:,1,:] - artificial_dissipation[:,:,1]  # East face (index 1)

    # Compute corrected flux for the west face
    F_corrected[:,:,3,:] = F[:,:,3,:] + artificial_dissipation[:,:,3]  # West face (index 3)

    # Compute corrected flux for the north face
    F_corrected[:,:,2,:] = F[:,:,2,:] - artificial_dissipation[:,:,2]  # North face (index 2)

    # Compute corrected flux for the south face
    F_corrected[:,:,0,:] = F[:,:,0,:] + artificial_dissipation[:,:,0]  # South face (index 0)

    return None