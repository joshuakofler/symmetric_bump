# TODO: Done

from constants import *
import global_vars as gv
import numpy as np
import calculate_artificial_dissipation as ad

def update_flux(state_vector):
    # Update the flux vector components (f) in the x-direction
    gv.f[:, :, 0] = gv.rho[:,:] * gv.u[:,:,0]                 # Mass flux (density)
    gv.f[:, :, 1] = gv.rho[:,:] * np.power(gv.u[:,:,0],2) + gv.p[:,:]     # Momentum flux in x-direction
    gv.f[:, :, 2] = gv.rho[:,:] * gv.u[:,:,0] * gv.u[:,:,1]      # Momentum flux in y-direction
    gv.f[:, :, 3] = gv.rho[:,:] * gv.u[:,:,0] * gv.H[:,:]        # Energy flux in x-direction

    # Update the flux vector components (g) in the y-direction
    gv.g[:, :, 0] = gv.rho[:,:] * gv.u[:,:,1]                 # Mass flux (density)
    gv.g[:, :, 1] = gv.rho[:,:] * gv.u[:,:,0] * gv.u[:,:,1]      # Momentum flux in x-direction
    gv.g[:, :, 2] = gv.rho[:,:] * np.power(gv.u[:,:,1], 2) + gv.p[:,:]     # Momentum flux in y-direction
    gv.g[:, :, 3] = gv.rho[:,:] * gv.u[:,:,1] * gv.H[:,:]        # Energy flux in y-direction

    # # Calculate the interior fluxes at the east face (index 1) using vectorized operations.
    # # For each cell, the east face flux is computed as the average of values from 
    # # the current cell and the adjacent cell to the east in the x-direction.
    # # These values are scaled by the normal vector components at the east face.

    # # f[:-1,:,:] selects all except the last in the x-direction (current cells).
    # # f[:,1:,:] selects all except the first in the x-direction (east neighbors).
    # # The two are averaged to calculate the flux contribution from f at the east face.
    # # Similarly, g[:,:-1,:] and g[:,1:,:] contribute the flux from g.
    # # ndS[:,:-1,1,0] and ndS[:,:-1,1,1] are the normal vector components at the east face.
    # gv.F[:-1,:,1,:] = (0.5 * (f[:-1,:,:] + f[1:,:,:]) * gv.ndS[:-1,:,1,0][..., np.newaxis]
    #                  + 0.5 * (g[:-1,:,:] + g[1:,:,:]) * gv.ndS[:-1,:,1,1][..., np.newaxis])

    # # Assign the east face flux to the west face (index 3) of the neighboring cell.
    # # This ensures continuity of fluxes across the east-west interface between cells.
    # # The flux is negated because the normal vectors for the east and west faces
    # # point in opposite directions.
    # gv.F[1:,:,3,:] = -gv.F[:-1,:,1,:]
    
    # # Calculate the interior fluxes at the north face (index 2) using vectorized operations.
    # # For each cell, the north face flux is computed as the average of values from 
    # # the current cell and the adjacent cell to the north in the y-direction.
    # # These values are scaled by the normal vector components at the north face.

    # # f[:,:-1,:] selects all rows except the last in the y-direction (current cells).
    # # f[:,1:,:] selects all rows except the first in the y-direction (north neighbors).
    # # The two are averaged to calculate the flux contribution from f at the north face.
    # # Similarly, g[:,:-1,:] and g[:,1:,:] contribute the flux from g.
    # # ndS[:,:-1,2,0] and ndS[:,:-1,2,1] are the normal vector components at the north face.
    # gv.F[:,:-1,2,:] = (0.5 * (f[:,:-1,:] + f[:,1:,:]) * gv.ndS[:,:-1,2,0][..., np.newaxis] 
    #                  + 0.5 * (g[:,:-1,:] + g[:,1:,:]) * gv.ndS[:,:-1,2,1][..., np.newaxis])

    # # Assign the north face flux to the south face (index 0) of the neighboring cell.
    # # This ensures continuity of fluxes across the north-south interface between cells.
    # # The flux is negated because the normal vectors for the north and south faces
    # # point in opposite directions.
    # gv.F[:,1:,0,:] = -gv.F[:,:-1,2,:]

    # The code blocks below are equivalent to the above vectorized implementation,
    # but uses explicit looping over all cells to calculate fluxes.

    # Calculate interior fluxes at the east/west face using a loop
    for i, j in np.ndindex(NUM_CELLS_X-1, NUM_CELLS_Y):
        # East face (index 1) flux calculation
        # Average the values of f and g between adjacent cells and scale by
        # the normal vector components at the east face.
        gv.F[i,j,1,:] = (0.5 * (gv.f[i,j,:] + gv.f[i+1,j,:]) * gv.ndS[i,j,1,0])
                       #+ 0.5 * (gv.g[i,j,:] + gv.g[i+1,j,:]) * gv.ndS[i,j,1,1])

        # Copy the calculated flux directly to the west face (index 3) of the neighbor cell.
        gv.F[i+1,j,3,:] = -gv.F[i,j,1,:]
    # Calculate the interior fluxes at the north face using a loop
    for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y-1):
        # Calculate the north face (index 2) flux between the current cell (i, j)
        # and the adjacent cell to the north (i, j+1).
        gv.F[i,j,2,:] = (0.5 * (gv.f[i,j,:] + gv.f[i,j+1,:]) * gv.ndS[i,j,2,0] 
                       + 0.5 * (gv.g[i,j,:] + gv.g[i,j+1,:]) * gv.ndS[i,j,2,1])
    
        # Copy the calculated north face flux to the south face (index 0)
        # of the adjacent cell (i, j+1), with a negation to account for
        # the direction of the normal vector.
        gv.F[i,j+1,0,:] = -gv.F[i,j,2,:]

    #######################################################################
    # Boundary treatment
    #######################################################################
    
    # North flux (gn) at the top wall (i, j = NUM_CELLS_Y-1)
    gv.F[:, -1, 2, 0] = 0.0
    gv.F[:, -1, 2, 1] = 0.0
    gv.F[:, -1, 2, 2] = gv.p[:, -1] * gv.ndS[:, -1, 2, 1]
    gv.F[:, -1, 2, 3] = 0.0

    # South flux (fs, gs) at the bottom wall (i, j = 0)
    gv.F[:, 0, 0, 0] = 0.0
    gv.F[:, 0, 0, 1] = gv.p[:, 0] * gv.ndS[:, 0, 0, 0]
    gv.F[:, 0, 0, 2] = gv.p[:, 0] * gv.ndS[:, 0, 0, 1]
    gv.F[:, 0, 0, 3] = 0.0
    
    # West flux (fw) at the inlet cell (i = 0, j)
    # Set y-velocity component at inlet to 0
    # v_in = 0
    # Calculate speed of sound at inlet (c_in)
    c_in = (HEAT_CAPACITY_RATIO - 1) / 4 * (gv.u_infty - gv.u[0,:,0]) + 0.5 * (gv.c_infty + gv.c[0,:])
    # Calculate inlet velocity (u_in)
    u_in = gv.u[0,:,0] + 2 / (HEAT_CAPACITY_RATIO - 1) * (c_in - gv.c[0,:])
    # Calculate inlet temperature (T_in)
    T_in = c_in**2 / (HEAT_CAPACITY_RATIO * GAS_CONSTANT)
    # Calculate inlet density (rho_in)
    rho_in = np.power(GAS_CONSTANT * T_in * gv.rho_infty**HEAT_CAPACITY_RATIO / gv.p_infty, 1/(HEAT_CAPACITY_RATIO-1))
    # Calculate inlet pressure (p_in)
    p_in = rho_in * GAS_CONSTANT * T_in
    # Calculate inlet enthalpy (H_in)
    H_in = SPECIFIC_HEAT_CP * T_in + 0.5 * (np.power(u_in, 2)) # + np.power(v_in,2))

    # West flux (fw) at the inlet cell (i = 0, j)
    gv.F[0,:,3,0] = rho_in * u_in * gv.ndS[0, :, 3, 0]                          # Mass flux (density)
    gv.F[0,:,3,1] = (rho_in * np.power(u_in, 2) + p_in) * gv.ndS[0, :, 3, 0]    # Momentum flux in x-direction
    gv.F[0,:,3,2] = 0 # rho_in * u_in * v_in * gv.ndS[0, :, 3, 0]               # Momentum flux in y-direction
    gv.F[0,:,3,3] = rho_in * u_in * H_in * gv.ndS[0, :, 3, 0]                   # Energy flux in x-direction

    # East flux (fe) at the outlet cell (i = NUM_CELLS_X-1, j)
    # Set outlet pressure (p_out) to atmospheric pressure
    p_out = ATMOSPHERIC_PRESSURE * np.ones(NUM_CELLS_Y, 'd')
    #p_out = gv.p[-1, :]
    # Calculate outlet density (rho_out)
    rho_out = gv.rho[-1, :] * np.power((p_out[:] / gv.p[-1, :]), 1/HEAT_CAPACITY_RATIO)
    # Calculate outlet temperature (T_out)
    T_out = p_out/(rho_out * GAS_CONSTANT)
    # Calculate speed of sound at outlet (c_out)
    c_out = np.sqrt(HEAT_CAPACITY_RATIO * GAS_CONSTANT * T_out)
    # Calculate streamwise velocity at outlet (u_out)
    u_out = gv.u[-1, :, 0] + 2 / (HEAT_CAPACITY_RATIO - 1) * (gv.c[-1, :] - c_out)
    # Set y-velocity (v_out) equal to the adjacent cell's value
    v_out = gv.u[-1, :, 1]
    # Calculate outlet enthalpy (H_out)
    H_out = SPECIFIC_HEAT_CP * T_out + 0.5 * (np.power(u_out, 2) + np.power(v_out, 2))

    # East flux (fe) at the outlet cell (i = NUM_CELLS_X-1, j)
    gv.F[-1,:,1,0] = rho_out * u_out * gv.ndS[-1, :, 1, 0]                          # Mass flux (density)
    gv.F[-1,:,1,1] = (rho_out * np.power(u_out, 2) + p_out) * gv.ndS[-1, :, 1, 0]   # Momentum flux in x-direction
    gv.F[-1,:,1,2] = rho_out * u_out * v_out * gv.ndS[-1, :, 1, 0]                  # Momentum flux in y-direction
    gv.F[-1,:,1,3] = rho_out * u_out * H_out * gv.ndS[-1, :, 1, 0]                  # Energy flux in x-direction

    #######################################################################
    # Add artificial dissipation
    #######################################################################

    # Add numerical flux corrections to all faces
    ad.update_artificial_dissipation(state_vector)

    # Compute corrected flux for the east face
    gv.F_corrected[:,:,1,:] = gv.F[:,:,1,:] - gv.artificial_dissipation[:,:,1]  # East face (index 1)

    # Compute corrected flux for the west face
    gv.F_corrected[:,:,3,:] = gv.F[:,:,3,:] + gv.artificial_dissipation[:,:,3]  # West face (index 3)

    # Compute corrected flux for the north face
    gv.F_corrected[:,:,2,:] = gv.F[:,:,2,:] - gv.artificial_dissipation[:,:,2]  # North face (index 2)

    # Compute corrected flux for the south face
    gv.F_corrected[:,:,0,:] = gv.F[:,:,0,:] + gv.artificial_dissipation[:,:,0]  # South face (index 0)

    return None