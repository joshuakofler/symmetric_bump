# WIP
# TODO: Calculate/Define flux at the boundaries

from globals import *
import numpy as np
import calculate_artificial_dissipation as ad
from mesh import get_normal_vector

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
    F[:, -1, 2, 2] = p[:,-1]
    F[:, -1, 2, 3] = 0.0
    
    # south flux at bottom wall
    # vn = 0
    nx, ny = get_normal_vector(np.arange(0,NUM_CELLS_X,1), 0, 'S')

    F[:, 0, 0, 0] = 0.0
    F[:, 0, 0, 1] = p[:, 0] * nx
    F[:, 0, 0, 2] = p[:, 0] * ny
    F[:, 0, 0, 3] = 0.0
    
    # west flux at inlet
    F[0,:,3,:] = 300

    # east flux at outlet
    F[-1,:,1,:] = 400

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