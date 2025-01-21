# import modules
from constants import *
import global_vars as gv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import mesh

def Massflow():
    fig2, ax2 = plt.subplots()
    
    iter = np.linspace(1,gv.iteration,gv.iteration)

    ax2.plot(iter, gv.m_in[:])
    ax2.plot(iter, gv.m_out[:])

    return None

def Mach_number():
    fig, ax = plt.subplots()

    #######################################################################
    # Plot parameters
    #######################################################################
    show_horizontal_grid = True
    show_vertical_grid = True
    show_face_points = True
    show_center_points = True
    show_diagonal_lines = False
    show_cell_area = False
    
    show_normal_vector = False
    show_normal_vector_south = True
    show_normal_vector_north = True
    show_normal_vector_west = True
    show_normal_vector_east = True

    x_axis_limits = [-DOMAIN_LENGTH-gv.cell_dx/2, 2*DOMAIN_LENGTH+gv.cell_dx/2]
    y_axis_limits = [0-gv.cell_dy[0]/2, CHANNEL_HEIGHT+gv.cell_dy[0]/2]

    save_grid_plot = False
    if save_grid_plot:
        filename = "../img/Mach_number.pdf"
    #######################################################################
    # Plot the Grid
    #######################################################################
    
    if show_vertical_grid:
        # Plot vertical grid lines at each y-face.
        for i, j in np.ndindex(NUM_CELLS_X, NUM_FACES_Y):
            ax.plot(
                [gv.face_x_coords[i], gv.face_x_coords[i+1]],  # x-coordinates for the vertical line
                [gv.face_y_coords[i, j], gv.face_y_coords[i+1, j]],  # y-coordinates for the line at this vertical slice
                '--', color="gray", linewidth=0.5  # Dashed gray line with thin width
            )

    if show_horizontal_grid:
        # Plot horizontal grid lines at each x-face.
        for i, j in np.ndindex(NUM_FACES_X, NUM_CELLS_Y):
            ax.plot(
                [gv.face_x_coords[i], gv.face_x_coords[i]],  # x-coordinate remains constant (vertical face)
                [gv.face_y_coords[i, j], gv.face_y_coords[i, j+1]],  # Connect y-coordinates for this x-face
                '--', color="gray", linewidth=0.5  # Dashed gray line
            )

    # Compute Mach number
    u_mag = 0.5 * (gv.u[:, :, 0]**2 + gv.u[:, :, 1]**2)
    M = u_mag / gv.c

    # Normalize for colormap
    norm = Normalize(vmin=M.min(), vmax=M.max())
    cmap = plt.cm.viridis

    for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y):
        # Get cell vertices
        vertices = [
            mesh.get_vertex_coordinates(i, j, 'A'),
            mesh.get_vertex_coordinates(i, j, 'B'),
            mesh.get_vertex_coordinates(i, j, 'C'),
            mesh.get_vertex_coordinates(i, j, 'D')
        ]

        # Compute color for the cell
        color = cmap(norm(M[i, j]))

        # Plot the cell as a polygon
        polygon = plt.Polygon(vertices, facecolor=color, edgecolor='k')
        ax.add_patch(polygon)

    # Add colorbar
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cb = plt.colorbar(mappable, ax=ax, orientation='vertical')
    cb.set_label("Mach number")

    ax.set_xlim(x_axis_limits)
    ax.set_ylim(y_axis_limits)

    ax.set_title("Mesh")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    if save_grid_plot:
        fig.savefig(filename)

    return None