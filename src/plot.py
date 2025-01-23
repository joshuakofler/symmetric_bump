# import modules
from constants import *
import global_vars as gv
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

import os
# own libs
import cell
import mesh

def plot_convergence_history(fig):
    ax = fig.gca()

    iterations = np.linspace(1, gv.iteration + 1, gv.iteration)

    ax.plot(iterations, gv.m_in[:], 'k', label=r"Inlet mass flow")
    ax.plot(iterations, gv.m_out[:], '--k', label=r"Outlet mass flow")

    ax.set_xlim([1, gv.iteration + 1])

    ax.set_xlabel("Iteration Number", fontsize=10)
    ax.set_ylabel("Mass Flow [kg/m/s]", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Set x-ticks to be only integers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Title and grid
    ax.set_title("Convergence History of the Inlet and Outlet Mass Flow", fontsize=12, pad=10)
    ax.grid(visible=True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)

    # Legend
    ax.legend(loc="lower right", fontsize=10)

    # Save the plot to a file
    file_path = os.path.join(OUTPUT_DIR, f"convergence_history_{gv.iteration}.pdf")
    fig.savefig(file_path, bbox_inches="tight")
    print(f"Plot saved at: {file_path}")

    return None

def plot_cell_data(fig, fluid_property, property_name, acronym = "DEFAULT"):
    ax = fig.gca()
    #----------------------------------------------------------------------------------
    # Convert fluid property data to a numpy array
    data = np.array(fluid_property)
    
    # Flags to show grid lines
    show_vertical_grid = False
    show_horizontal_grid = False

    # Set axis limits
    x_axis_limits = [-DOMAIN_LENGTH, 2*DOMAIN_LENGTH]
    y_axis_limits = [0, CHANNEL_HEIGHT]

    # Flag to save the plot
    save_plot = True
    if save_plot:
        # Create the output directory if it doesn't exist
        iteration_dir = os.path.join(OUTPUT_DIR, str(gv.iteration))
        if not os.path.exists(iteration_dir):
            os.makedirs(iteration_dir)
            print(f"Created directory: {iteration_dir}")
        
        # Create a writer to save the data to a file
        file_path = os.path.join(iteration_dir, f"{gv.iteration}_C_{acronym}_{NUM_CELLS_X}_{NUM_CELLS_Y}.pdf")
        print(f"Saving plot to: {file_path}")
    #----------------------------------------------------------------------------------

    if show_vertical_grid:
        # Plot vertical grid lines at each y-face.
        for i, j in np.ndindex(NUM_CELLS_X, NUM_FACES_Y):
            ax.plot(
                [gv.face_x_coords[i], gv.face_x_coords[i+1]],  # x-coordinates for the vertical line
                [gv.face_y_coords[i, j], gv.face_y_coords[i+1, j]],  # y-coordinates for the line at this vertical slice
                '--', color="gray", linewidth=0.25  # Dashed gray line with thin width
            )
            print(f"Plotted vertical grid line at x={gv.face_x_coords[i]}")

    if show_horizontal_grid:
        # Plot horizontal grid lines at each x-face.
        for i, j in np.ndindex(NUM_FACES_X, NUM_CELLS_Y):
            ax.plot(
                [gv.face_x_coords[i], gv.face_x_coords[i+1]],  # x-coordinates for the horizontal line
                [gv.face_y_coords[i, j], gv.face_y_coords[i, j+1]],  # y-coordinates for the line at this horizontal slice
                '--', color="gray", linewidth=0.25  # Dashed gray line with thin width
            )
            print(f"Plotted horizontal grid line at y={gv.face_y_coords[i, j]}")

    #----------------------------------------------------------------------------------
    # Plot the fluid property data

    # Normalize colormap
    norm = Normalize(vmin=data.min(), vmax=data.max())
    cmap = plt.cm.inferno

    for i, j in np.ndindex(NUM_CELLS_X, NUM_CELLS_Y):
        # Get cell vertices
        vertices = [
            mesh.get_vertex_coordinates(i, j, 'A'),
            mesh.get_vertex_coordinates(i, j, 'B'),
            mesh.get_vertex_coordinates(i, j, 'C'),
            mesh.get_vertex_coordinates(i, j, 'D')
        ]
        # Compute color for the cell
        color = cmap(norm(data[i,j]))

        # Plot the cell as a polygon
        polygon = plt.Polygon(vertices, facecolor=color, ec=color)
        ax.add_patch(polygon)

    # Add colorbar
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cb = plt.colorbar(mappable, ax=ax, orientation='vertical')
    cb.set_label(property_name, fontsize=18)
    cb.ax.tick_params(labelsize=16)

    # Set axis limits
    ax.set_xlim(x_axis_limits)
    ax.set_ylim(y_axis_limits)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Set plot title and labels
    ax.set_title(f"Distribution of {property_name} computed on a ({NUM_CELLS_X}x{NUM_CELLS_Y}) mesh (cell center values)", fontsize=20, pad=32)
    ax.text(0.5, 1.02, f"Iteration: {gv.iteration}, Max: {data.max():.3g}, Min: {data.min():.3g}", transform=ax.transAxes, fontsize=18, ha='center')
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$y$", fontsize=18)

    if save_plot:
        # Save the plot to a file
        fig.savefig(file_path, bbox_inches="tight")
        print(f"Plot saved at: {file_path}")
    return None

def plot_face_data(fig, fluid_property, property_name, acronym = "DEFAULT"):
    ax = fig.gca()
    #----------------------------------------------------------------------------------
    show_vertical_grid = False
    show_horizontal_grid = False

    x_axis_limits = [-DOMAIN_LENGTH, 2*DOMAIN_LENGTH]
    y_axis_limits = [0, CHANNEL_HEIGHT]

    # Flag to save the plot
    save_plot = True
    if save_plot:
        # Create the output directory if it doesn't exist
        iteration_dir = os.path.join(OUTPUT_DIR, str(gv.iteration))
        if not os.path.exists(iteration_dir):
            os.makedirs(iteration_dir)
            print(f"Created directory: {iteration_dir}")
        
        # Create a writer to save the data to a file
        file_path = os.path.join(iteration_dir, f"{gv.iteration}_F_{acronym}_{NUM_CELLS_X}_{NUM_CELLS_Y}.pdf")
        print(f"Saving plot to: {file_path}")
    #----------------------------------------------------------------------------------

    if show_vertical_grid:
        # Plot vertical grid lines at each y-face.
        for i, j in np.ndindex(NUM_CELLS_X, NUM_FACES_Y):
            ax.plot(
                [gv.face_x_coords[i], gv.face_x_coords[i+1]],  # x-coordinates for the vertical line
                [gv.face_y_coords[i, j], gv.face_y_coords[i+1, j]],  # y-coordinates for the line at this vertical slice
                '--', color="gray", linewidth=0.25  # Dashed gray line with thin width
            )

    if show_horizontal_grid:
        # Plot horizontal grid lines at each x-face.
        for i, j in np.ndindex(NUM_FACES_X, NUM_CELLS_Y):
            ax.plot(
                [gv.face_x_coords[i], gv.face_x_coords[i]],  # x-coordinate remains constant (vertical face)
                [gv.face_y_coords[i, j], gv.face_y_coords[i, j+1]],  # Connect y-coordinates for this x-face
                '--', color="gray", linewidth=0.25  # Dashed gray line
            )

    #----------------------------------------------------------------------------------
    # plot the parameter
    data = np.zeros((NUM_CELLS_X-1, NUM_CELLS_Y-1), 'd')

    for i in range(NUM_CELLS_X-1):
        for j in range(NUM_CELLS_Y-1):
            data[i,j] = cell.get_point_data(i, j, fluid_property)

    # Normalize colormap
    norm = Normalize(vmin=data.min(), vmax=data.max())
    cmap = plt.cm.inferno

    for i, j in np.ndindex(NUM_CELLS_X-1, NUM_CELLS_Y-1):
        # Get cell vertices
        vertices = [
            mesh.get_point_coordinates(i+1,j),
            mesh.get_point_coordinates(i+1,j+1),
            mesh.get_point_coordinates(i,j+1),
            mesh.get_point_coordinates(i,j)
        ]
        # Compute color for the cell
        color = cmap(norm(data[i,j]))

        # Plot the cell as a polygon
        polygon = plt.Polygon(vertices, facecolor=color, ec=color)
        ax.add_patch(polygon)

    # Add colorbar
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cb = plt.colorbar(mappable, ax=ax, orientation='vertical')
    cb.set_label(property_name, fontsize=18)
    cb.ax.tick_params(labelsize=16)

    # Set axis limits
    ax.set_xlim(x_axis_limits)
    ax.set_ylim(y_axis_limits)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Set plot title and labels
    ax.set_title(f"Distribution of {property_name} computed on a ({NUM_CELLS_X}x{NUM_CELLS_Y}) mesh (interpolated to vertex points)", fontsize=20, pad=32)
    ax.text(0.5, 1.02, f"Iteration: {gv.iteration}, Max: {data.max():.3g}, Min: {data.min():.3g}", transform=ax.transAxes, fontsize=18, ha='center')
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$y$", fontsize=18)

    if save_plot:
        # Save the plot to a file
        fig.savefig(file_path, bbox_inches="tight")
        print(f"Plot saved at: {file_path}")
    return None