"""
This module provides functions to plot convergence history, cell data, face data, and the Cp distribution.

- `plot_convergence_history`: Plots the convergence of the inlet and outlet mass flow rates over the simulation iterations.
- `plot_cell_data`: Visualizes fluid properties computed at the cell center, such as Mach number or pressure, on the computational grid.
- `plot_face_data`: Visualizes fluid properties INTERPOLATED to the cell faces, allowing for analysis of values at the face centers.
- `plot_Cp`: Plots the Cp distribution at the bottom wall.

Each function includes options to customize plot appearance, save the figure, and set axis labels and titles. 
The plots are saved as PDF files to the appropriate directory for each iteration.

Notes:
    - The functions rely on global variables from `global_vars` and other imported modules to handle the grid and simulation data.
    - Plots can be saved to a specific directory defined by the `parent_directory` argument.
    - The plotting functions provide options for customizing the display, including grid visibility and color normalization.
"""

from calendar import c
import numpy as np
from constants import *
import global_vars as gv

# Import matplotlib libraries for plotting
from matplotlib import pyplot as plt  # Import the PyPlot interface for creating plots
from matplotlib.cm import ScalarMappable  # For mapping scalar data to colors
from matplotlib.colors import Normalize  # To normalize data for colormap scaling
from matplotlib.ticker import MaxNLocator  # For controlling tick locations on axes

# Import standard library for file and directory operations
import os  # For managing directories and file paths

# Import own libraries for cell and mesh calculations
import cell
import mesh

# Import data input/output library
import data_io as io

def plot_convergence_history(fig, iteration_dir=OUTPUT_DIR, save_plot=False):
    """
    Plots the convergence history of the inlet and outlet mass flow.
    Saves the plot as a PDF file in the specified directory.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object where the plot will be drawn.
    parent_directory : str, optional
        The directory where the plot will be saved. Defaults to OUTPUT_DIR.
    
    Returns:
    --------
    None
    """
    # Get current axis
    ax = fig.gca()
    #----------------------------------------------------------------------------------

    if save_plot:        
        # Create a writer to save the data to a file
        file_path = os.path.join(iteration_dir,  f"{gv.iteration}_convergence_history.pdf")
    
    #----------------------------------------------------------------------------------

    # Prepare iteration range
    iterations = np.arange(0, gv.iteration + 1)
    
    # Plot inlet and outlet mass flow
    ax.plot(iterations, gv.m_in[:], 'k', label=r"Inlet mass flow")
    ax.plot(iterations, gv.m_out[:], '--k', label=r"Outlet mass flow")
    
    # Set axis limits and labels
    ax.set_xlim([1, gv.iteration])
    ax.set_xlabel("Iteration Number", fontsize=10)
    ax.set_ylabel("Mass Flow [kg/m/s]", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Set x-ticks to be integers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add title, grid, and legend
    ax.set_title("Convergence History of the Inlet and Outlet Mass Flow", fontsize=12, pad=10)
    ax.grid(visible=True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.legend(loc="lower right", fontsize=10)
    
    if save_plot:
        # Save the plot to a file
        fig.savefig(file_path, bbox_inches="tight")
        print(f"Convergence history plot for iteration {gv.iteration} saved at: {file_path}")

    return None

def plot_cell_data(fig, fluid_property, property_name, acronym = "DEFAULT", iteration_dir=OUTPUT_DIR, save_plot=False):
    """
    Plots the distribution of a specified fluid property (e.g., temperature, pressure, etc.)
    over the computational domain, and optionally saves the plot to a PDF file.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object where the plot will be drawn.
    fluid_property : array-like
        A 2D array containing the fluid property data to plot.
    property_name : str
        The name of the fluid property being plotted (e.g., 'Temperature').
    acronym : str, optional
        An acronym to be used in the filename when saving the plot. Default is "DEFAULT".
    iteration_dir : str, optional
        The directory where the plot will be saved. Defaults to OUTPUT_DIR.
    save_plot : bool, optional
        Whether to save the plot as a PDF. Default is False.

    Returns:
    --------
    None
    """
    # Get current axis
    ax = fig.gca()
    #----------------------------------------------------------------------------------
    # Convert fluid property data to a numpy array
    data = np.array(fluid_property)
    
    # Flags to edit figure
    show_vertical_grid = False
    show_horizontal_grid = False

    show_title = True
    show_incident_mach_number = True

    # Choose colormap
    cmap = plt.cm.jet

    # Set axis limits
    x_axis_limits = [-DOMAIN_LENGTH, 2*DOMAIN_LENGTH]
    y_axis_limits = [0, CHANNEL_HEIGHT]

    if save_plot:        
        # Format with two decimal places, replace '.' with '_'
        formatted_mach_number = f"M_{gv.M[0,:].mean():.2f}".replace(".", "_")

        # Create a writer to save the data to a file
        file_path = os.path.join(iteration_dir, f"{gv.iteration}_F_{formatted_mach_number}_{acronym}_{NUM_CELLS_X}_{NUM_CELLS_Y}.pdf")
    
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
    if show_title:
        ax.set_title(f"Distribution of {property_name} computed on a ({NUM_CELLS_X}x{NUM_CELLS_Y}) mesh (cell center values)", fontsize=20, pad=32)
        ax.text(0.5, 1.02, f"Iteration: {gv.iteration}, Max: {data.max():.3g}, Min: {data.min():.3g}", transform=ax.transAxes, fontsize=18, ha='center')
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$y$", fontsize=18)

    if show_incident_mach_number:
        ax.text(
            0.03,
            0.9,
            (r"$M_{\infty} = $" + f"{gv.M[0,:].mean():.2f}"),
            fontsize = 22,
            color="black",
            bbox=dict(facecolor="white", alpha=0.4, edgecolor="gray", boxstyle="round,pad=0.35"),
            horizontalalignment = 'left',
            verticalalignment = 'center',
            transform = ax.transAxes
        )

    if save_plot:
        # Save the plot to a file
        fig.savefig(file_path, bbox_inches="tight")
        print(f"Plot saved at: {file_path}")

    return None

def plot_face_data(fig, fluid_property, property_name, acronym = "DEFAULT", iteration_dir=OUTPUT_DIR, save_plot=False):
    """
    Plots the distribution of a specified fluid property (e.g., temperature, pressure, etc.)
    over the computational domain, and optionally saves the plot to a PDF file.

    Interpolates the cell values to the faces.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object where the plot will be drawn.
    fluid_property : array-like
        A 2D array containing the fluid property data to plot.
    property_name : str
        The name of the fluid property being plotted (e.g., 'Temperature').
    acronym : str, optional
        An acronym to be used in the filename when saving the plot. Default is "DEFAULT".
    iteration_dir : str, optional
        The directory where the plot will be saved. Defaults to OUTPUT_DIR.
    save_plot : bool, optional
        Whether to save the plot as a PDF. Default is False.

    Returns:
    --------
    None
    """
    # Get current axis
    ax = fig.gca()
    #----------------------------------------------------------------------------------
    # Flags to edit figure
    show_vertical_grid = False
    show_horizontal_grid = False

    show_title = True
    show_incident_mach_number = True

    # Choose colormap
    cmap = plt.cm.jet

    # Set axis limits
    x_axis_limits = [-DOMAIN_LENGTH, 2*DOMAIN_LENGTH]
    y_axis_limits = [0, CHANNEL_HEIGHT]

    if save_plot:        
        # Format with two decimal places, replace '.' with '_'
        formatted_mach_number = f"M_{gv.M[0,:].mean():.2f}".replace(".", "_")

        # Create a writer to save the data to a file
        file_path = os.path.join(iteration_dir, f"{gv.iteration}_F_{formatted_mach_number}_{acronym}_{NUM_CELLS_X}_{NUM_CELLS_Y}.pdf")
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
    if show_title:
        ax.set_title(f"Distribution of {property_name} computed on a ({NUM_CELLS_X}x{NUM_CELLS_Y}) mesh (interpolated to vertex points)", fontsize=20, pad=32)
        ax.text(0.5, 1.02, f"Iteration: {gv.iteration}, Max: {data.max():.3g}, Min: {data.min():.3g}", transform=ax.transAxes, fontsize=18, ha='center')
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$y$", fontsize=18)

    if show_incident_mach_number:
        ax.text(
            0.03,
            0.9,
            (r"$M = $" + f"{gv.M[0,:].mean():.2f}"),
            fontsize = 22,
            color="black",
            bbox=dict(facecolor="white", alpha=0.4, edgecolor="gray", boxstyle="round,pad=0.35"),
            horizontalalignment = 'left',
            verticalalignment = 'center',
            transform = ax.transAxes
        )

    if save_plot:
        # Save the plot to a file
        fig.savefig(file_path, bbox_inches="tight")
        print(f"Plot saved at: {file_path}")
    return None

def plot_Cp(fig, iteration_dir=OUTPUT_DIR, save_plot=False):
    """
    Plots the pressure coefficient (Cp) on the bottom wall and optionally saves the plot to a PDF file.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object where the plot will be drawn.
    parent_directory : str, optional
        The directory where the plot will be saved. Defaults to OUTPUT_DIR.
    save_plot : bool, optional
        Whether to save the plot as a PDF. Default is False.

    Returns:
    --------
    None
    """
    # Get current axis
    ax = fig.gca()
    #----------------------------------------------------------------------------------

    show_grid = True
    show_title = True

    # Axis limits and x-values
    x_axis_limits = [-DOMAIN_LENGTH, 2 * DOMAIN_LENGTH]
    x = np.linspace(-DOMAIN_LENGTH, 2 * DOMAIN_LENGTH, NUM_CELLS_X)

    y_axis_limits = [-0.6, 0.3]
    y_axis_ticks = [-0.6, -0.3, 0, 0.3]

    if save_plot:        
        # Create a writer to save the data to a file
        file_path = os.path.join(iteration_dir, f"{gv.iteration}_Cp_{NUM_CELLS_X}_{NUM_CELLS_Y}.pdf")
    
    export_in_csv = False

    #----------------------------------------------------------------------------------

    # Calculate the pressure coefficient along the bottom wall
    Cp = cell.calculate_pressure_coefficient()

    # Plot Cp with styling
    ax.plot(x, Cp, linestyle=':', color='gray', linewidth=0.5,
            marker='o', markerfacecolor='black', markeredgecolor='black')

    # Set axis limits and labels
    ax.set_xlim(x_axis_limits)
    ax.set_ylim(y_axis_limits)
    ax.set_yticks(y_axis_ticks)

    ax.set_xlabel(r"$x$", fontsize=10)
    ax.set_ylabel(r"$C_p$", fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Add title and grid
    if show_title:
        ax.set_title(r"Pressure Coefficient $C_p$ on the Bottom Wall", fontsize=12, pad=10)
    if show_grid:
        ax.grid(visible=True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    
    if save_plot:
        # Save the plot to a file
        fig.savefig(file_path, bbox_inches="tight")
        print(f"Pressure coefficient plot for iteration {gv.iteration} saved at: {file_path}")

    if export_in_csv:
        # Save the calculated data as CSV
        io.save_two_arrays_in_csv(x, Cp, 'x', 'Cp', iteration_dir)

    return None