"""
This module manages data input and output for the project.
It handles saving simulation results at designated iteration points, 
tracking progress, and storing key milestones for analysis.
"""

from constants import *
import global_vars as gv
import numpy as np

from datetime import datetime  # Import datetime for timestamping output

import os  # Import os for handling file and directory operations
import glob  # Import glob for finding files using wildcard matching
import re  # Import re for regular expression operations

import vtk  # Import vtk for handling VTK file operations
import csv  # Import csv for reading and writing CSV files

# Set a default width for printing dividers
max_width = 75

def simplify_file_path(file_path, base_path):
    """Simplify the file path by stripping the base path."""
    return file_path.replace(base_path, "").lstrip(os.sep)

def clear_folder_structure():
    """
    Clear the folder structure within the output directory (OUTPUT_DIR).

    This function checks if the base directory (OUTPUT_DIR) exists. If it does, it iterates 
    through all subfolders inside it, deletes the contents of each subfolder, and then removes 
    the subfolder itself. If OUTPUT_DIR does not exist, the function simply prints a message.

    Note:
    - This function is recursive and ensures that all nested directories and files are deleted.
    - It avoids issues with removing non-empty folders by processing directories from the bottom up.

    Returns:
    None
    """
    # Check if the base directory exists
    if os.path.exists(OUTPUT_DIR):
        # Loop through each folder inside the base directory
        for folder in os.listdir(OUTPUT_DIR):
            # Get the full path to the current folder
            folder_path = os.path.join(OUTPUT_DIR, folder)
            
            # Check if the path corresponds to a directory
            if os.path.isdir(folder_path):
                # Walk through the directory tree in reverse (bottom-up)
                for root, dirs, files in os.walk(folder_path, topdown=False):
                    # Delete all files in the current folder
                    for file in files:
                        file_path = os.path.join(root, file)
                        os.remove(file_path)  # Remove the file
                        print(f"Deleted file: {file_path}")

                    # Delete all subdirectories in the current folder
                    for subdir in dirs:
                        subdir_path = os.path.join(root, subdir)
                        os.rmdir(subdir_path)  # Remove the empty subdirectory
                        print(f"Deleted subdirectory: {subdir_path}")
                
                # Remove the now-empty top-level folder
                os.rmdir(folder_path)
                print(f"Deleted folder: {folder_path}")
    else:
        # Print a message if the base directory does not exist
        print(f"[INFO] The directory {OUTPUT_DIR} does not exist.")

    # Print a blank line for spacing in the console
    print("\n")
    return None

def initialize_folder_structure():
    """
    Initialize the folder structure for storing simulation outputs.

    This function creates a dedicated folder for each iteration specified in the
    global variable `gv.output_iterations`. The folders are created inside the 
    simulation directory (`gv.sim_dir`), and a confirmation message is printed 
    for each folder created. The path displayed in the message is simplified 
    to show only the relevant portion relative to the project directory.

    Returns:
    None
    """
    global max_width

    # Print the header for folder initialization
    print("~" * max_width)
    print("  -- Initialize Folder Structure --  ")
    print("~" * max_width)

    # Create folders for each iteration
    for iteration in gv.output_iterations:
        # Construct the folder name based on the iteration number
        folder_name = os.path.join(gv.sim_dir, f"{int(iteration)}")
        
        # Create the folder (if it doesn't already exist)
        os.makedirs(folder_name, exist_ok=True)

        # Simplify the file path for cleaner output display
        simplified_file_path = simplify_file_path(folder_name, str(PROJECT_DIR))

        # Print a confirmation message for the created folder
        print(f"Created folder: .{os.sep}{simplified_file_path}")

    # Print a closing divider line
    print("~" * max_width)
    
    return None

def save_iteration(iteration):
    # Create VTK Points for the grid with face coordinates
    vtk_points = vtk.vtkPoints()
    for j in range(NUM_FACES_Y):
        for i in range(NUM_FACES_X):
            # Extract face coordinates from gv (vertex coordinates)
            x = gv.face_x_coords[i]
            y = gv.face_y_coords[i, j]
            vtk_points.InsertNextPoint(x, y, 0)  # Assuming 2D grid, z=0

    # Create the structured grid (vtkStructuredGrid)
    structured_grid = vtk.vtkStructuredGrid()
    structured_grid.SetDimensions(NUM_FACES_X, NUM_FACES_Y, 1)  # Set the grid dimensions

    # Set the points for the grid
    structured_grid.SetPoints(vtk_points)

    # Create density field (rho)
    vtk_rho = vtk.vtkFloatArray()
    vtk_rho.SetName("Density")
    vtk_rho.SetNumberOfComponents(1)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_rho.InsertNextValue(gv.rho[i, j])

    structured_grid.GetCellData().SetScalars(vtk_rho)

    # Create velocity vector field (U) with 2 components (u, v)
    vtk_velocity = vtk.vtkFloatArray()
    vtk_velocity.SetName("Velocity")
    vtk_velocity.SetNumberOfComponents(2)  # 2 components for the vector (u, v)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_velocity.InsertNextTuple2(gv.u[i, j, 0], gv.u[i, j, 1])  # Assuming gv.u contains 2 components for velocity in x and y

    structured_grid.GetCellData().AddArray(vtk_velocity)

    # Create total energy field (E)
    vtk_energy = vtk.vtkFloatArray()
    vtk_energy.SetName("Total Energy")
    vtk_energy.SetNumberOfComponents(1)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_energy.InsertNextValue(gv.E[i, j])

    structured_grid.GetCellData().AddArray(vtk_energy)

    # Create internal energy field (e)
    vtk_internal_energy = vtk.vtkFloatArray()
    vtk_internal_energy.SetName("Internal Energy")
    vtk_internal_energy.SetNumberOfComponents(1)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_internal_energy.InsertNextValue(gv.e[i, j])

    structured_grid.GetCellData().AddArray(vtk_internal_energy)

    # Create temperature field (T)
    vtk_temperature = vtk.vtkFloatArray()
    vtk_temperature.SetName("Temperature")
    vtk_temperature.SetNumberOfComponents(1)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_temperature.InsertNextValue(gv.T[i, j])

    structured_grid.GetCellData().AddArray(vtk_temperature)

    # Create speed of sound field (c)
    vtk_speed_of_sound = vtk.vtkFloatArray()
    vtk_speed_of_sound.SetName("Speed of Sound")
    vtk_speed_of_sound.SetNumberOfComponents(1)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_speed_of_sound.InsertNextValue(gv.c[i, j])

    structured_grid.GetCellData().AddArray(vtk_speed_of_sound)

    # Create pressure field (p)
    vtk_pressure = vtk.vtkFloatArray()
    vtk_pressure.SetName("Pressure")
    vtk_pressure.SetNumberOfComponents(1)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_pressure.InsertNextValue(gv.p[i, j])

    structured_grid.GetCellData().AddArray(vtk_pressure)

    # Create total enthalpy field (H)
    vtk_enthalpy = vtk.vtkFloatArray()
    vtk_enthalpy.SetName("Total Enthalpy")
    vtk_enthalpy.SetNumberOfComponents(1)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_enthalpy.InsertNextValue(gv.H[i, j])

    structured_grid.GetCellData().AddArray(vtk_enthalpy)

    # Create Mach number field (M)
    vtk_mach = vtk.vtkFloatArray()
    vtk_mach.SetName("Mach Number")
    vtk_mach.SetNumberOfComponents(1)

    for j in range(NUM_CELLS_Y):
        for i in range(NUM_CELLS_X):
            vtk_mach.InsertNextValue(gv.M[i, j])

    structured_grid.GetCellData().AddArray(vtk_mach)
    
    # Add m_in to the PolyData
    vtk_m_in = vtk.vtkFloatArray()
    vtk_m_in.SetName("Inlet Massflow")  # Name for the m_in field
    vtk_m_in.SetNumberOfComponents(1)  # 1 component per point

    # Ensure that gv.m_in is a list or numpy array with the same length as the number of points
    for index, value in enumerate(gv.m_in):
        if(index <= gv.iteration):
            vtk_m_in.InsertNextValue(value)  # Insert each value of m_in into the vtk array

    structured_grid.GetFieldData().AddArray(vtk_m_in)

    # Add m_out to the PolyData
    vtk_m_out = vtk.vtkFloatArray()
    vtk_m_out.SetName("Outlet Massflow")  # Name for the m_out field
    vtk_m_out.SetNumberOfComponents(1)  # 1 component per point

    # Ensure that gv.m_out is a list or numpy array with the same length as the number of points
    for index, value in enumerate(gv.m_out):
        if(index <= gv.iteration):
            vtk_m_out.InsertNextValue(value)  # Insert each value of m_out into the vtk array

    structured_grid.GetFieldData().AddArray(vtk_m_out)

    # Create the output directory if it doesn't exist
    file_path = os.path.join(gv.sim_dir, str(iteration), f"{iteration}.vts")

    # Save the structured grid to file using vtkXMLStructuredGridWriter
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(structured_grid)
    writer.Write()

    # Add entry for PVD file (used for time series visualization)
    gv.pvd_entries.append(f'    <DataSet timestep="{iteration}" file="{iteration}/{iteration}.vts"/>')
    
    # Get the size of the file
    file_size = os.path.getsize(file_path)
    
    # Get the current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print_data_save_info(iteration, file_path, file_size, current_time)

    return None

def read_iteration(file_path):
    # Initialize the VTK reader for .vts files
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()  # Read the file

    # Get the structured grid data
    structured_grid = reader.GetOutput()

    # Extract grid dimensions
    dimensions = structured_grid.GetDimensions()
    NUM_FACES_X, NUM_FACES_Y = dimensions[0], dimensions[1]

    # Extract point coordinates (grid face positions)
    points = structured_grid.GetPoints()
    gv.face_x_coords = np.zeros(NUM_FACES_X)
    gv.face_y_coords = np.zeros((NUM_FACES_X, NUM_FACES_Y))

    for j in range(NUM_FACES_Y):
        for i in range(NUM_FACES_X):
            x, y, _ = points.GetPoint(i + j * NUM_FACES_X)  # Extract x, y (ignoring z)
            gv.face_x_coords[i] = x
            gv.face_y_coords[i, j] = y

    # Extract cell-based scalar fields
    cell_data = structured_grid.GetCellData()

    def extract_field(field_name):
        """Helper function to extract a field from cell data."""
        vtk_array = cell_data.GetArray(field_name)
        if not vtk_array:
            return None  # Field might not be present
        data = np.zeros((NUM_FACES_X - 1, NUM_FACES_Y - 1))  # Adjust for cell-centered data
        for j in range(NUM_FACES_Y - 1):
            for i in range(NUM_FACES_X - 1):
                data[i, j] = vtk_array.GetValue(i + j * (NUM_FACES_X - 1))
        return data

    # Load each field into gv
    gv.rho = extract_field("Density")
    
    # Extract velocity as a 2-component vector field
    gv.u = np.zeros((NUM_FACES_X - 1, NUM_FACES_Y - 1, 2))
    velocity_array = cell_data.GetArray("Velocity")
    if velocity_array:
        for j in range(NUM_FACES_Y - 1):
            for i in range(NUM_FACES_X - 1):
                gv.u[i, j, 0], gv.u[i, j, 1] = velocity_array.GetTuple2(i + j * (NUM_FACES_X - 1))

    gv.E = extract_field("Total Energy")
    gv.e = extract_field("Internal Energy")
    gv.T = extract_field("Temperature")
    gv.c = extract_field("Speed of Sound")
    gv.p = extract_field("Pressure")
    gv.H = extract_field("Total Enthalpy")
    gv.M = extract_field("Mach Number")

    # Extract global field data (e.g., mass flow rates)
    field_data = structured_grid.GetFieldData()

    def extract_field_data(field_name):
        """Helper function to extract global field data."""
        vtk_array = field_data.GetArray(field_name)
        if not vtk_array:
            return None
        return np.array([vtk_array.GetValue(i) for i in range(vtk_array.GetNumberOfTuples())])

    if (MAX_ITERATIONS - gv.iteration) == 0:
        gv.m_in[:] = extract_field_data("Inlet Massflow")
        gv.m_out[:] = extract_field_data("Outlet Massflow")
    else:
        gv.m_in[:-(MAX_ITERATIONS-gv.iteration)] = extract_field_data("Inlet Massflow")
        gv.m_out[:-(MAX_ITERATIONS-gv.iteration)] = extract_field_data("Outlet Massflow")

    # Fill the state vector
    gv.state_vector[:, :, 0] = gv.rho[:,:]
    gv.state_vector[:, :, 1] = gv.rho[:,:] * gv.u[:,:,0]
    gv.state_vector[:, :, 2] = gv.rho[:,:] * gv.u[:,:,1]
    gv.state_vector[:, :, 3] = gv.rho[:,:] * gv.E[:,:]

    print(f"[INFO] Successfully loaded .{os.sep}{simplify_file_path(file_path, str(PROJECT_DIR))}")
    print("~" * max_width)

    return None

def save_pvd():
    """
    Creates and saves a .pvd file that references all saved .vtp files.

    The .pvd file is an XML-based file format used by VTK to manage a collection of 
    .vtp files (e.g., for visualization or simulation data). This function compiles 
    all entries from `gv.pvd_entries` into a single .pvd file and saves it in the 
    simulation directory (`gv.sim_dir`) with a timestamped filename.

    Returns:
    --------
    None
    """
    global max_width

    # Create the XML content for the .pvd file
    pvd_content = (
        """<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
    <Collection>
{}
    </Collection>
</VTKFile>
        """.format("\n".join(gv.pvd_entries)))  # Format and join the pvd entries with indentation

    # Get the current timestamp for the filename
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Construct the .pvd filename using the current timestamp and simulation parameters
    pvd_filename = os.path.join(
        gv.sim_dir, f"{current_time}_solution_{NUM_CELLS_X}_{NUM_CELLS_Y}.pvd"
    )
    
    # Save the .pvd file
    with open(pvd_filename, "w") as f:
        f.write(pvd_content)

    # Display a message indicating the data output
    print("\n")
    print("~" * max_width)
    print("  -- Data Output  --  ")
    print("~" * max_width)

    # Print confirmation of the saved file
    print(f"\nSaved PVD file: .{os.sep}{simplify_file_path(pvd_filename, str(PROJECT_DIR))}")
    
    return None

def read_existing_pvd(pvd_filename=None):
    """
    Reads an existing .pvd file to extract and store <DataSet> entries.

    If no filename is provided, this function searches the `sim_dir` directory 
    for the oldest `.pvd` file and uses it. If no `.pvd` file is found, it returns
    a warning and exits gracefully.

    Parameters:
    -----------
    pvd_filename (str, optional): The path to a specific .pvd file. If not provided,
                                    the function searches for the oldest `.pvd` file
                                    in the `sim_dir`.

    Returns:
    --------
    None
    """
    # Display a message indicating the data read
    print("\n")
    print("~" * max_width)
    print("  -- Read simulation  --  ")
    print("~" * max_width)

    # If no filename is given, find the oldest .pvd file in the directory
    if pvd_filename is None:
        pvd_files = glob.glob(os.path.join(gv.sim_dir, "*.pvd"))
        
        if not pvd_files:
            print(f"[WARNING] No .pvd files found in directory:\n          .{os.sep}{simplify_file_path(pvd_filename, str(PROJECT_DIR))}")
            return None
        
        # Sort files by creation time and pick the oldest one
        pvd_filename = max(pvd_files, key=os.path.getctime)
        print(f"[INFO] No filename provided. Using the oldest file:\n       .{os.sep}{simplify_file_path(pvd_filename, str(PROJECT_DIR))}")

    # Check if the specified or discovered file exists
    if not os.path.exists(pvd_filename):
        print(f"[WARNING] PVD file '{pvd_filename}' does not exist.")
        return None

    # Read the content of the .pvd file
    with open(pvd_filename, "r") as f:
        content = f.read()

    # Use a regular expression to extract all <DataSet> entries
    existing_entries = re.findall(r"<DataSet[\s\S]*?/>\s*", content)

    # If no entries are found, issue a warning and return an empty list
    if not existing_entries:
        print(f"[WARNING] No <DataSet> entries found in '{pvd_filename}'.")
        return None

    # Add the first entry as-is (without indentation), and indent subsequent entries
    gv.pvd_entries.extend(
        [existing_entries[0].strip()] + [f"\t{entry.strip()}" for entry in existing_entries[1:]]
    )

    print("-" * max_width)

    return None

def save_two_arrays_in_csv(x, data, x_header, data_header, iteration_dir=OUTPUT_DIR):
    """
    Saves two 1D arrays (x and corresponding data) to a CSV file.

    Parameters:
    -----------
    x : 1D array (list or np.ndarray)
        The array of x values (e.g., spatial coordinates).
    data : 1D array (list or np.ndarray)
        The array of data values (e.g., Cp, velocity, etc.) corresponding to the x values.
    x_header : str
        The header for the x-values column.
    data_header : str
        The header for the data-values column.
    iteration_dir : str, optional
        The directory where the file will be saved. Defaults to OUTPUT_DIR.
    
    Returns:
    --------
    None
    """
    # Ensure the input arrays are 1D (single rows of data)
    if len(x.shape) != 1 or len(data.shape) != 1:
        raise ValueError("Both input arrays must be 1D (single rows of data).")
    
    # Check if both arrays have the same length
    if len(x) != len(data):
        raise ValueError("The two arrays must have the same length.")
    
    file_path = os.path.join(iteration_dir, f"{gv.iteration}_{data_header}_{NUM_CELLS_X}_{NUM_CELLS_Y}.csv")

    # Save the arrays to a CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow([x_header, data_header])
        
        # Write the data
        for x, data in zip(x, data):
            writer.writerow([x, data])
    
    print(f"Data has been exported to {file_path}")
    return None

def print_data_save_info(iteration, file_path, file_size, current_time):
    """
    Display information about a successfully saved data file.

    Parameters:
    iteration (int): The current iteration number.
    file_path (str): The full path to the saved file.
    file_size (float): The size of the saved file in bytes.
    current_time (str): The timestamp when the file was saved.
    """
    global max_width

    # Simplify the file path relative to the project directory
    simplified_file_path = simplify_file_path(file_path, str(PROJECT_DIR))

    print("\n")
    # Print header with divider lines
    print("~" * max_width)
    print("  -- Data Saved Successfully --  ")
    print("~" * max_width)

    # Print details of the saved file
    print("Details")
    print("-" * max_width)
    print(f" Iteration:\t\t{iteration}")
    print(f" File Path:\t\t.{os.sep}{simplified_file_path}")
    print(f" File Size:\t\t{file_size / 1024:.2f} KB")  # Convert size to KB
    print(f" Timestamp:\t\t{current_time}")
    print("~" * max_width)

def print_iteration_residual(iteration, R_start, R_final):
    """
    Print residuals at the start and end of a specific iteration.
    
    Parameters:
    iteration (int): The iteration number.
    R_start (dict): Residuals at the start of the iteration (keys: 'rho', 'qux', 'quy', 'qE').
    R_final (dict): Residuals at the end of the iteration (keys: 'rho', 'qux', 'quy', 'qE').
    """
    global max_width

    print("\n")
    # Print the divider line
    print("~" * max_width)
    print(f"  -- Iteration {iteration} Residuals --  ")
    print("~" * max_width)

    # Print iteration information
    print("Residual Information")
    print("-" * max_width)

    # Print maximum residuals at the start
    print(" Start Residuals:")
    print(r"   ρ:   {:.5e}".format(R_start[0]))
    print(r"   ρu:  {:.5e}".format(R_start[1]))
    print(r"   ρv:  {:.5e}".format(R_start[2]))
    print(r"   ρE:  {:.5e}".format(R_start[3]))
    # Print a separator
    print("-" * max_width)

    # Print maximum residuals at the end
    print(" Final Residuals:")
    print(r"   ρ:   {:.5e}".format(R_final[0]))
    print(r"   ρu:  {:.5e}".format(R_final[1]))
    print(r"   ρv:  {:.5e}".format(R_final[2]))
    print(r"   ρE:  {:.5e}".format(R_final[3]))

    # Print a closing divider
    print("~" * max_width)
    
    return None

def print_simulation_info():
    """
    Print the initial simulation parameters to the console in a well-formatted way.

    Parameters:
    -----------
    sim_dir (str): Directory where the simulation output files will be saved.

    Returns:
    --------
    None
    """
    global max_width
    
    simplified_file_path = simplify_file_path(str(gv.sim_dir), str(PROJECT_DIR))

    # Print the divider line
    print("~" * max_width)
    print(f"  -- Simulation Info --  ")
    print("~" * max_width)

    # Print simulation metadata
    print("Simulation Metadata")
    print("-" * max_width)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Print the simulation information with aligned columns
    print(f"{' Iteration:':<20} {gv.iteration:<30}")
    print(f"{' Output Directory:':<20} .{os.sep}{simplified_file_path:<30}")
    print(f"{' Timestamp:':<20} {timestamp:<30}")

    print("-" * max_width)

    # Print computational grid information
    print("Computational Grid")
    print("-" * max_width)
    print(f" Number of Cells in X-Direction: {NUM_CELLS_X}")
    print(f" Number of Cells in Y-Direction: {NUM_CELLS_Y}")
    
    print("-" * max_width)

    # Print geometry information
    print("Geometry Information")
    print("-" * max_width)
    if USE_CIRCULAR_ARC:
        print(" Bump Shape: Circular Arc")
    else:
        print(" Bump Shape: Custom Function")

    print("-" * max_width)

    # Print simulation settings
    print("Simulation Settings")
    print("-" * max_width)
    print(f" Maximum Iterations: {MAX_ITERATIONS}")
    if USE_SUBSONIC_AD:
        print(" Artificial Dissipation: Advanced (valid for shocks)")
    else:
        print(" Artificial Dissipation: Standard (subsonic)")

    # Print a closing divider
    print("~" * max_width)
    print("\n")

    return None