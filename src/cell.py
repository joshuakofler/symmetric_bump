#%%
# TODO: Done
# this calculates/updates all the quantities for a cell 

from constants import *
import global_vars as gv
import numpy as np

from mesh import get_cell_dy

def initialize():
    # define the initial state of all parameters
    # calculate the upstream parameters
    calculate_inlet_properties()
    
    # use the upstream conditions as inital state    
    gv.rho[:,:] = gv.rho_infty

    gv.u[:,:,0] = gv.u_infty
    gv.u[:,:,1] = 0.0

    e_init = SPECIFIC_HEAT_CV * gv.T_infty
    gv.E[:,:] = e_init + 0.5 * (gv.u[:,:,0]**2 + gv.u[:,:,1]**2)

    gv.state_vector[:, :, 0] = gv.rho[:,:]
    gv.state_vector[:, :, 1] = gv.rho[:,:] * gv.u[:,:,0]
    gv.state_vector[:, :, 2] = gv.rho[:,:] * gv.u[:,:,1]
    gv.state_vector[:, :, 3] = gv.rho[:,:] * gv.E[:,:]

    update_cell_properties(gv.state_vector)

    update_in_out_massflow()

    return None

def calculate_inlet_properties():
    # stagnation temperature
    gv.T_infty[:] = ATMOSPHERIC_TEMPERATURE / (1 + (HEAT_CAPACITY_RATIO - 1)/2 * np.power(UPSTREAM_MACH_NUMBER,2))
    # stagnation pressure
    gv.p_infty[:] = ATMOSPHERIC_PRESSURE / (np.power((1 + (HEAT_CAPACITY_RATIO - 1)/2 * np.power(UPSTREAM_MACH_NUMBER,2)),
                                                ((HEAT_CAPACITY_RATIO)/(HEAT_CAPACITY_RATIO - 1))))
    # stagnation density    
    gv.rho_infty[:] = gv.p_infty / (GAS_CONSTANT * gv.T_infty)

    gv.c_infty[:] = np.sqrt(HEAT_CAPACITY_RATIO * GAS_CONSTANT * gv.T_infty)
    gv.u_infty[:] = UPSTREAM_MACH_NUMBER * gv.c_infty

    return None

def update_cell_properties(state_vector):
    """
    Updates the properties (density, velocity, temperature, pressure, etc.) of all cells in the grid
    based on the state vector.

    Args:
        state_vector (np.ndarray): A 3D array where each cell contains the state variables:
                                    - rho: density
                                    - ux: x-component of velocity
                                    - uy: y-component of velocity
                                    - E: total energy

    Returns:
        None
    """
    # Extract the state variables from the state vector
    gv.rho[:,:] = state_vector[:,:,0]          # Density
    gv.u[:,:,0] = state_vector[:,:,1] / gv.rho[:,:]    # x-component of velocity
    gv.u[:,:,1] = state_vector[:,:,2] / gv.rho[:,:]    # y-component of velocity
    gv.E[:,:] = state_vector[:,:,3] / gv.rho[:,:]      # Total energy

    # Calculate internal energy (e)
    gv.e[:,:] = calculate_internal_energy(gv.E, gv.u)

    # Calculate temperature (T) based on internal energy
    gv.T[:,:] = calculate_temperature(gv.e)

    # Calculate local speed of sound (c) based on temperature
    gv.c[:,:] = calculate_local_speed_of_sound(gv.T)
    
    # Calculate pressure (p) from density and temperature
    gv.p[:,:] = calculate_pressure(gv.rho, gv.T)

    # Calculate total enthalpy (H)
    gv.H[:,:] = calculate_total_enthalpy(gv.rho, gv.E, gv.p)

    # Calculate Mach number (M)
    gv.M[:,:] = calculate_mach_number(gv.u, gv.c)

    # No return as this is a function to update values in place
    return None

def update_in_out_massflow():
    """
    Update the inlet and outlet mass flow for the current iteration and save the values for convergence analysis.

    The function computes the mass flow at the inlet and outlet of the system using the `calculate_massflow` function.
    The inlet mass flow is calculated for the first mesh line (index 0), and the outlet mass flow is calculated for 
    the last mesh line (index NUM_CELLS_X-1). These values are then stored for later analysis of the convergence history.

    This function is typically called once per iteration to track the evolution of mass flow over time.

    Args:
        None

    Returns:
        None
    """
    
    # Calculates the inlet mass flow for the current iteration
    gv.m_in[gv.iteration] = calculate_massflow(gv.rho, gv.u, 0)

    # Calculates the outlet mass flow for the current iteration
    gv.m_out[gv.iteration] = calculate_massflow(gv.rho, gv.u, NUM_CELLS_X-1)

    return None

def calculate_internal_energy(E, u):
    """
    Calculate the internal energy element-wise for each point in the grid by subtracting the kinetic energy
    from the total energy.

    The internal energy is calculated as:
        e = E - (u_x^2 + u_y^2) / 2

    Where:
        e is the internal energy per unit mass,
        E is the total energy per unit mass,
        u_x and u_y are the velocity components in the x and y directions.

    Args:
        E (np.ndarray): A 2D array representing the total energy per unit mass at each grid point.
        u (np.ndarray): A 3D array representing the velocity components at each grid point. The shape of u is (2, n, m), where
                        u[0] contains the x-component (u_x) of the velocity, and u[1] contains the y-component (u_y).

    Returns:
        np.ndarray: A 2D array of internal energy values per unit mass at each grid point.
    """
    # Calculate the kinetic energy at each point: (u_x^2 + u_y^2) / 2
    kinetic_energy = 0.5 * (np.power(u[:,:,0],2) + np.power(u[:,:,1],2))
    
    # Subtract the kinetic energy from the total energy to get the internal energy
    internal_energy = E - kinetic_energy
    
    return internal_energy

def calculate_temperature(e):
    """
    Calculate the temperature from the internal energy using the specific heat at constant volume.

    The temperature is calculated using the relationship between internal energy and specific heat:
        T = e / C_V

    Args:
        e (np.ndarray): The internal energy (e) per unit mass (in J/kg) for each cell.

    Returns:
        np.ndarray: The temperature (T) in Kelvin (K) for each cell, calculated from the internal energy.
    """
    # Temperature is calculated by dividing the internal energy by the specific heat at constant volume
    temperature = e / SPECIFIC_HEAT_CV

    return temperature

def calculate_local_speed_of_sound(T):
    """
    Calculate the local speed of sound based on the temperature.

    The speed of sound in an ideal gas is given by the formula:
        c = sqrt(γ * R * T)

    Where:
        c is the speed of sound,
        γ (HEAT_CAPACITY_RATIO) is the ratio of specific heats (C_P / C_V),
        R is the specific gas constant,
        T is the temperature in Kelvin.

    Args:
        T (np.ndarray): The temperature (T) in Kelvin (K) for each cell.

    Returns:
        np.ndarray: The local speed of sound (c) for each cell, calculated from the temperature.
    """
    # Speed of sound is calculated as the square root of (γ * R * T)
    speed_of_sound = np.sqrt(HEAT_CAPACITY_RATIO * GAS_CONSTANT * T)

    return speed_of_sound

def calculate_pressure(rho, T):
    """
    Calculate the pressure of an ideal gas based on its density and temperature.

    The pressure of an ideal gas is calculated using the ideal gas law:
        p = ρ * R * T

    Where:
        p is the pressure,
        ρ (rho) is the density of the gas in kg/m³,
        R is the specific gas constant,
        T is the temperature in Kelvin.

    Args:
        rho (np.ndarray): The density (ρ) of the gas in kg/m³ for each cell.
        T (np.ndarray): The temperature (T) in Kelvin (K) for each cell.

    Returns:
        np.ndarray: The pressure (p) in Pascals (Pa) for each cell, calculated from the density and temperature.
    """
    # Pressure is calculated using the ideal gas law: p = ρ * R * T
    pressure = rho * GAS_CONSTANT * T

    return pressure

def calculate_total_enthalpy(rho, E, p):
    """
    Calculate the total enthalpy of the system based on density, total energy, and pressure.

    The total enthalpy (h) is calculated using the following formula:
        H = E + p / ρ

    Where:
        H is the total enthalpy,
        E is the total energy per unit mass,
        p is the pressure,
        ρ (rho) is the density of the gas.

    Args:
        rho (np.ndarray): The density (ρ) of the gas in kg/m³ for each cell.
        E (np.ndarray): The total energy (E) per unit mass in J/kg for each cell.
        p (np.ndarray): The pressure (p) in Pascals (Pa) for each cell.

    Returns:
        np.ndarray: The total enthalpy (H) in J/kg for each cell, calculated from the density, total energy, and pressure.
    """
    # Total enthalpy is calculated as: H = E + p / ρ
    total_enthalpy = E + p / rho

    return total_enthalpy

def calculate_mach_number(u, c):
    """
    Calculate the Mach number at each point in the grid.

    The Mach number (M) is calculated using the following formula:
        M = √(u_x² + u_y²) / c

    Where:
        M is the Mach number,
        u_x and u_y are the velocity components in the x and y directions, respectively,
        c is the speed of sound at each grid point.

    Args:
        u (np.ndarray): The velocity field (u_x, u_y) in m/s with shape (n, m, 2), where n and m are the number of grid points in the x and y directions.
        c (np.ndarray): The speed of sound at each point in the grid, in m/s, with shape (n, m), where n and m are the number of grid points in the x and y directions.

    Returns:
        np.ndarray: The Mach number (M) at each point in the grid, with shape (n, m).
    """
    
    M = np.sqrt(np.power(u[:,:,0],2) + np.power(u[:,:,1],2)) / c[:,:]
    return M

def calculate_massflow(rho, u, cell_x_index):
    """
    Calculate the mass flow through a vertical mesh line (indexed by cell_x_index) using a second-order trapezoidal integration method.

    The mass flow (m) is calculated using the following formula:
        m = Σ [ 0.5 * (ρ_i * u_i + ρ_(i+1) * u_(i+1)) * Δy ]

    Where:
        m is the mass flow,
        ρ (rho) is the density of the gas in kg/m³ at each cell,
        u is the velocity in m/s at each cell,
        Δy is the cell spacing in the y-direction (vertical distance between adjacent cells).

    Args:
        rho (np.ndarray): The density (ρ) of the gas in kg/m³, with shape (1, n), where n is the number of cells in the x-direction.
        u (np.ndarray): The velocity (u) of the gas in m/s, with shape (1, n, 1), where n is the number of cells in the x-direction.
        iteration (int): The current iteration in the simulation, not used directly in the calculation but could be useful for debugging or logging.
        cell_x_index (int): The index of the vertical mesh line in the x-direction for which the mass flow is to be calculated.

    Returns:
        float: The total mass flow (m) through the vertical mesh line at cell_x_index, in kg/s.
    """
    # Apply the trapezoidal rule to calculate mass flow
    m = (0.5 * (rho[cell_x_index, :-1] * u[cell_x_index, :-1, 0] + rho[cell_x_index, 1:] * u[cell_x_index, 1:, 0]) * get_cell_dy(cell_x_index)).sum() 
    
    return m

def get_all_cell_properties():
    """
    Returns the properties (density, velocity, temperature, pressure, etc.) of all cells in the grid
    based on the state vector.

    Args:
        state_vector (np.ndarray): A 3D array where each cell contains the state variables:
                                    - rho: density
                                    - ux: x-component of velocity
                                    - uy: y-component of velocity
                                    - E: total energy

    Returns:
        cell_properties (np.ndarray): A 7D array whit all state variables at every grid cell
    """
    cell_properties = np.array([NUM_CELLS_X, NUM_CELLS_Y, 5], 'd')

    # Extract the state variables from the state vector
    rho = gv.state_vector[:,:,0]  # Density
    gv.u[:,:,0] = gv.state_vector[:,:,1] / rho      # x-component of velocity
    gv.u[:,:,1] = gv.state_vector[:,:,2] / rho      # y-component of velocity
    E = gv.state_vector[:,:,3] / rho       # Total energy

    # Calculate internal energy (e)
    e = calculate_internal_energy(E, gv.state_vector[:,:,1:])

    # Calculate temperature (T) based on internal energy
    T = calculate_temperature(e)

    # Calculate local speed of sound (c) based on temperature
    c = calculate_local_speed_of_sound(T)

    # Calculate pressure (p) from density and temperature
    p = calculate_pressure(rho, T)

    # Calculate total enthalpy (H)
    H = calculate_total_enthalpy(rho, E, p)

    cell_properties[:,:,0] = e
    cell_properties[:,:,1] = T
    cell_properties[:,:,2] = c
    cell_properties[:,:,3] = p
    cell_properties[:,:,4] = H

    # No return as this is a function to update values in place
    return cell_properties

def print_cell_properties(cell_x_index, cell_y_index):
    """
    Print all relevant properties of a specific cell in the grid.

    This function will print the properties of the cell at the given (cell_x_index, cell_y_index) location,
    including density, velocity components, internal energy, temperature, pressure, and enthalpy.

    Args:
        cell_x_index (int): The x-index of the cell in the grid.
        cell_y_index (int): The y-index of the cell in the grid.
        rho (np.ndarray): A 2D array representing the density at each grid point.
        u (np.ndarray): A 3D array representing the velocity components (u_x, u_y) at each grid point.
        E (np.ndarray): A 2D array representing the total energy at each grid point.
    """
    # Get the properties for the specified cell
    density = gv.rho[cell_x_index, cell_y_index]
    ux = gv.u[cell_x_index, cell_y_index, 0]
    uy = gv.u[cell_x_index, cell_y_index, 1]
    total_energy = gv.E[cell_x_index, cell_y_index]

    internal_energy = gv.e[cell_x_index, cell_y_index]
    
    temperature = gv.T[cell_x_index, cell_y_index]
    
    pressure = gv.p[cell_x_index, cell_y_index]
    
    speed_of_sound = gv.c[cell_x_index, cell_y_index]
    
    enthalpy = gv.H[cell_x_index, cell_y_index]
    
    # Print the properties of the specified cell
    print("\n\n")
    print(f"Properties of Cell: ({cell_x_index}, {cell_y_index})\n")
    print(f"Density: {density} kg/m³")
    print(f"Velocity components: u_x = {ux} m/s, u_y = {uy} m/s")
    print(f"Total Energy: {total_energy} J/kg")
    print(f"Internal Energy: {internal_energy} J/kg")
    print(f"Temperature: {temperature} K")
    print(f"Pressure: {pressure} Pa")
    print(f"Speed of Sound: {speed_of_sound} m/s")
    print(f"Enthalpy: {enthalpy} J/kg")

def get_point_data(cell_x_index, cell_y_index, prop):
    return np.mean([prop[cell_x_index+1, cell_y_index], 
                    prop[cell_x_index+1, cell_y_index+1],
                    prop[cell_x_index, cell_y_index+1], 
                    prop[cell_x_index, cell_y_index]])