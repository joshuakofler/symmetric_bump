#%%
# TODO: Done
# this calculates/updates all the quantities for a cell 

from constants import *
import global_vars as gv
import numpy as np

def initialize():
    # define the initial state of all parameters
    # calculate the upstream parameters
    calculate_inlet_properties()
    
    # use the upstream conditions as inital state    
    gv.rho[:,:] = gv.rho_infty

    gv.u[:,:,0] = gv.u_infty
    gv.u[:,:,1] = 0.1

    e_init = SPECIFIC_HEAT_CV * ATMOSPHERIC_TEMPERATURE
    gv.E[:,:] = e_init + 0.5 * (gv.u[:,:,0]**2 + gv.u[:,:,1]**2)

    gv.state_vector[:, :, 0] = gv.rho[:,:]
    gv.state_vector[:, :, 1] = gv.rho[:,:] * gv.u[:,:,0]
    gv.state_vector[:, :, 2] = gv.rho[:,:] * gv.u[:,:,1]
    gv.state_vector[:, :, 3] = gv.rho[:,:] * gv.E[:,:]

    update_cell_properties(gv.state_vector)

    return None

def calculate_inlet_properties():
    # # stagnation temperature
    T_0 = ATMOSPHERIC_TEMPERATURE * (1 + (HEAT_CAPACITY_RATIO - 1)/2 * UPSTREAM_MACH_NUMBER**2)
    # # stagnation pressure
    p_0 = ATMOSPHERIC_PRESSURE * (1 + (HEAT_CAPACITY_RATIO - 1)/2 * UPSTREAM_MACH_NUMBER**2)**(HEAT_CAPACITY_RATIO/(HEAT_CAPACITY_RATIO-1))

    gv.rho_infty[:] = p_0 / (GAS_CONSTANT * T_0)

    gv.c_infty[:] = np.sqrt(HEAT_CAPACITY_RATIO * GAS_CONSTANT * T_0)
    
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
    gv.u[:,:,0] = state_vector[:,:,1] / gv.rho    # x-component of velocity
    gv.u[:,:,1] = state_vector[:,:,2] / gv.rho    # y-component of velocity
    gv.E[:,:] = state_vector[:,:,3] / gv.rho      # Total energy

    # Calculate internal energy (e)
    gv.e = calculate_internal_energy(gv.E, gv.u)

    # Calculate temperature (T) based on internal energy
    gv.T = calculate_temperature(gv.e)

    # Calculate local speed of sound (c) based on temperature
    gv.c = calculate_local_speed_of_sound(gv.T)
    
    # Calculate pressure (p) from density and temperature
    gv.p = calculate_pressure(gv.rho, gv.T)

    # Calculate total enthalpy (H)
    gv.H = calculate_total_enthalpy(gv.rho, gv.E, gv.p)

    # No return as this is a function to update values in place
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
    kinetic_energy = 0.5 * (u[:,:,0]**2 + u[:,:,1]**2)
    
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

def calculate_massflow():
    # inlet
    gv.m_in[gv.iteration] = (0.5 * (gv.rho[0,:-1]*gv.u[0,:-1,0] + gv.rho[0,1:]*gv.u[0,1:,0]) * gv.cell_dy[0]).sum()
    # outlet
    gv.m_out[gv.iteration] = (0.5 * (gv.rho[-1,:-1]*gv.u[-1,:-1,0] + gv.rho[-1,1:]*gv.u[-1,1:,0]) * gv.cell_dy[-1]).sum()
    
    return None

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
