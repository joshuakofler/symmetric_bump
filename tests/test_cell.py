#%%
import unittest
import numpy as np
import sys
import os
import importlib

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.globals import *  # Import global variables (rho, u, etc.)
from src.cell import update_cell_properties, calculate_internal_energy, calculate_local_speed_of_sound, calculate_pressure, calculate_temperature, calculate_total_enthalpy


# importlib.reload(update_cell_properties)

class TestUpdateCellProperties(unittest.TestCase):
    # def setUp(self):
    #     global state_vector
    #     self.grid_size = (NUM_CELLS_X, NUM_CELLS_Y)
        
    #     # Create a mock state vector (using global variables from globals.py)
    #     state_vector = np.zeros((*self.grid_size, 4))

    def test_update_cell_properties(self):
        global e

        #e[:,:] = 234

        # Create a mock state vector (based on your grid size)
        state_vector[:,:,0] = 122.0  # rho (density)
        state_vector[:,:,1] = 2.0  # rho * ux (momentum in x)
        state_vector[:,:,2] = 3.0  # rho * uy (momentum in y)
        state_vector[:,:,3] = 10.0  # rho * E (total energy)

        # Call the function to update the properties (this will modify global variables)
        update_cell_properties(state_vector)

        # Assert updated variables
        assert np.allclose(rho[:,:], state_vector[:,:,0], atol=1e-6)
        assert np.allclose(u[:,:,0], state_vector[:,:,1] / rho, atol=1e-6)
        assert np.allclose(u[:,:,1], state_vector[:,:,2] / rho, atol=1e-6)
        assert np.allclose(E[:,:], state_vector[:,:,3] / rho, atol=1e-6)

        # Check calculated properties
        expected_e = calculate_internal_energy(E, u)
        print(e)
        assert np.allclose(e, expected_e, atol=1e-6)

        expected_T = calculate_temperature(expected_e)
        assert np.allclose(T, expected_T, atol=1e-6)

        expected_c = calculate_local_speed_of_sound(expected_T)
        assert np.allclose(c, expected_c, atol=1e-6)

        expected_p = calculate_pressure(rho, expected_T)
        assert np.allclose(p, expected_p, atol=1e-6)

        expected_H = calculate_total_enthalpy(rho, E, expected_p)
        assert np.allclose(H, expected_H, atol=1e-6)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
