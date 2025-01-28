#%%
# Constants

# Basics
from pathlib import Path

# Get the current directory (which is inside the 'src' folder)
CURRENT_DIR = Path(__file__).parent
# Get the project directory
PROJECT_DIR = CURRENT_DIR.parent
# Go one level up and then append 'output' to the path
OUTPUT_DIR = PROJECT_DIR / "output"

MAX_ITERATIONS = 5

# Computational domain dimensions
DOMAIN_LENGTH = 1  # Base length scale

CHANNEL_HEIGHT = DOMAIN_LENGTH
CHANNEL_LENGTH = 3 * DOMAIN_LENGTH

# Bump coefficient
BUMP_COEFFICIENT = 0.08

# Number of grid points
NUM_CELLS_X = int(65)  # For debugging purposes, use values like 6, 12, 24, etc.
NUM_CELLS_Y = int(33)

# Number of faces in the grid
NUM_FACES_X = int(NUM_CELLS_X + 1)
NUM_FACES_Y = int(NUM_CELLS_Y + 1)

# Gas constant (for dry air)
GAS_CONSTANT = 287.05   # J / kgK

# Specific heat capacities (commented placeholders)
SPECIFIC_HEAT_CV = 718      # J / kgK
SPECIFIC_HEAT_CP = 1005     # J / kgK

# Heat capacity ratio
HEAT_CAPACITY_RATIO = 1.4

# Upstream Mach number
UPSTREAM_MACH_NUMBER = 0.1

# Atmospheric conditions
ATMOSPHERIC_PRESSURE = 101300       # Pressure in Pascals
ATMOSPHERIC_TEMPERATURE = 288       # Temperature in Kelvin

# Runge-Kutta scheme
# RK_ALPHA_2 = 1/4
# RK_ALPHA_3 = 1/3
# RK_ALPHA_4 = 1/2
RK_ALPHA = [1/4, 1/3, 1/2, 1]

CFL = 2

# Artificial Dissipation
# The coefficient k(2) is typically of order 1
ARTIFICIAL_DISSIPATION_KAPPA_2 = 1
# The coefficient k(4) should be small
ARTIFICIAL_DISSIPATION_KAPPA_4 = 1/256