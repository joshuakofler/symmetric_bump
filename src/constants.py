#%%
# Constants

# Computational domain dimensions
DOMAIN_LENGTH = 1  # Base length scale

CHANNEL_HEIGHT = DOMAIN_LENGTH
CHANNEL_LENGTH = 3 * DOMAIN_LENGTH

# Bump coefficient
BUMP_COEFFICIENT = 0.08

# Number of grid points
NUM_CELLS_X = int(6)  # For debugging purposes, use values like 6, 12, 24, etc.
NUM_CELLS_Y = int(5)

# Number of faces in the grid
NUM_FACES_X = int(NUM_CELLS_X + 1)
NUM_FACES_Y = int(NUM_CELLS_Y + 1)

# Specific heat capacities (commented placeholders)
# SPECIFIC_HEAT_CV = ...
# SPECIFIC_HEAT_CP = ...

# Heat capacity ratio
HEAT_CAPACITY_RATIO = 1.4

# Upstream Mach number
UPSTREAM_MACH_NUMBER = 0.1

# Atmospheric conditions
ATMOSPHERIC_PRESSURE = 101300       # Pressure in Pascals
ATMOSPHERIC_TEMPERATURE = 288       # Temperature in Kelvin

# Runge-Kutta scheme
RK_ALPHA_2 = 1/4
RK_ALPHA_3 = 1/3
RK_ALPHA_4 = 1/2