#%%
# constants


# computational domain
L = 1

channel_height = L
channel_length = 3 * L

# bump coefficient
epsilon = 0.08


# number of points
Nx_c = 12        # for debuging reasons use 6,12,24,..
Ny_c = 8

# number of faces
Nx_f = Nx_c + 1
Ny_f = Ny_c + 1


# atmospheric pressure
pa = 101300     # Pa
# atmospheric temperature
Ta = 288        # K