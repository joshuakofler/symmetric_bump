#%%
import numpy as np
import matplotlib.pyplot as plt
from artificial_visc_FVM import *
from time import sleep
# 
# Shocktube of a barotropic gas using McCormack scheme 
# (with artificial diffusion)
# given by p = K rho**gamma (K from initial conditions)
#
# initial density state:
#
#  	rho |
#    rho0l--+---------
#           |         |
#           |         |
#    rho0r--+          --------
#           +---------+--------+--> x
#           0        L/2       L
#
#    In the following sketch the symbols used are:
#     |  for faces
#     o  for centers
#     :  for domain boundaries (that coincide with the first and last face)
#
#  Centers: i= 0     1     2     3           Nc-4  Nc-3  Nc-2  Nc-1 
#           :                                                       :
#           |  o  |  o  |  o  |  o  | ....  |  o  |  o  |  o  |  o  |
#           :                                                       :
#  Faces: i=0     1     2     3     4     Nf-5  Nf-4  Nf-3  Nf-2  Nf-1 
#
#

# Input variables
L    = 4      # Domain length
Nc = 100      # Number of cells (also centers)
Nf = Nc+1     # Number of faces
CFL  = 0.9    # CFL number
nstep = 50     # number of time steps
k2 = 1.0 ; k4 = 1.0/32.0; # parameters for the artificial viscosity model

# Input gas constants
gamma = 1.4  
R     = 288.7 # [J/(kg K)]

# Input left state
p0l      = 1e5 # Pa
T0l      = 300.0 # K
rho0l    = p0l/(R*T0l) # kg/m**3
K        = p0l/rho0l**gamma

# Input right state
p0r     = 1e4 # Pa
rho0r   = (p0r/K)**(1./gamma);# since we deal with a barotropic gas

# Define the grid:
dx = L/Nc
xf = np.linspace(0.,L,Nf);
xc = 0.5*(xf[:-1]+xf[1:])
assert(np.abs(dx - (xf[1]-xf[0])<1e-10))
assert(np.abs(dx - (xc[1]-xc[0])<1e-10))

# Allocate variables
ux    = np.zeros((Nc, ),'d'); # velocity (center quantity)
rho   = np.zeros((Nc, ),'d'); # density  (center quantity)
p     = np.zeros((Nc, ),'d'); # pressure (center quantity)
U     = np.zeros((Nc,2),'d'); # U[:,0] = rho, U[:,1] = rho*ux (center quantity)
Ustar = np.zeros((Nc,2),'d'); # U at intermediate step of MacCormack
F     = np.zeros((Nf,2),'d'); # Fluxes
F_l = np.zeros((Nf-1,2),'d');   # F[i  ,:] -> flux at face i-1/2.
F_r = np.zeros((Nf-1,2),'d');   # F[i+1,:] -> flux at face i+1/2.
F_L = 0                         # F[ 0,:] Flux at left boundary
F_R = 0                         # F[-1,:] Flux at right boundary
# 0. [TASK]
# Initial condition
# Here you need to define the initial condition for rho, ux, p and U
ux[:] = 0

rho[:-Nc//2] = rho0l
rho[Nc//2:] = rho0r

p[:-Nc//2] = p0l
p[Nc//2:] = p0r

U[:,0] = rho[:]
U[:,1] = rho[:]*ux[:]**2

# initialize iteration procedure and set plots
totaltime = 0;

fig,axList=plt.subplots(3,1)

axList[0].plot(xc,rho,'r')
axList[0].set_ylabel(r"$\rho$")

axList[1].plot(xc,ux,'b')
axList[1].set_ylabel(r"$u_x$")

axList[2].plot(xc,p,'k') 
axList[2].set_ylabel("$p$")

axList[0].set_xticks([])
axList[1].set_xticks([])

axList[0].set_ylim(rho0r-0.1,rho0l+0.1)
axList[1].set_ylim(-301,301)
axList[2].set_ylim(0,p0l*1.1)

curves=[plt.findobj(axList[0],plt.Line2D, include_self=False),
        plt.findobj(axList[1],plt.Line2D, include_self=False),
        plt.findobj(axList[2],plt.Line2D, include_self=False)]

plt.ion()
plt.show()


# Time loop.
for istep in range(nstep):
    #print("Step = " + nstep)
    # We start with ux,rho,p,U equal to values in previous time step. 
    # or intial condition if istep=0
    
    # 1. [TASK] Determine the time step from the CFL condition
    u_max = float(1.0)
    for ix in range(Nc):
        # Obtain the speed of sound
        c = np.sqrt(gamma * p[ix]/rho[ix])
        # Obtain maximum propagation speed umax, 
        # which is the max of |u|+c, |u|-c,
        u_max =  max(u_max, np.abs(ux[ix])+c)

    u_max = np.max(np.abs(ux)+np.sqrt(gamma *p/rho))
    
    #  now we can obtain the time step from the CFL condition
    dt = CFL * dx / u_max
    
    totaltime = totaltime+dt;
    
    # 2. McCormack scheme 

    # 2.a [TASK]: 
    # Compute physical flux at the interior faces and 
    # at the boundaries. 
    # Since this is the firt step of the McCormack scheme, 
    # use forward approximation for the fluxes
    # Pressure is extrapolated.
    pLeft  = p[0]
    pRight = p[-1]
    ix = 0
    F[ix,0] = 0.0
    F[ix,1] = pLeft
    for ix_f in range(1,Nf-2+1): 
        ix_c = ix_f
        F[ix_f,0] = rho[ix_c] * ux[ix_c] 
        F[ix_f,1] = rho[ix_c] * ux[ix_c]**2 + p[ix_c]
    ix = Nf-1
    F[ix,0] = 0.0
    F[ix,1] = pRight
    
    # 2.b add artificial dissipation to the flux
    F[1:Nf-1,:] -= artificial_visc_FVM(U,rho,ux,p,Nc,K,k2,k4,gamma)
    
    # 2.c [TASK] obtain Ustar
    # Ustar[:,:] = U[:,:]
    for i in range(Nc):
        Ustar[i,0] = U[i,0] - dt/dx * (F[i+1,0] - F[i,0])
        Ustar[i,1] = U[i,1] - dt/dx * (F[i+1,1] - F[i, 1])

    # Second stage
    # 2.d [TASK] decode Ustar into rho, ux, p
    rho = Ustar[:,0]
    ux = Ustar[:,1] / rho
    p[:] = K * rho * gamma

    # 2.e [TASK]:  same as 2.a, but for the second substage
    pLeft  = p[0]
    pRight = p[-1]
    ix = 0
    F[ix,0] = 0.0
    F[ix,1] = pLeft
    for ix_f in range(1,Nf-2+1): 
        ix_c = ix_f-1
        F[ix_f,0] = rho[ix_c] * ux[ix_c] 
        F[ix_f,1] = rho[ix_c] * ux[ix_c]**2 + p[ix_c]
    ix = Nf-1
    F[ix,0] = 0.0
    F[ix,1] = pRight
    
    # 2.f add artificial dissipation to the flux
    F[1:Nf-1,:] -= artificial_visc_FVM(Ustar,rho,ux,p,Nc,K,k2,k4,gamma) 
    
    # 2.g [TASK] obtain new U by applying the second stage of the McCormack scheme
    U[:,0] = 1/2 * (Ustar[:,0] + U[:,0]) - 0.5 * dt/dx * (F[1:,0] - F[:-1,0])
    U[:,1] = 1/2 * (Ustar[:,1] + U[:,1]) - 0.5 * dt/dx * (F[1:,1] - F[:-1,1])


    # 2.h [TASK] Decode the variables, from U obtain rho, ux, p
    rho = U[:,0]
    ux = U[:,1] / rho
    p[:] = K * rho * gamma

    # Plot 

    # Look at the density, pressure and velocity
    curves[0][0].set_ydata(rho)
    curves[1][0].set_ydata(ux)
    curves[2][0].set_ydata(p)
    fig.canvas.draw()
    fig.canvas.flush_events()
    # uncomment/comment any of the following lines
    # to follow the evolution of the solution as desired
    sleep(0.1)
    
input("press any key to continue")
