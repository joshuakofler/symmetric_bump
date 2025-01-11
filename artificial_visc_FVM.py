import numpy as np

#
def artificial_visc_FVM(U, rho, ux, p, Nx, K, k2, k4, gamma):
    #
    # steps
    # 1. compute s_candidate = |den/num| at centers
    # 2. compute s at faces from neighbor centers (-2,+2)
    # 3. compute eps1, eps2 from s, k2, k4,...
    # 4. compute Delta U in space at faces
    # 5. compute D
    buff_c = np.zeros((Nx,), 'd')
    buff_f = np.zeros((Nx + 1, 2), 'd')
    r_f = np.zeros((Nx + 1,), 'd')
    e2_f = np.zeros((Nx + 1,), 'd')
    e4_f = np.zeros((Nx + 1,), 'd')
    
    # compute auxiliar buffers
    r_f[1:-1] = np.abs(0.5 * (ux[:-1] + ux[1:])) + 0.5 * (buff_c[:-1] + buff_c[1:])
    buff_c[:] = np.sqrt(K * gamma * np.power(rho, gamma - 1.0))
    
    # pressure switch
    # s_candidate at centers
    buff_c[:] = 0.0
    buff_c[1:-1] = np.abs((p[:-2] - 2.0 * p[1:-1] + p[2:]) / (p[:-2] + 2.0 * p[1:-1] + p[2:]))
    buff_c[0] = buff_c[1]
    buff_c[-1] = buff_c[-2]
    
    # s at faces
    for i in range(Nx + 1 - 4):
        i_f = i + 2
        buff_f[i_f, 0] = np.max(buff_c[i:(i + 4)])
        
    # boundary conditions for faces
    buff_f[0, 0] = buff_f[2, 0]
    buff_f[1, 0] = buff_f[2, 0]
    buff_f[-1, 0] = buff_f[-3, 0]
    buff_f[-2, 0] = buff_f[-3, 0]
    
    e2_f[1:-1] = 0.5 * k2 * buff_f[1:-1, 0] * r_f[1:-1]
    e4_f[1:-1] = np.maximum(0.0, 0.5 * k4 * r_f[1:-1] - e2_f[1:-1])

    buff_f[1:-1, :] = U[1:, :] - U[:-1, :]
    buff_f[1:-1, 0] = (e2_f[1:-1] + 2.0 * e4_f[1:-1]) * buff_f[1:-1, 0] - e4_f[1:-1] * (buff_f[:-2, 0] + buff_f[2:, 0])
    buff_f[1:-1, 1] = (e2_f[1:-1] + 2.0 * e4_f[1:-1]) * buff_f[1:-1, 1] - e4_f[1:-1] * (buff_f[:-2, 1] + buff_f[2:, 1])
    
    return buff_f[1:-1, :]
