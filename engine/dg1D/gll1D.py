import numpy as np
from scipy.special import legendre

# Nós de referencia
def gll_nodes(N,Np):
    if N == 0: return np.array([0.0])
    x = np.zeros(Np)
    x[0], x[-1] = -1.0, 1.0
    if N > 1:
        x[1:-1] = legendre(N).deriv().roots
    return np.sort(x)

def legendre_norm(N, x):
    return legendre(N)(x) * np.sqrt((2.0 * N + 1.0) / 2.0)

def grad_legendre_norm(N, x):
    return legendre(N).deriv()(x) * np.sqrt((2.0 * N + 1.0) / 2.0)

def ref(N,Np):
    r = gll_nodes(N,Np)
    return r

def fator_geometrico(K,Np):
    N = Np-1
    VX = np.linspace(-2, 2, K + 1)

    x = np.zeros((Np, K))
    for k in range(K):
        x[:, k] = VX[k] + 0.5 * (ref(N,Np) + 1.0) * (VX[k+1] - VX[k])

    J = 0.5 * (VX[1:] - VX[:-1])  
    rx = 1.0 / J                  
    Fscale = 1.0 / J              

    nx = np.zeros((2, K))
    nx[0, :] = -1.0 
    nx[1, :] = 1.0   

    return x, Fscale, rx, nx

def vmap(Np,K):
    vmapM = np.zeros((2, K), dtype=int)
    vmapM[0, :] = np.arange(0, K * Np, Np)
    vmapM[1, :] = np.arange(Np - 1, K * Np, Np)

    vmapP = np.zeros((2, K), dtype=int)
    vmapP[0, 1:] = vmapM[1, 0:-1]
    vmapP[1, 0:-1] = vmapM[0, 1:]

    vmapP[0, 0] = vmapM[0, 0]      
    vmapP[1, -1] = vmapM[1, -1]

    # Condições de Contorno Periódicas
    #vmapP[0, 0] = vmapM[1, -1]  # A face esquerda do 1º elemento "enxerga" a face direita do último
    #vmapP[1, -1] = vmapM[0, 0]  # A face direita do último elemento "enxerga" a face esquerda do 1º

    vmapM_flat = vmapM.flatten(order='F')
    vmapP_flat = vmapP.flatten(order='F')

    mapB = np.where(vmapM_flat == vmapP_flat)[0]
    vmapB = vmapM_flat[mapB]

    return vmapM, vmapP, mapB, vmapB