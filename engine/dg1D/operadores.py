import numpy as np
import gll1D as gll

def matriz_diferenciaçao(Np):
    V = np.zeros((Np, Np))
    Vr = np.zeros((Np, Np))
    N = Np - 1
    r = gll.ref(N,Np)

    for i in range(Np):
        for j in range(Np):
            V[i, j] = gll.legendre_norm(j, r[i])
            Vr[i, j] = gll.grad_legendre_norm(j, r[i])

    invV = np.linalg.inv(V)
    Dr = Vr @ invV
    return Dr, V

def Lift(Np,V):
    Dr, V = matriz_diferenciaçao(Np)
    Emat = np.zeros((Np, 2))
    Emat[0, 0] = 1.0   # Face esquerda
    Emat[-1, 1] = 1.0  # Face direita
    LIFT = V @ (V.T @ Emat)
    return LIFT