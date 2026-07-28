import numpy as np
from aux_func3D import Simplex3DP
from Nodes3D import rsttoabc

import sys
from pathlib import Path

caminho_alvo = str(Path(__file__).resolve().parent.parent)

if caminho_alvo not in sys.path:
    sys.path.append(caminho_alvo)

from dg2D.aux_func import JacobiP, GradJacobiP

def Vandermonde3D(N,r,s,t):
    '''Inicializa a matriz de Vandermonde 3D onde V_{ij} = phi_j{r_i,s_i,t_i}'''

    Np = int(((N+1)*(N+2)*(N+3)) / 6)

    V3D = np.zeros((len(r),Np))

    a, b, c = rsttoabc(r,s,t)

    sk = 0
    for i in range(N + 1):
        for j in range(N + 1 -i):
            for k in range(N +1 -i -j):
                V3D[:,sk] = Simplex3DP(a,b,c,i,j,k)
                sk += 1

    return V3D

def GradSimplex3DP(a,b,c,id,jd,kd):
    '''Propósito: Retorna as derivadas da base modal no simplexo 3D (pg 419 HestHaven)'''

    fa = JacobiP(a,0,0,id)
    gb = JacobiP(b,2*id + 1,0,jd)
    hc = JacobiP(c,2*(id+jd)+2,0,kd)

    dfa = GradJacobiP(a,0,0,id)
    dgb = GradJacobiP(b,2*id + 1,0,jd)
    dhc = GradJacobiP(c,2*(id+jd)+2,0,kd)

    # derivada em r
    V3Dr = dfa*(gb*hc)
    if id > 0: V3Dr = V3Dr*((0.5*(1-b))**(id-1))
    if id + jd > 0: V3Dr = V3Dr*((0.5*(1-c))**(id+jd-1))

    # derivada em s
    V3Ds = 0.5*(1+a)*V3Dr
    tmp = dgb*((0.5*(1-b))**(id))
    if id > 0: tmp += (-0.5*id)*(gb*(0.5*(1-b))**(id-1))
    if id + jd > 0: tmp = tmp*((0.5*(1-c))**(id+jd-1))
    tmp = fa*(tmp*hc)
    V3Ds += tmp

    # derivada em t
    V3Dt = 0.5*(1+a)*V3Dr+0.5*(1+b)*tmp
    tmp = dhc*((0.5*(1-c))**(id+jd))
    if(id+jd>0): tmp = tmp-0.5*(id+jd)*(hc*((0.5*(1-c))**(id+jd-1)))
    tmp = fa*(gb*tmp)
    tmp = tmp*((0.5*(1-b))**id)
    V3Dt = V3Dt+tmp

    V3Dr = V3Dr*(2**(2*id+jd+1.5))
    V3Ds = V3Ds*(2**(2*id+jd+1.5))
    V3Dt = V3Dt*(2**(2*id+jd+1.5))

    return V3Dr, V3Ds, V3Dt

def GradVandermonde3D(N,r,s,t):
    '''Inicializa o gradiente da base modal (i,j,k) em (r,s,t) de ordem N'''

    Np = int((N+1)*(N+2)*(N+3)/6)
    V3Dr = np.zeros((len(r), Np))
    V3Ds = np.zeros_like(V3Dr)
    V3Dt = np.zeros_like(V3Dr)

    a, b, c = rsttoabc(r, s, t)

    sk = 0
    for i in range(N+1):
        for j in range(N+1-i):
            for k in range(N+1-i-j):
                V3Dr[:,sk], V3Ds[:,sk], V3Dt[:,sk] = GradSimplex3DP(a,b,c,i,j,k)
                sk += 1

    return V3Dr, V3Ds, V3Dt

def Dmatrices3D(N,r,s,t,V):
    '''Inicializa as matrizes de diferenciação no simplexo'''

    Vr, Vs, Vt = GradVandermonde3D(N,r,s,t)
    invV = np.linalg.inv(V)
    Dr = Vr @ invV
    Ds = Vs @ invV
    Dt = Vt @ invV

    return Dr, Ds, Dt

