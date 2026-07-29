import numpy as np
from aux_func3D import Simplex3DP
from Nodes3D import rsttoabc

import sys
from pathlib import Path

caminho_alvo = str(Path(__file__).resolve().parent.parent)

if caminho_alvo not in sys.path:
    sys.path.append(caminho_alvo)

from dg2D.aux_func import JacobiP, GradJacobiP
from dg2D.operators2D import Vandermonde2D

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


def calcular_fmask3D(r, s, t, tol=1e-10):
    """
    Constrói a matriz de índices Fmask para o triângulo de referência.
    Retorna uma matriz onde cada coluna contém os índices dos nós 
    pertencentes a uma das faces.
    """
    # 1. Encontra os índices (IDs) dos pontos que satisfazem a geometria da face
    # O [0] no final serve para extrair o array de dentro da tupla que o np.where retorna
    fmask1 = np.where(np.abs(t + 1.0) < tol)[0]
    fmask2 = np.where(np.abs(s + 1.0) < tol)[0]
    fmask3 = np.where(np.abs(r + s + t + 1.0) < tol)[0]
    fmask4 = np.where(np.abs(r + 1.0) < tol)[0] 

    Fmask = np.column_stack((fmask1, fmask2, fmask3,fmask4))
    
    return Fmask

def Lift3D(N,r,s,t,Fmask, V):
    '''COmputa o operador de superfície para volume em 3D na formulação do DG'''

    Np = (N+1)*(N+2)*(N+3)/6
    Nfp = (N+1)*(N+2)/2
    Nfaces = 4

    Emat = np.zeros((int(Np),int(Nfaces*Nfp)))

    for face in range(N):
        if face == 0: faceR = r[Fmask[:,0]]; faceS = s[Fmask[:,0]]
        if face == 1: faceR = r[Fmask[:,1]]; faceS = t[Fmask[:,1]]
        if face == 2: faceR = s[Fmask[:,2]]; faceS = t[Fmask[:,2]]
        if face == 3: faceR = s[Fmask[:,3]]; faceS = t[Fmask[:,3]]

        VFace = Vandermonde2D(N,faceR, faceS)
        massinv = VFace @ VFace.T
        massFace = np.linalg.inv(massinv)

        idr = Fmask[:,face]
        idc = np.arange(face * Nfp, (face + 1) * Nfp)

        Emat[idr,idc] = Emat[idr,idc] + massFace

    LIFT = V @ (V.T @ Emat)

    return LIFT

def Grad3D(U,Dr,Ds,Dt,rx,sx,tx,ry,sy,ty,rz,sz,tz):
    '''Computa as derivadas espaciais físicas elementais de U'''

    # Derivadas no tetraedro de referência
    dUdr = Dr @ U
    dUds = Ds @ U
    dUdt = Dt @ U

    # Toma as derivadas espaciais físicas usando regra da cadeia
    dUdx = rx * dUdr + sx * dUds + tx * dUdt
    dUdy = ry * dUdr + sy * dUds + ty * dUdt
    dUdz = rz * dUdr + sz * dUds + tz * dUdt

    return dUdx, dUdy, dUdz

def Div3D(Ux,Uy,Uz,Dr,Ds,Dt,rx,sx,tx,ry,sy,ty,rz,sz,tz):
    '''Computa o divergente espacial físico do vetor (Ux,Uy,Uz) '''

    # Derivada local de Ux no tetraedro de referência
    ddr = Dr @ Ux
    dds = Ds @ Ux
    ddt = Dt @ Ux

    # dUx/dx
    divU = rx * ddr + sx * dds + tx * ddt

    # Derivada local de Uy no tetraedro de referência
    ddr = Dr @ Uy
    dds = Ds @ Uy
    ddt = Dt @ Uy

    # Adiciona dUy/dy ao divergente
    divU += ry * ddr + sy * dds + ty * ddt

    # Derivada local de Uz no tetraedro de referência
    ddr = Dr @ Uz
    dds = Ds @ Uz
    ddt = Dt @ Uz

    # Adiciona dUz/dz ao divergente
    divU += rz * ddr + sz * dds + tz * ddt

    return divU

def Curl3D(Ux,Uy,Uz,Dr,Ds,Dt,rx,sx,tx,ry,sy,ty,rz,sz,tz):
    '''Computa o rotacional espacial de (Ux,Uy,Uz)'''

    # Derivada local de Ux no tetraedro de referência
    ddr = Dr @ Ux
    dds = Ds @ Ux
    ddt = Dt @ Ux

    # Adiciona os componentes do rotacional
    curly = (rz * ddr + sz * dds + tz * ddt)
    curlz = - (ry * ddr + sy * dds + ty * ddt)

    # Derivada local de Uy no tetraedro de referência
    ddr = Dr @ Uy
    dds = Ds @ Uy
    ddt = Dt @ Uy

    # Adiciona os componentes do rotacional
    curlx = - (rz * ddr + sz * dds + tz * ddt)
    curlz += (rx * ddr + sx * dds + tx * ddt)

    # Derivada local de Uz no tetraedro de referência
    ddr = Dr @ Uz
    dds = Ds @ Uz
    ddt = Dt @ Uz

    # Adiciona os componentes do rotacional
    curlx += (ry * ddr + sy * dds + ty * ddt)
    curly += - (rx * ddr + sx * dds + tx * ddt)

    return curlx, curly, curlz



