import numpy as np
from dg2D.aux_func import JacobiGL

def rsttoabc(r,s,t):
    '''Transfer from (r,s,t) -> to (a,b,c) coordinates in triangle'''

    Np = len(r)
    a = np.zeros(Np)
    b = np.zeros(Np)
    c = np.zeros(Np)

    for n in range(Np):

        if (s[n] + t[n]) != 0:
            a[n] = 2*(1 + r[n])/(-s[n] -t[n]) - 1
        else:
            a[n] = -1

        if t[n] != -1:
            b[n] = 2*(1+s[n])/(1 -t[n]) -1
        else:
            b[n] = -1

    c = t

    return a, b, c

def evalwarp(p, rout, x):
    """
    Computa a distorção (warp) 1D usando polinômios de Lagrange.
    
    Parâmetros:
    - p: Ordem polinomial (N)
    - rout: Nós de referência 1D (ex: nós GLL) de tamanho (p+1,)
    - x: Coordenadas arbitrárias onde o warp será avaliado (array NumPy)
    
    Retorno:
    - eval: O valor do warp calculado para cada ponto em x.
    """
    # 1. Cria uma grade de pontos equidistantes (r) na mesma extensão de rout [-1, 1]
    # No MATLAB do Hesthaven, isso costuma ser equivalente a np.linspace(-1, 1, p+1)
    req = np.linspace(-1, 1, p + 1)
    
    # 2. Inicializa a matriz de Lagrange com zeros (tamanho len(x) por p+1)
    # Cada coluna i representa a i-ésima função base de Lagrange L_i(x)
    V = np.zeros((len(x), p + 1))
    
    for i in range(p + 1):
        # Inicializa a base de Lagrange com 1.0 para cada ponto em x
        L = np.ones_like(x, dtype=float)
        for j in range(p + 1):
            if i != j:
                # Produto produtorio da base de Lagrange: L_i(x) = Prod_{j != i} (x - req[j]) / (req[i] - req[j])
                L = L * (x - req[j]) / (req[i] - req[j])
        V[:, i] = L

    # 3. A diferença entre os nós GLL (rout) e os nós equidistantes (req) é o deslocamento base
    # (rout - req) vetor linha (p+1,)
    warp_base = rout - req
    
    # 4. Multiplica a matriz de Lagrange pelo vetor de deslocamentos para avaliar nos pontos x
    eval = np.dot(V, warp_base)
    
    return eval

def evalshift(p,pval,L1,L2,L3):
    '''Compute two-dimensional Warp & Blend transform'''

    # 1) Computa a distribuição GLL nodal 
    gaussX = -JacobiGL(0,0,p)

    # 2) Computa a função blending em cada nó de cada aresta
    blend1 = L2*L3
    blend2 = L1*L3
    blend3 = L1*L2

    # 3) Quantidade de "warp" para cada nó de cada aresta
    warpfactor1 = 4*evalwarp(p,gaussX,L3-L2)
    warpfactor2 = 4*evalwarp(p,gaussX,L1-L3)
    warpfactor3 = 4*evalwarp(p,gaussX,L2-L1)

    # 4) Combinação de warp e blend
    warp1 = blend1*warpfactor1*(1+ (pval*L1)**2)
    warp2 = blend2*warpfactor2*(1+ (pval*L2)**2)
    warp3 = blend3*warpfactor3*(1+ (pval*L3)**2)

    # 5) Calcula o deslocamento no triângulo equilátero
    dx = 1*warp1 + np.cos(2*(np.pi/3))*warp2 + np.cos(4*(np.pi/3))*warp3
    dy = 0*warp1 + np.sin(2*(np.pi/3))*warp2 + np.sin(4*(np.pi/3))*warp3

    return dx, dy

def WarpShiftFace3D(p,pval,pval2,L1,L2,L3,L4):
    '''Compute warp factor used in creating 3D Warp & Blend nodes'''

    dtan1, dtan2 = evalshift(p,pval,L2,L3,L4)
    warpx = dtan1
    warpy = dtan2

    return warpx, warpy

def Nodes3D(p):

    

    return X, Y, Z