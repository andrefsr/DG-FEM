import numpy as np

import sys
from pathlib import Path

caminho_alvo = str(Path(__file__).resolve().parent.parent)

if caminho_alvo not in sys.path:
    sys.path.append(caminho_alvo)

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

def EquiNodes3D(p):
    """
    Gera nós equidistantes no tetraedro de referência padrão (r, s, t).
    """
    # Se a ordem for 0, retorna o centroide
    if p == 0:
        return np.array([-1.0]), np.array([-1.0]), np.array([-1.0])
        
    # Número total de nós, exatamente como na sua imagem
    N = int((p + 1) * (p + 2) * (p + 3) / 6)
    
    r = np.zeros(N)
    s = np.zeros(N)
    t = np.zeros(N)
    
    sk = 0
    for k in range(p + 1):
        for j in range(p + 1 - k):
            for i in range(p + 1 - j - k):
                # Mapeia os índices (0 até p) para o intervalo [-1, 1]
                r[sk] = -1.0 + 2.0 * i / p
                s[sk] = -1.0 + 2.0 * j / p
                t[sk] = -1.0 + 2.0 * k / p
                sk += 1
                
    return r, s, t

def Nodes3D(p):
    '''Compute Warp & Blend nodes.
    INPUT: p é a ordem polinomial do interpolante.
    OUTPUT: Vetor X,Y,Z da coordenada dos nós no tetaedro equilátero. '''

    # Escolhe o parâmetro de blending otimizado
    alphastore = np.array([0.0, 0.0, 0.0, 0.1002, 1.1332, 1.5608, 1.3413, 1.2577, 1.1603, 1.10153, 0.6080, 0.4523, 0.8856, 0.8717, 0.9655])
    if p < 15: alpha = alphastore[p]
    else: alpha = 1.0 

    # Número total de nós  e tolerância
    N = int(((p+1)*(p+2)*(p+3)) / 6)
    tol = 1e-10

    r, s, t = EquiNodes3D(p) # Cria nós equidistantes
    L1 = (1 + t)/2
    L2 = (1 + s)/2
    L3 = -1*(1 + r + s + t)/2
    L4 = (1 + r)/2

    # Define os vértices do tetraedro
    v1 = np.array([-1, -1/np.sqrt(3), -1/np.sqrt(6)])
    v2 = np.array([1, -1/np.sqrt(3), -1/np.sqrt(6)])
    v3 = np.array([0.0, 2/np.sqrt(3), -1/np.sqrt(6)])
    v4 = np.array([0.0, 0.0, 3/np.sqrt(6)])
    
    # Eixos tangentes ortogonais nas faces 1-4
    t1 = np.zeros((4,len(v1)))
    t2 = np.zeros_like(t1)

    t1[0,:] = v2 - v1
    t1[1,:] = v2 - v1
    t1[2,:] = v3 - v2
    t1[3,:] = v3 - v1

    t2[0,:] = v3 - 0.5*(v1 + v2)
    t2[1,:] = v4 - 0.5*(v1 + v2)
    t2[2,:] = v4 - 0.5*(v2 + v3)
    t2[3,:] = v4 - 0.5*(v1 + v3)

    for n in range(4): # Normalizando as tangentes
        t1[n,:] = t1[n,:]/np.abs(t1[n,:])
        t2[n,:] = t2[n,:]/np.abs(t2[n,:])

    # Warp e Blend para cada face
    XYZ = L3*v1 + L4*v2 + L2*v3 + L1*v4
    shift = np.zeros_like(XYZ)

    for face in range(1,4):
        if face == 1: La = L1; Lb = L2; Lc = L3; Ld = L4
        if face == 2: La = L2; Lb = L1; Lc = L3; Ld = L4
        if face == 3: La = L3; Lb = L1; Lc = L4; Ld = L2
        if face == 4: La = L4; Lb = L1; Lc = L3; Ld = L2

        warp1, warp2 = WarpShiftFace3D(p,alpha,alpha,La,Lb,Lc,Ld)

        blend = Lb*Lc*Ld

        denom = (Lb+0.5*La)*(Lc+0.5*La)*(Ld+0.5*La)
        ids = np.where(denom>tol)
        blend[ids] = (1+ (alpha*La[ids])**2)*blend[ids]/denom[ids]

        shift += blend*warp1*t1[face,:] + blend*warp2*t2[face,:]

        ids = np.where((La<tol) and ((Lb>tol) + (Lc>tol) + (Ld>tol) < 3))
        shift[ids,:] = warp1[ids]*t1[face] + warp2[ids]*t2[face]

    XYZ += shift

    X = XYZ[:,0]
    Y = XYZ[:,1]
    Z = XYZ[:,2]

    return X, Y, Z

def xyztorst(x,y,z):
    ''''''

    # Define os vértices do tetraedro
    v1 = np.array([-1, -1/np.sqrt(3), -1/np.sqrt(6)])
    v2 = np.array([1, -1/np.sqrt(3), -1/np.sqrt(6)])
    v3 = np.array([0.0, 2/np.sqrt(3), -1/np.sqrt(6)])
    v4 = np.array([0.0, 0.0, 3/np.sqrt(6)])

    rhs = np.concatenate(x.T,y.T,z.T) - 0.5*(v2.T + v3.T + v4.T - v1.T) @ np.ones(len(x))
    A = np.concatenate(0.5*(v2-v1).T, 0.5*(v3-v1).T, 0.5*(v4-v1).T)
    RST = A @ np.linalg.inv(rhs)

    r = RST[0,:].T
    s = RST[1,:].T
    t = RST[2,:].T


    return r, s, t