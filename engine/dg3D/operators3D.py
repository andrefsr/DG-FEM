import numpy as np
from aux_func3D import Simplex3DP
from Nodes3D import rsttoabc

def Vandermonde3D(N,r,s,t):

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

