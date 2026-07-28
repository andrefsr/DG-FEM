import numpy as np
from dg2D.aux_func import JacobiP

def Simplex3DP(a,b,c,i,j,k):
    '''Evaluate 3D orthonormal polynonual on symplex at (a,b,c) of order (i,j,k)'''

    h1 = JacobiP(a,0,0,i)
    h2 = JacobiP(b,2*i+1,0,j)
    h3 = JacobiP(c,2*(i+j)+2,0,k)
    P = 2*np.sqrt(2)*h1*h2*((1-b)**i)*h3*((1-c)**(i+j))

    return P