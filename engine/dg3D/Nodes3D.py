import numpy as np

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