from types import SimpleNamespace
import numpy as np
from Nodes3D import Nodes3D, xyztorst
from operators3D import Vandermonde3D, Dmatrices3D, calcular_fmask3D, Lift3D, GradVandermonde3D
from grid3D import GeometricFactors3D, Normals3D, tiConnect3D, BuildMaps3D

def StartUp3D(N,EToV,VX,VY,VZ):

    Np = int((N+1)*(N+2)*(N+3)/6)
    Nfp = int((N+1)*(N+2)/2)
    Nfaces = 4
    NODETOOL = 1e-7
    K = EToV.shape[0]

    x, y, z = Nodes3D(N)
    r, s, t = xyztorst(x,y,z)

    V = Vandermonde3D(N,r,s,t)
    invV = np.linalg.inv(V)
    MassMatrix = invV.T @ invV
    Dr, Ds, Dt = Dmatrices3D(N,r,s,t,V)

    va = EToV[:,0].T
    vb = EToV[:,1].T
    vc = EToV[:,2].T
    vd = EToV[:,3].T

    r_col = r[:, None]
    s_col = s[:, None]
    t_col = t[:, None]

    x = 0.5*(-(1+r_col+s_col+t_col)*VX[va]+(1+r_col)*VX[vb]+(1+s_col)*VX[vc]+(1+t_col)*VX[vd])
    y = 0.5*(-(1+r_col+s_col+t_col)*VY[va]+(1+r_col)*VY[vb]+(1+s_col)*VY[vc]+(1+t_col)*VY[vd])
    z = 0.5*(-(1+r_col+s_col+t_col)*VZ[va]+(1+r_col)*VZ[vb]+(1+s_col)*VZ[vc]+(1+t_col)*VZ[vd])

    Fmask = calcular_fmask3D(r,s,t,NODETOOL)
    fmask_flat = Fmask.flatten(order='F')
    Fx = x[fmask_flat, :]
    Fy = y[fmask_flat, :]
    Fz = z[fmask_flat, :]

    LIFT = Lift3D(N,r,s,t,Fmask,V)

    rx, ry, rz, sx, sy, sz, tx, ty, tz, J = GeometricFactors3D(x,y,z,Dr,Ds,Dt)

    nx, ny, nz, sJ = Normals3D(rx,ry,rz,sx,sy,sz,tx,ty,tz,J,Fmask,N,K)
    Fscale = sJ/J[fmask_flat,:]

    EToE, EToF = tiConnect3D(EToV)

    vmapM, vmapP, vmapB, mapB = BuildMaps3D(K,Np,Nfp,Nfaces,Fmask,EToE,EToF,x,y,z,NODETOOL)

    Vr, Vs, Vt = GradVandermonde3D(N,r,s,t)

    Drw = (V @ Vr.T) @ np.linalg.inv(V @ V.T)
    Dsw = (V @ Vs.T) @ np.linalg.inv(V @ V.T)
    Dtw = (V @ Vt.T) @ np.linalg.inv(V @ V.T)


    # Empacota TODAS as variáveis da malha em um único objeto organizado
    malha = SimpleNamespace(
        N=N, Np=Np, Nfp=Nfp, Nfaces=Nfaces, K=K,
        r=r, s=s, t=t, x=x, y=y, z=z,
        V=V, invV=invV, MassMatrix=MassMatrix, 
        Dr=Dr, Ds=Ds, Dt=Dt, Drw=Drw, Dsw=Dsw, Dtw=Dtw, LIFT=LIFT,
        rx=rx, sx=sx, tx=tx, ry=ry, sy=sy, ty=ty, rz=rz, sz=sz, tz=tz, J=J,
        nx=nx, ny=ny, nz=nz, sJ=sJ, Fscale=Fscale,
        EToE=EToE, EToF=EToF, Fmask=Fmask, Fx=Fx, Fy=Fy, Fz=Fz, 
        vmapM=vmapM, vmapP=vmapP, vmapB=vmapB, mapB=mapB #,mapM=mapM, mapP=mapP
    )
    
    return malha

def dtscale3D(J,sJ):
    '''Calcula o raio do círculo inscrito de cada triângulo da malha 
    para ser usado na condição de estabilidade CFL (passo de tempo).'''

    J_elem = J[0, :]
    sJ_max_elem = np.max(sJ, axis=0)  

    dtscale_elem = J_elem / sJ_max_elem  

    dtscale_global = np.min(dtscale_elem)
    
    return dtscale_global