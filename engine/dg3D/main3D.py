import numpy as np
from setup3D import StartUp3D
from Maxwell3D import Maxwell3D
from mesh_reader3D import MeshReader3D

N = 5

VX, VY, VZ, EToV = MeshReader3D('engine/dg3D/malhas3D/cavidade3D.msh')

malha = StartUp3D(N,EToV,VX,VY,VZ)

### Condições iniciais
Ez = np.sin(np.pi*malha.x)*np.sin(np.pi*malha.y)
#Ez = np.zeros((malha.Np,malha.K))
#Ez = np.exp(-(malha.x**2 + malha.y**2) / (0.1**2))
Hx = np.zeros((malha.Np,malha.K))
Hy = np.zeros((malha.Np,malha.K))
Hz = np.zeros((malha.Np,malha.K))
Ex = np.zeros((malha.Np,malha.K))
Ey = np.zeros((malha.Np,malha.K))

FinalTime = 5
Hx, Hy, Hz, Ex, Ey, Ez = Maxwell3D(Hx,Hy,Hz,Ex,Ey,Ez,FinalTime,malha,CFL=0.2,pml=False)