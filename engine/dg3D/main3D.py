import numpy as np
import meshio

from setup3D import StartUp3D
from Maxwell3D import Maxwell3D
from mesh_reader3D import MeshReader3D

N = 3

VX, VY, VZ, EToV = MeshReader3D('engine/dg3D/malhas3D/cavidade3D_grossa.msh')

malha = StartUp3D(N,EToV,VX,VY,VZ)

### Condições iniciais
#Ez = np.sin(np.pi*malha.x)*np.sin(np.pi*malha.y)
Ez = np.zeros((malha.Np,malha.K))
#Ez = np.exp(-(malha.x**2 + malha.y**2) / (0.1**2))
Hx = np.zeros((malha.Np,malha.K))
Hy = np.zeros((malha.Np,malha.K))
Hz = np.zeros((malha.Np,malha.K))
Ex = np.zeros((malha.Np,malha.K))
Ey = np.zeros((malha.Np,malha.K))

FinalTime = 2.5
Hx, Hy, Hz, Ex, Ey, Ez, t_final = Maxwell3D(Hx,Hy,Hz,Ex,Ey,Ez,FinalTime,malha,CFL=0.8,pml=False)

##############

Ez_analitico = np.sin(np.pi*malha.x)*np.sin(np.pi*malha.y)*np.cos(np.pi*np.sqrt(2)*t_final)

M = np.linalg.inv(malha.V @ malha.V.T) 

erro_nodal = Ez - Ez_analitico

integral_erro_sq = np.sum(malha.J * (erro_nodal * (M @ erro_nodal)))
En = np.sqrt(integral_erro_sq)

integral_analitico_sq = np.sum(malha.J * (Ez_analitico * (M @ Ez_analitico)))
wt = np.sqrt(integral_analitico_sq)

erro_L2_real = En / wt

#print(f't_f = {t_final:.2f}')
#print(f'L2 error for Ez: {erro_L2_real:.3e}')

print(Ez.max())

def exportar_paraview(malha, campo, nome_arquivo="resultado_3D.vtu"):
    '''Exporta os nós e o campo para visualização avançada no ParaView'''
    
    # Achatamos os dados
    x = malha.x.ravel(order='F')
    y = malha.y.ravel(order='F')
    z = malha.z.ravel(order='F')
    campo_flat = campo.ravel(order='F')
    
    pontos = np.vstack((x, y, z)).T
    
    # No DG, os nós são desconectados, então salvamos como um conjunto de vértices ("vertex")
    # np.arange(len(x)) cria uma lista de 0 até o número total de nós
    celulas = [("vertex", np.arange(len(x)).reshape(-1, 1))]
    
    # Salvamos o arquivo .vtu
    meshio.write_points_cells(
        nome_arquivo,
        pontos,
        celulas,
        point_data={"Campo_Ez": campo_flat}
    )
    print(f"Arquivo {nome_arquivo} salvo com sucesso! Abra-o no ParaView.")

exportar_paraview(malha,Ez)

#t_f = 0.50
#L2 error for Ez: 3.726e-04