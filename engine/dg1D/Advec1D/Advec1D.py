import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

import gll1D as gll 
import operadores as op 

N = 8
K = 10
Np = N + 1
a_vel = 1 
FinalTime = 10

vmapM, vmapP = gll.vmap(Np,K)
x, Fscale, rx, nx = gll.fator_geometrico(K,Np)
Dr, V = op.matriz_diferenciaçao(Np)
LIFT = op.Lift(Np,V)

def advecrhs1D(u, time, a):
    alpha = 0.0
    
    u_flat = u.flatten(order='F')
    
    uM = u_flat[vmapM]
    uP = u_flat[vmapP]
    
    #uP[0, 0] = -np.sin(a * time)
    
    du = (uM - uP) * (a * nx - (1.0 - alpha) * np.abs(a * nx)) / 2.0
    
    flux_term = Fscale * du
    surface_term = LIFT @ flux_term
    
    volume_term = -a * rx * (Dr @ u)
    
    return volume_term + surface_term

def advec1D(u, FinalTime, a):
    pp = []
    time = 0.0
    xmin = np.min(np.abs(x[0, :] - x[1, :]))
    
    CFL = 0.5
    dt = CFL / np.abs(a) * xmin
    Nsteps = int(np.ceil(FinalTime / dt))
    dt = FinalTime / Nsteps
    
    resu = np.zeros_like(u)
    
    rk4a = [0.0, -567301805773.0/1357537059087.0, -2404267990393.0/2016746695238.0, -3550918686646.0/2091501179385.0, -1275806237668.0/842570457699.0]
    rk4b = [1432997174477.0/9575080441755.0, 5161836677717.0/13612068292357.0, 1720146321549.0/2090206949498.0, 3134564353537.0/4481467310338.0, 2277821191437.0/14882151754819.0]
    rk4c = [0.0, 1432997174477.0/9575080441755.0, 2526269341429.0/6820363962896.0, 2006345519317.0/3224310063776.0, 2802321613138.0/2924317926251.0]

    for n in range(Nsteps):

        for intrk in range(5):
            timelocal = time + rk4c[intrk] * dt
            rhsu = advecrhs1D(u, timelocal, a)
            resu = rk4a[intrk] * resu + dt * rhsu
            u = u + rk4b[intrk] * resu
            pp.append(u.flatten(order='F'))
        #u[0,0] =  -np.sin(2*np.pi*n*dt)
        time += dt
        
    return u, pp

#u_initial = np.sin(x)
kx = 2*np.pi
u_initial = np.sin(kx*x)

#u_initial = np.exp(-100 * (x - 0.5)**2)
#u_initial = np.zeros_like(x)
#mask1 = (x >= 0.25) & (x <= 0.5)
#u_initial[mask1] = 4*x[mask1] -1
#mask2 = (x >= 0.5) & (x <= 0.75)
#u_initial[mask2] = -4*x[mask2] + 3

#mask =(x >= 0.3) & (x <= 0.7)
#u_initial[mask] = 1

VX = np.linspace(0, 1, K + 1)


plt.figure()
plt.plot(VX,np.zeros_like(VX),'o')
plt.plot(x.flatten(order='F'),u_initial.flatten(order='F'))
plt.show()

u_final, pp = advec1D(u_initial, FinalTime, a_vel)

plt.figure()
plt.title(f'DGTD 1D (Hesthaven)\n K = {K}, N = {N}, Tempo = {FinalTime}', fontsize=12)
plt.plot(x.flatten(order='F'), u_final.flatten(order='F'), label='Solução Numérica', color='royalblue')
#plt.plot(x.flatten(order='F'), np.sin(kx*x.flatten(order='F') - a_vel * FinalTime), '--', label='Exata', color='darkorange')
plt.ylabel('u(x)')
plt.xlabel('x')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Ajustando o eixo X
# Usamos o mesmo achatamento (flatten) que você usou no seu plot estático
x_flat = x.flatten(order='F')

# 2. Configuração inicial da figura
fig, ax = plt.subplots()
ax.set_title(f'DGTD 1D (Hesthaven) - Animação\nK = {K}, N = {N}', fontsize=12)
ax.set_xlabel('x')
ax.set_ylabel('u(x)')
ax.grid(alpha=0.3)

# Fixar os limites do eixo Y é crucial em animações para o gráfico não "pular"
# Como a condição inicial é um np.sin(x), os valores variam de -1 a 1.
ax.set_ylim(-1.5, 1.5) 
ax.set_xlim(x_flat.min(), x_flat.max())

# Criamos a linha vazia (pegamos o primeiro estado de pp)
linha_num, = ax.plot(x_flat, pp[0], color='royalblue', label='Solução Numérica')
ax.legend()
plt.tight_layout()

passo = 15 # Pula de 10 em 10 frames (aumente este número para acelerar mais)

def atualizar(frame_index):
    # O frame_index agora vai pular de 10 em 10
    linha_num.set_ydata(pp[frame_index])
    return linha_num,

# O range(0, len(pp), passo) gera os números: 0, 10, 20, 30...
ani = FuncAnimation(fig, atualizar, frames=range(0, len(pp), passo), interval=20, blit=True)

ani.save("exercicio6.gif",writer="pillow",fps=30)


plt.show()
