import numpy as np
import matplotlib.pyplot as plt

import operadores as op
import gll1D as gll

N = 8
K = 20
Np = N + 1
Nfaces = 2
Nfp = 1
FinalTime = 10 

eps = np.ones((Np,K))*8.854*10**(-12)
mu = np.ones((Np,K))*4*np.pi*10**(-7)
sigma = np.zeros_like(eps)
Zimp = np.sqrt(mu/eps)

vmapM, vmapP, mapB, vmapB = gll.vmap(Np,K)
x, Fscale, rx, nx = gll.fator_geometrico(K,Np)
Dr, V = op.matriz_diferenciaçao(Np)
LIFT = op.Lift(Np,V)
pp = []

mask1 = (x >= 0) & (x <= 0.5)
eps[mask1] = 3*8.854*10**(-12)
sigma[mask1] = 10

dE =  np.zeros((Nfp*Nfaces,K))
dH =  np.zeros((Nfp*Nfaces,K))
ZimpM = np.zeros_like(dE)
ZimpP = np.zeros_like(dE)
YimpM = np.zeros_like(dE)
YimpP = np.zeros_like(dE)

def MaxwellRHS1D(E, H):
    
    E_flat = E.flatten(order='F')
    H_flat = H.flatten(order='F')
    
    dE = E_flat[vmapM] - E_flat[vmapP]
    dH = H_flat[vmapM] - H_flat[vmapP]
    
    Zimp_flat = Zimp.flatten(order='F')
    ZimpM = Zimp_flat[vmapM]
    ZimpP = Zimp_flat[vmapP]
    YimpM = 1 / ZimpM
    YimpP = 1 / ZimpP
    
    # ---------------------------------------------------------
    # CONDIÇÃO DE CONTORNO DE SILVER-MÜLLER (Absorvente)
    # Campos externos definidos como zero para não injetar ondas
    # ---------------------------------------------------------
    Ebc = 0.0
    Hbc = 0.0
    
    mapB_2d = np.unravel_index(mapB, dE.shape, order='F')
    
    dE[mapB_2d] = E_flat[vmapB] - Ebc
    dH[mapB_2d] = H_flat[vmapB] - Hbc
    
    # Cálculo dos fluxos Upwind
    fluxE = (nx * ZimpP * dH - dE) / (ZimpM + ZimpP)
    fluxH = (nx * YimpP * dE - dH) / (YimpM + YimpP)
    
    rhsE = (-rx * (Dr @ H) + LIFT @ (Fscale * fluxE)) / eps
    rhsH = (-rx * (Dr @ E) + LIFT @ (Fscale * fluxH)) / mu
    
    rhsE -= (sigma / eps) * E

    return rhsE, rhsH

def Maxwell1D(E, H):
    time = 0
    
    # 1. Correção: Inicializar ambos os resíduos
    resE = np.zeros((Np, K))
    resH = np.zeros((Np, K)) 
    
# Calcula a velocidade da onda em cada ponto e pega a máxima
    c_velocidade = 1.0 / np.sqrt(eps * mu)
    c_max = np.max(c_velocidade)
    
    # Menor distância entre nós na malha
    xmin = np.min(np.abs(x[0,:] - x[1,:]))
    
    # Reduzir o CFL para garantir estabilidade no DG com RK4
    CFL = 0.2
    
    # O novo dt considera a velocidade da luz!
    dt = CFL * (xmin / c_max)
    FinalTime = 1500*dt
    
    Nsteps = int(np.ceil(FinalTime / dt)) 
    dt = FinalTime / Nsteps
    
    # 2. Correção: Converter Nsteps para inteiro
    Nsteps = int(np.ceil(FinalTime / dt)) 
    dt = FinalTime / Nsteps
    
    rk4a = [0.0, -567301805773.0/1357537059087.0, -2404267990393.0/2016746695238.0, -3550918686646.0/2091501179385.0, -3270041069962.0/2362476832515.0]
    rk4b = [1432997174477.0/9575080441755.0, 5161836677717.0/13612068292357.0, 1720146321549.0/2090206949498.0, 3134564353537.0/4481467310338.0, 2277821191437.0/14882151754819.0]
    
    for n in range(Nsteps):
        # 3. Correção: range(5) para garantir os 5 estágios (índices 0 a 4)
        for INTRK in range(5): 
            rhsE, rhsH = MaxwellRHS1D(E, H)
            
            resE = rk4a[INTRK] * resE + dt * rhsE
            resH = rk4a[INTRK] * resH + dt * rhsH
            
            E += rk4b[INTRK] * resE
            H += rk4b[INTRK] * resH
            pp.append(E.flatten(order='F'))
            if n <= 500:
                E[3,0] = np.exp(-((n*dt - 1.8e-9)**2  / (0.6e-9)**2))
        time += dt
        
    return E, H

#E = np.exp(-100 * (x - 0.5)**2)
E = np.zeros_like(x)
H = np.zeros_like(E)

E, H = Maxwell1D(E,H)

plt.figure()
plt.title(f'DGTD 1D (Hesthaven)\n K = {K}, N = {N}, Tempo = {FinalTime}', fontsize=12)
plt.plot(x.flatten(order='F'), E.flatten(order='F'), label='Solução Numérica', color='royalblue')
#plt.plot(x.flatten(order='F'), np.sin(kx*x.flatten(order='F') - a_vel * FinalTime), '--', label='Exata', color='darkorange')
plt.ylabel('E(x)')
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
ax.axvline(x= 0,ls='--',color='red')
ax.axvline(x=0.5,ls='--',color='red')
plt.tight_layout()

passo = 15 # Pula de 10 em 10 frames (aumente este número para acelerar mais)

def atualizar(frame_index):
    # O frame_index agora vai pular de 10 em 10
    linha_num.set_ydata(pp[frame_index])
    return linha_num,

# O range(0, l en(pp), passo) gera os números: 0, 10, 20, 30...
ani = FuncAnimation(fig, atualizar, frames=range(0, len(pp), passo), interval=20, blit=True)

#ani.save("P2 exercício 3.gif",writer="pillow",fps=30)

plt.show()