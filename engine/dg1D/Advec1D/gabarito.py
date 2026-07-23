import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

# ==========================================
# 1. PARÂMETROS GLOBAIS
# ==========================================
N = 8          # Ordem do polinômio
K = 10         # Número de elementos
Np = N + 1     # Nós por elemento
a_vel = 2 * np.pi  # Velocidade de advecção
FinalTime = 10

# ==========================================
# 2. BASE POLINOMIAL E NÓS GLL
# ==========================================
def gll_nodes(N):
    """Calcula os pontos de Gauss-Lobatto-Legendre."""
    if N == 0: return np.array([0.0])
    x = np.zeros(Np)
    x[0], x[-1] = -1.0, 1.0
    if N > 1:
        # As raízes da derivada do Polinômio de Legendre
        x[1:-1] = legendre(N).deriv().roots
    return np.sort(x)

def legendre_norm(N, x):
    """Polinômio de Legendre Ortonormalizado."""
    return legendre(N)(x) * np.sqrt((2.0 * N + 1.0) / 2.0)

def grad_legendre_norm(N, x):
    """Derivada do Polinômio de Legendre Ortonormalizado."""
    return legendre(N).deriv()(x) * np.sqrt((2.0 * N + 1.0) / 2.0)

# Inicializa os nós no elemento de referência [-1, 1]
r = gll_nodes(N)

# ==========================================
# 3. OPERADORES MATRICIAIS (V, Dr, LIFT)
# ==========================================
V = np.zeros((Np, Np))
Vr = np.zeros((Np, Np))

for i in range(Np):
    for j in range(Np):
        V[i, j] = legendre_norm(j, r[i])
        Vr[i, j] = grad_legendre_norm(j, r[i])

invV = np.linalg.inv(V)
Dr = Vr @ invV

# Matriz LIFT (assume base ortonormal, logo M = (V @ V.T)^-1)
Emat = np.zeros((Np, 2))
Emat[0, 0] = 1.0   # Face esquerda
Emat[-1, 1] = 1.0  # Face direita
LIFT = V @ (V.T @ Emat)

# ==========================================
# 4. GERAÇÃO DA MALHA E GEOMETRIA
# ==========================================
VX = np.linspace(0, 1, K + 1)

# Mapeamento do domínio de referência para o domínio físico
x = np.zeros((Np, K))
for k in range(K):
    x[:, k] = VX[k] + 0.5 * (r + 1.0) * (VX[k+1] - VX[k])

# Fatores geométricos 1D
J = 0.5 * (VX[1:] - VX[:-1])  # Jacobiano tem tamanho (K,)
rx = 1.0 / J                  # dr/dx tem tamanho (K,)
Fscale = 1.0 / J              # Fator de escala para as faces

# Vetor Normal das faces (2 faces por elemento)
nx = np.zeros((2, K))
nx[0, :] = -1.0  # Normal apontando para fora na esquerda
nx[1, :] = 1.0   # Normal apontando para fora na direita

# ==========================================
# 5. MAPAS DE CONECTIVIDADE (CRÍTICO EM PYTHON)
# ==========================================
# vmapM: Índices dos nós das fronteiras internas do elemento
vmapM = np.zeros((2, K), dtype=int)
vmapM[0, :] = np.arange(0, K * Np, Np)       # Primeiro nó de cada elemento
vmapM[1, :] = np.arange(Np - 1, K * Np, Np)  # Último nó de cada elemento

# vmapP: Índices dos vizinhos externos (Conectividade)
vmapP = np.zeros((2, K), dtype=int)
vmapP[0, 1:] = vmapM[1, 0:-1]  # Esquerda recebe da direita do anterior
vmapP[1, 0:-1] = vmapM[0, 1:]  # Direita recebe da esquerda do próximo

# Condições de Contorno não-periódicas (Inflow/Outflow natural)
vmapP[0, 0] = vmapM[0, 0]      
vmapP[1, -1] = vmapM[1, -1]

# ==========================================
# 6. CÁLCULO DO RESÍDUO (RHS)
# ==========================================
def advecrhs1D(u, time, a):
    alpha = 0.0  # 0.0 = Upwind (Estável), 1.0 = Central (Instável)
    
    # Achata a matriz em ordem Fortran para bater com os índices do vmap
    u_flat = u.flatten(order='F')
    
    uM = u_flat[vmapM]
    uP = u_flat[vmapP]
    
    # Condição de Contorno de Inflow (Dirichlet)
    uP[0, 0] = -np.sin(a * time)
    
    # Salto de fluxo nas faces (Upwind flux)
    du = (uM - uP) * (a * nx - (1.0 - alpha) * np.abs(a * nx)) / 2.0
    
    # Termo de fluxo de superfície
    flux_term = Fscale * du
    surface_term = LIFT @ flux_term
    
    # Termo de volume
    volume_term = -a * rx * (Dr @ u)
    
    return volume_term + surface_term

# ==========================================
# 7. INTEGRADOR DE TEMPO (Runge-Kutta 45 Low-Storage)
# ==========================================
def advec1D(u, FinalTime, a):
    time = 0.0
    xmin = np.min(np.abs(x[0, :] - x[1, :]))
    
    CFL = 0.5
    dt = CFL / a * xmin
    Nsteps = int(np.ceil(FinalTime / dt))
    dt = FinalTime / Nsteps
    
    resu = np.zeros_like(u)
    
    # Coeficientes RK do Hesthaven
    rk4a = [0.0, -567301805773.0/1357537059087.0, -2404267990393.0/2016746695238.0, -3550918686646.0/2091501179385.0, -1275806237668.0/842570457699.0]
    rk4b = [1432997174477.0/9575080441755.0, 5161836677717.0/13612068292357.0, 1720146321549.0/2090206949498.0, 3134564353537.0/4481467310338.0, 2277821191437.0/14882151754819.0]
    rk4c = [0.0, 1432997174477.0/9575080441755.0, 2526269341429.0/6820363962896.0, 2006345519317.0/3224310063776.0, 2802321613138.0/2924317926251.0]

    for _ in range(Nsteps):
        for intrk in range(5):
            timelocal = time + rk4c[intrk] * dt
            rhsu = advecrhs1D(u, timelocal, a)
            resu = rk4a[intrk] * resu + dt * rhsu
            u = u + rk4b[intrk] * resu
        time += dt
        
    return u

# ==========================================
# 8. EXECUÇÃO E PLOTAGEM
# ==========================================
# Condição Inicial
u_initial = np.sin(x)

# Executa simulação
u_final = advec1D(u_initial, FinalTime, a_vel)

# Plotando
plt.figure(figsize=(8, 5))
plt.title(f'DGTD 1D (Hesthaven)\n K = {K}, N = {N}, Tempo = {FinalTime}', fontsize=12)
plt.plot(x.flatten(order='F'), u_final.flatten(order='F'), label='Solução Numérica', color='royalblue')
plt.plot(x.flatten(order='F'), np.sin(x.flatten(order='F') - a_vel * FinalTime), '--', label='Exata', color='darkorange')
plt.ylabel('u(x)')
plt.xlabel('x')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()