import numpy as np
import aux_func as aux
import setup
import operators2D as op2D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

mu0 = 4*np.pi*10**(-7)
eps0 = 8.854*10**(-12)
c0 = 1/np.sqrt(eps0*mu0)

def sigmas(malha):
    # Parâmetros da PML
    p = 4.0           # Grau do polinômio (2 ou 3 são comuns)
    sigma_max = 500.0  # Força máxima da absorção nas bordas extremas (sigma_0 da imagem)
    L = 0.5           # Limite do domínio físico (onde a PML começa)

    # Inicializando matrizes de zeros com o tamanho da malha
    sigmax = np.zeros_like(malha.x)
    sigmay = np.zeros_like(malha.y)
    dx_sigmax = np.zeros_like(malha.x)
    dy_sigmay = np.zeros_like(malha.y)

    # --- Construindo a Esponja em X ---
    # Região Direita (x >= 1)
    mask_rx = malha.x >= L
    sigmax[mask_rx] = sigma_max * (malha.x[mask_rx] - L)**p
    dx_sigmax[mask_rx] = p * sigma_max * (malha.x[mask_rx] - L)**(p-1)

    # Região Esquerda (x <= -1)
    # Usamos np.abs para garantir que a base seja positiva antes de elevar a 'p'
    mask_lx = malha.x <= -L
    dist_lx = np.abs(malha.x[mask_lx] + L)
    sigmax[mask_lx] = sigma_max * (dist_lx)**p
    dx_sigmax[mask_lx] = -p * sigma_max * (dist_lx)**(p-1) # Derivada direcional em x

    # --- Construindo a Esponja em Y ---
    # Região Superior (y >= 1)
    mask_ry = malha.y >= L
    sigmay[mask_ry] = sigma_max * (malha.y[mask_ry] - L)**p
    dy_sigmay[mask_ry] = p * sigma_max * (malha.y[mask_ry] - L)**(p-1)

    # Região Inferior (y <= -1)
    mask_ly = malha.y <= -L
    dist_ly = np.abs(malha.y[mask_ly] + L)
    sigmay[mask_ly] = sigma_max * (dist_ly)**p
    dy_sigmay[mask_ly] = -p * sigma_max * (dist_ly)**(p-1)

    return sigmax, sigmay, dx_sigmax, dy_sigmay

def MaxwellRhs2D_PML(Hx, Hy, Ez, Px, Py, Qx, Qy, malha, time, sigmax, sigmay, dx_sigmax, dy_sigmay):
    '''Calcula o fluxo (lado direito) das equações de Maxwell 2D para o modo TM'''

    # 1. Achata as matrizes em 1D (ordem Fortran) para os mapas de conectividade funcionarem
    Hx_flat = Hx.flatten(order='F')
    Hy_flat = Hy.flatten(order='F')
    Ez_flat = Ez.flatten(order='F')
    
    # 2. Calcula o salto (Minus - Plus) nas faces
    dHx = Hx_flat[malha.vmapM] - Hx_flat[malha.vmapP]
    dHy = Hy_flat[malha.vmapM] - Hy_flat[malha.vmapP]
    dEz = Ez_flat[malha.vmapM] - Ez_flat[malha.vmapP]
    
    #################################################################################################
    # 3. Condição de Contorno: Condutor Elétrico Perfeito (PEC)
    # Na parede (mapB), não há salto magnético, e o salto elétrico reflete perfeitamente
    dHx[malha.mapB] = 0.0
    dHy[malha.mapB] = 0.0
    dEz[malha.mapB] = 2.0 * Ez_flat[malha.vmapB]
    #################################################################################################
    
    # 4. Retorna os saltos para o formato 2D (Nós_da_Face x Elementos) 
    # para podermos multiplicar ponto-a-ponto com os vetores normais
    shape_faces = (malha.Nfp * malha.Nfaces, malha.K)
    dHx = dHx.reshape(shape_faces, order='F')
    dHy = dHy.reshape(shape_faces, order='F')
    dEz = dEz.reshape(shape_faces, order='F')
    
    # 5. Fluxos de Fronteira (Upwind)
    alpha = 1.0
    ndotdH = malha.nx * dHx + malha.ny * dHy
    
    fluxHx =  malha.ny * dEz + alpha * (ndotdH * malha.nx - dHx)
    fluxHy = -malha.nx * dEz + alpha * (ndotdH * malha.ny - dHy) # Corrigido para -dHy
    fluxEz = -malha.nx * dHy + malha.ny * dHx - alpha * dEz
    
    # 6. Derivadas Locais (Operadores de Volume)
    # Agora passamos os argumentos que elas exigem
    Ezx, Ezy = op2D.Grad2D(Ez, malha.rx, malha.sx, malha.ry, malha.sy, malha.Dr, malha.Ds)
    CuHx, CuHy, CuHz = op2D.Curl2D(Hx, Hy, None, malha.rx, malha.sx, malha.ry, malha.sy, malha.Dr, malha.Ds)
    
    # 7. Montagem do RHS final: Volume + Fluxo(Borda)
    # Correção: LIFT exige multiplicação de matriz (@)
    rhsHx = -Ezy + malha.LIFT @ (malha.Fscale * fluxHx) / 2.0
    rhsHy =  Ezx + malha.LIFT @ (malha.Fscale * fluxHy) / 2.0
    rhsEz = CuHz + malha.LIFT @ (malha.Fscale * fluxEz) / 2.0

    ###### Bloco ADE-PML
    rhsPx = sigmax * Hy
    rhsPy = sigmay * Hx
    rhsQx = -sigmax * Qx - Hy
    rhsQy = -sigmay * Qy - Hx

    rhsHx -= sigmay*(2*Hx + Py)
    rhsHy -= sigmax*(2*Hy + Px)
    rhsEz += - dx_sigmax * Qx + dy_sigmay *Qy

    ######################
    #f = 2
    #rhsEz += 2*np.pi*f*np.sin(2.0 * np.pi * f * time)*np.exp(-(malha.x**2 + malha.y**2) / 0.1**2)
    #t0 = 0.5  # Instante em que o pulso atinge o pico
    #tau = 0.2

    #rhsEz += -2.0 * (time - t0) / (tau**2) * np.exp(-((time - t0) / tau)**2)*np.exp(-(malha.x**2 + malha.y**2) / 0.1**2)

    return rhsHx, rhsHy, rhsEz, rhsPx, rhsPy, rhsQx, rhsQy

def MaxwellRhs2D_PEC(Hx, Hy, Ez, malha, time):
    '''Calcula o fluxo (lado direito) das equações de Maxwell 2D para o modo TM'''

    # 1. Achata as matrizes em 1D (ordem Fortran) para os mapas de conectividade funcionarem
    Hx_flat = Hx.flatten(order='F')
    Hy_flat = Hy.flatten(order='F')
    Ez_flat = Ez.flatten(order='F')
    
    # 2. Calcula o salto (Minus - Plus) nas faces
    dHx = Hx_flat[malha.vmapM] - Hx_flat[malha.vmapP]
    dHy = Hy_flat[malha.vmapM] - Hy_flat[malha.vmapP]
    dEz = Ez_flat[malha.vmapM] - Ez_flat[malha.vmapP]
    
    # 3. Condição de Contorno: Condutor Elétrico Perfeito (PEC)
    # Na parede (mapB), não há salto magnético, e o salto elétrico reflete perfeitamente
    dHx[malha.mapB] = 0.0
    dHy[malha.mapB] = 0.0
    dEz[malha.mapB] = 2.0 * Ez_flat[malha.vmapB]
    
    # 4. Retorna os saltos para o formato 2D (Nós_da_Face x Elementos) 
    # para podermos multiplicar ponto-a-ponto com os vetores normais
    shape_faces = (malha.Nfp * malha.Nfaces, malha.K)
    dHx = dHx.reshape(shape_faces, order='F')
    dHy = dHy.reshape(shape_faces, order='F')
    dEz = dEz.reshape(shape_faces, order='F')
    
    # 5. Fluxos de Fronteira (Upwind)
    alpha = 1.0
    ndotdH = malha.nx * dHx + malha.ny * dHy
    fluxHx =  malha.ny * dEz + alpha * (ndotdH * malha.nx - dHx)
    fluxHy = -malha.nx * dEz + alpha * (ndotdH * malha.ny - dHy)
    fluxEz = -malha.nx * dHy + malha.ny * dHx - alpha * dEz

    # 6. Derivadas Locais (Operadores de Volume)
    Ezx, Ezy = op2D.Grad2D(Ez, malha.rx, malha.sx, malha.ry, malha.sy, malha.Dr, malha.Ds)
    CuHx, CuHy, CuHz = op2D.Curl2D(Hx, Hy, None, malha.rx, malha.sx, malha.ry, malha.sy, malha.Dr, malha.Ds)
    
    # 7. Montagem do RHS final: Volume + Fluxo(Borda)
    rhsHx = -Ezy + malha.LIFT @ (malha.Fscale * fluxHx) / 2.0
    rhsHy =  Ezx + malha.LIFT @ (malha.Fscale * fluxHy) / 2.0
    rhsEz = CuHz + malha.LIFT @ (malha.Fscale * fluxEz) / 2.0

    #f =2
    #rhsEz += 2*np.pi*f*np.sin(2.0 * np.pi * f * time)*np.exp(-(malha.x**2 + malha.y**2) / 0.1**2)
    #t0 = 0.5  # Instante em que o pulso atinge o pico
    #tau = 0.2

    #rhsEz += -2.0 * (time - t0) / (tau**2) * np.exp(-((time - t0) / tau)**2)*np.exp(-(malha.x**2 + malha.y**2) / 0.1**2)

    return rhsHx, rhsHy, rhsEz

def Maxwell2D(Hx, Hy, Ez, FinalTime, malha,CFL,pml:bool):
    '''Integrate TM-mode Maxwell's until FinalTime starting with initial conditions Hx, Hy, Ez'''
    
    # 1. Matrizes do Runge-Kutta de Baixo Armazenamento (5 estágios, 4ª ordem)
    rk4a = np.array([0.0,
                    -567301805773.0 / 1357537059087.0,
                    -2404267990393.0 / 2016746695238.0,
                    -3550918686646.0 / 2091501179385.0,
                    -1275806237668.0 / 842570457699.0]) 
    
    rk4b = np.array([1432997174477.0 / 9575080441755.0,
                    5161836677717.0 / 13612068292357.0,
                    1720146321549.0 / 2090206949498.0,
                    3134564353537.0 / 4481467310338.0,
                    2277821191437.0 / 14882151754819.0])
    
    # A matriz rk4c é usada se o seu lado direito (RHS) depender do tempo absoluto (como fontes de antena pulsantes t=time+rk4c[INTRK]*dt)
    rk4c = np.array([0.0,
                    1432997174477.0 / 9575080441755.0,
                    2526269341429.0 / 6820363962896.0, 
                    2006345519317.0 / 3224310063776.0, 
                    2802321613138.0 / 2924317926251.0])

    time = 0.0
    apml = pml
    if apml == True:
        # Inicia os campos auxiliares
        Px = np.zeros((malha.Np, malha.K))
        Py = np.zeros((malha.Np, malha.K))
        Qx = np.zeros((malha.Np, malha.K))
        Qy = np.zeros((malha.Np, malha.K))

        # Calcula os mapas de absorção da PML
        sigmax, sigmay, dx_sigmax, dy_sigmay = sigmas(malha)
        
        resPx = np.zeros((malha.Np, malha.K))
        resPy = np.zeros((malha.Np, malha.K))
        resQx = np.zeros((malha.Np, malha.K))
        resQy = np.zeros((malha.Np, malha.K))

    # DICA DE OURO: Criar a triangulação uma única vez antes do loop 
    # economiza MUITO processamento!
    #x_flat = malha.x.flatten(order='F')
    #y_flat = malha.y.flatten(order='F')
    #triangulacao = mtri.Triangulation(x_flat, y_flat)
    # ------------------------------
    
    # Registradores residuais do RK (só precisamos de um para cada variável)
    resHx = np.zeros((malha.Np, malha.K))
    resHy = np.zeros((malha.Np, malha.K))
    resEz = np.zeros((malha.Np, malha.K))
    
    # 2. Cálculo do passo de tempo (CFL)
    # Cuidado: se JacobiGQ retornar (raízes, pesos), garanta que está pegando as raízes
    rLGL, _ = aux.JacobiGQ(0, 0, malha.N)
    #rLGL = aux.JacobiGL(0, 0, malha.N)
    rmin = np.abs(rLGL[0] - rLGL[1]) 
    
    # Chamando com as variáveis corretas
    dtscale = setup.dtscale2D(malha.x, malha.y, malha.r, malha.s)
    
    # O passo de tempo básico
    cfl = CFL
    dt = cfl*np.min(dtscale) * rmin * (2.0/3.0)

    pp = []
    t = []
    erro = []
    passo = 0

    Ez_pico = np.sin(np.pi*malha.x) * np.sin(np.pi*malha.y)
    integral_pico = np.sum(malha.J * (Ez_pico**2))
    wt_ref = np.sqrt(integral_pico)

    while time < FinalTime:
        
        # Trava de segurança: impede que a simulação passe do tempo final desejado
        #if time + dt > FinalTime:
        #    dt = FinalTime - time
            
        # Loop do Runge-Kutta (Agora com 5 estágios)
        for INTRK in range(5):
            
            # (Opcional) Se sua simulação tiver um pulso que depende do tempo, 
            # o tempo local de avaliação desse pulso seria:
            t_local = time + rk4c[INTRK] * dt
            
            # Chamada do RHS com todas as dependências corretas
            if apml == True:
                rhsHx, rhsHy, rhsEz, rhsPx,rhsPy, rhsQx, rhsQy = MaxwellRhs2D_PML(Hx, Hy, Ez, Px, Py, Qx, Qy, malha, t_local, sigmax, sigmay, dx_sigmax, dy_sigmay)
            else:
                rhsHx, rhsHy, rhsEz = MaxwellRhs2D_PEC(Hx,Hy,Ez,malha,t_local)
            # Atualiza o residual

            resHx = rk4a[INTRK] * resHx + dt * rhsHx
            resHy = rk4a[INTRK] * resHy + dt * rhsHy
            resEz = rk4a[INTRK] * resEz + dt * rhsEz

            if apml == True:
                resPx = rk4a[INTRK] * resPx + dt * rhsPx
                resPy = rk4a[INTRK] * resPy + dt * rhsPy
                resQx = rk4a[INTRK] * resQx + dt * rhsQx
                resQy = rk4a[INTRK] * resQy + dt * rhsQy
            
            # Atualiza o campo principal
            Hx = Hx + rk4b[INTRK] * resHx
            Hy = Hy + rk4b[INTRK] * resHy
            Ez = Ez + rk4b[INTRK] * resEz

            if apml == True:
                Px = Px + rk4b[INTRK] * resPx
                Py = Py + rk4b[INTRK] * resPy
                Qx = Qx + rk4b[INTRK] * resQx
                Qy = Qy + rk4b[INTRK] * resQy
        
        time += dt
        passo += 1

        if passo % 5 == 0:
            print(f"Tempo atual: {time:.4e} / {FinalTime:.4e}") 
        
        t.append(time)
        pp.append(Ez.copy())

        # -------------------------------------------------------------
        # 2. DENTRO DO LOOP (onde você já estava colocando)
        # -------------------------------------------------------------
        Ez_analitico = np.sin(malha.x)*np.sin(malha.y)*np.cos(np.sqrt(2)*time)
        erro_quadrado = (Ez - Ez_analitico)**2
        integral_erro = np.sum(malha.J * erro_quadrado)
        En = np.sqrt(integral_erro) # Erro Absoluto L2

        # Divide pelo referencial fixo, e nunca mais por zero!
        erro_L2_real = En / wt_ref 
        erro.append(erro_L2_real)

        #time += dt
        #passo += 1
    print('passos =', passo)
    return Hx, Hy, Ez, pp, t, erro


