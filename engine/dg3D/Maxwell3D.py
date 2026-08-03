import numpy as np
from operators3D import Curl3D
from setup3D import dtscale3D

def sigmas3D(malha):
    # Parâmetros da PML
    p = 4.0           # Grau do polinômio (2 ou 3 são comuns)
    sigma_max = 500.0  # Força máxima da absorção nas bordas extremas (sigma_0 da imagem)
    L = 0.3           # Limite do domínio físico (onde a PML começa)

    # Inicializando matrizes de zeros com o tamanho da malha
    sigmax = np.zeros_like(malha.x)
    sigmay = np.zeros_like(malha.y)
    sigmaz = np.zeros_like(malha.z)
    dx_sigmax = np.zeros_like(malha.x)
    dy_sigmay = np.zeros_like(malha.y)
    dz_sigmaz = np.zeros_like(malha.z)

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

    # --- Construindo a Esponja em Z ---
    # Região Superior (z >= 1)
    mask_rz = malha.z >= L
    sigmaz[mask_rz] = sigma_max * (malha.z[mask_rz] - L)**p
    dz_sigmaz[mask_rz] = p * sigma_max * (malha.z[mask_rz] - L)**(p-1)

    # Região Inferior (z <= -1)
    mask_lz = malha.z <= -L
    dist_lz = np.abs(malha.z[mask_lz] + L)
    sigmaz[mask_lz] = sigma_max * (dist_lz)**p
    dz_sigmaz[mask_lz] = -p * sigma_max * (dist_lz)**(p-1)

    return sigmax, sigmay, sigmaz, dx_sigmax, dy_sigmay, dz_sigmaz

def MaxwellRHS3D_PML(Hx,Hy,Hz,Ex,Ey,Ez,Px,Py,Pz,Mx,My,Mz,malha,time,sigmax,sigmay,sigmaz):

    Hx_flat = Hx.ravel(order='F')
    Hy_flat = Hy.ravel(order='F')
    Hz_flat = Hz.ravel(order='F')
    Ex_flat = Ex.ravel(order='F')
    Ey_flat = Ey.ravel(order='F')
    Ez_flat = Ez.ravel(order='F')

    # Armazena as diferenças de campos nas faces
    dHx = Hx_flat[malha.vmapP] - Hx_flat[malha.vmapM]
    dHy = Hy_flat[malha.vmapP] - Hy_flat[malha.vmapM]
    dHz = Hz_flat[malha.vmapP] - Hz_flat[malha.vmapM]
    dEx = Ex_flat[malha.vmapP] - Ex_flat[malha.vmapM]
    dEy = Ey_flat[malha.vmapP] - Ey_flat[malha.vmapM]
    dEz = Ez_flat[malha.vmapP] - Ez_flat[malha.vmapM]

    # Condição de contorno PEC (Ez+ = - Ez-)
    dHx[malha.mapB] = 0
    dHy[malha.mapB] = 0
    dHz[malha.mapB] = 0
    dEx[malha.mapB] = -2*Ex_flat[malha.vmapB]
    dEy[malha.mapB] = -2*Ey_flat[malha.vmapB]
    dEz[malha.mapB] = -2*Ez_flat[malha.vmapB]

    shape_faces = (malha.Nfp * malha.Nfaces, malha.K)
    dHx = dHx.reshape(shape_faces, order='F')
    dHy = dHy.reshape(shape_faces, order='F')
    dHz = dHz.reshape(shape_faces, order='F')
    dEx = dEx.reshape(shape_faces, order='F')
    dEy = dEy.reshape(shape_faces, order='F')
    dEz = dEz.reshape(shape_faces, order='F')

    alpha = 1 

    ndotdH = malha.nx * dHx + malha.ny * dHy + malha.nz * dHz
    ndotdE = malha.nx * dEx + malha.ny * dEy + malha.nz * dEz

    fluxHx = - malha.ny * dEz + malha.nz * dEy + alpha * (dHx - ndotdH * malha.nx)
    fluxHy = - malha.nz * dEx + malha.nx * dEz + alpha * (dHy - ndotdH * malha.ny)
    fluxHz = - malha.nx * dEy + malha.ny * dEx + alpha * (dHz - ndotdH * malha.nz)

    fluxEx =   malha.ny * dHz - malha.nz * dHy + alpha * (dEx - ndotdE * malha.nx)
    fluxEy =   malha.nz * dHx - malha.nx * dHz + alpha * (dEy - ndotdE * malha.ny)
    fluxEz =   malha.nx * dHy - malha.ny * dHx + alpha * (dEz - ndotdE * malha.nz)

    curlHx, curlHy, curlHz = Curl3D(Hx,Hy,Hz,malha.Dr,malha.Ds,malha.Dt,malha.rx,malha.sx,malha.tx,malha.ry,malha.sy,malha.ty,malha.rz,malha.sz,malha.tz)
    curlEx, curlEy, curlEz = Curl3D(Ex,Ey,Ez,malha.Dr,malha.Ds,malha.Dt,malha.rx,malha.sx,malha.tx,malha.ry,malha.sy,malha.ty,malha.rz,malha.sz,malha.tz)
 
    rhsHx = - curlEx + malha.LIFT @ (malha.Fscale * fluxHx) / 2.0
    rhsHy = - curlEy + malha.LIFT @ (malha.Fscale * fluxHy) / 2.0
    rhsHz = - curlEz + malha.LIFT @ (malha.Fscale * fluxHz) / 2.0

    rhsEx =   curlHx + malha.LIFT @ (malha.Fscale * fluxEx) / 2.0
    rhsEy =   curlHy + malha.LIFT @ (malha.Fscale * fluxEy) / 2.0
    rhsEz =   curlHz + malha.LIFT @ (malha.Fscale * fluxEz) / 2.0

    rhsPx = rhsEx - sigmay * Px
    rhsMx = rhsHx - sigmay * Mx

    rhsEx = rhsPx + sigmax * Px - sigmaz * Ex
    rhsHx = rhsMx + sigmax * Mx - sigmaz * Hx

    rhsPy = rhsEy - sigmaz * Py
    rhsMy = rhsHy - sigmaz * My

    rhsEy = rhsPy + sigmay * Py - sigmax * Ey
    rhsHy = rhsMy + sigmay * My - sigmax * Hy

    rhsPz = rhsEz - sigmax * Pz
    rhsMz = rhsHz - sigmax * Mz

    rhsEz = rhsPz + sigmaz * Mz - sigmay * Ez
    rhsHz = rhsMz + sigmaz * Mz - sigmay * Hz

    ### Fonte

    t0 = 0.5
    tau = 0.2

    rhsEz += -2.0 * (time - t0) / (tau**2) * np.exp(-((time - t0) / tau)**2)*np.exp(-(malha.x**2 + malha.y**2) / 0.1**2)

    return rhsHx, rhsHy, rhsHz, rhsEx, rhsEy, rhsEz, rhsPx, rhsPy, rhsPz, rhsMx, rhsMy, rhsMz

def MaxwellRHS3D_PEC(Hx,Hy,Hz,Ex,Ey,Ez,malha,time):
    '''Calcula o lado direito das equações de Maxwell na formulação do DG3D'''

    #Hx_flat = Hx.flatten(order='F')
    #Hy_flat = Hy.flatten(order='F')
    #Hz_flat = Hz.flatten(order='F')
    #Ex_flat = Ex.flatten(order='F')
    #Ey_flat = Ey.flatten(order='F')
    #Ez_flat = Ez.flatten(order='F')

    # Usa .ravel() em vez de .flatten() para NÃO alocar memória nova.
    Hx_flat = Hx.ravel(order='F')
    Hy_flat = Hy.ravel(order='F')
    Hz_flat = Hz.ravel(order='F')
    Ex_flat = Ex.ravel(order='F')
    Ey_flat = Ey.ravel(order='F')
    Ez_flat = Ez.ravel(order='F')

    # Armazena as diferenças de campos nas faces
    dHx = Hx_flat[malha.vmapP] - Hx_flat[malha.vmapM]
    dHy = Hy_flat[malha.vmapP] - Hy_flat[malha.vmapM]
    dHz = Hz_flat[malha.vmapP] - Hz_flat[malha.vmapM]
    dEx = Ex_flat[malha.vmapP] - Ex_flat[malha.vmapM]
    dEy = Ey_flat[malha.vmapP] - Ey_flat[malha.vmapM]
    dEz = Ez_flat[malha.vmapP] - Ez_flat[malha.vmapM]

    # Condição de contorno PEC (Ez+ = - Ez-)
    dHx[malha.mapB] = 0
    dHy[malha.mapB] = 0
    dHz[malha.mapB] = 0
    dEx[malha.mapB] = -2*Ex_flat[malha.vmapB]
    dEy[malha.mapB] = -2*Ey_flat[malha.vmapB]
    dEz[malha.mapB] = -2*Ez_flat[malha.vmapB]

    shape_faces = (malha.Nfp * malha.Nfaces, malha.K)
    dHx = dHx.reshape(shape_faces, order='F')
    dHy = dHy.reshape(shape_faces, order='F')
    dHz = dHz.reshape(shape_faces, order='F')
    dEx = dEx.reshape(shape_faces, order='F')
    dEy = dEy.reshape(shape_faces, order='F')
    dEz = dEz.reshape(shape_faces, order='F')

    alpha = 1 

    ndotdH = malha.nx * dHx + malha.ny * dHy + malha.nz * dHz
    ndotdE = malha.nx * dEx + malha.ny * dEy + malha.nz * dEz

    fluxHx = - malha.ny * dEz + malha.nz * dEy + alpha * (dHx - ndotdH * malha.nx)
    fluxHy = - malha.nz * dEx + malha.nx * dEz + alpha * (dHy - ndotdH * malha.ny)
    fluxHz = - malha.nx * dEy + malha.ny * dEx + alpha * (dHz - ndotdH * malha.nz)

    fluxEx =   malha.ny * dHz - malha.nz * dHy + alpha * (dEx - ndotdE * malha.nx)
    fluxEy =   malha.nz * dHx - malha.nx * dHz + alpha * (dEy - ndotdE * malha.ny)
    fluxEz =   malha.nx * dHy - malha.ny * dHx + alpha * (dEz - ndotdE * malha.nz)

    curlHx, curlHy, curlHz = Curl3D(Hx,Hy,Hz,malha.Dr,malha.Ds,malha.Dt,malha.rx,malha.sx,malha.tx,malha.ry,malha.sy,malha.ty,malha.rz,malha.sz,malha.tz)
    curlEx, curlEy, curlEz = Curl3D(Ex,Ey,Ez,malha.Dr,malha.Ds,malha.Dt,malha.rx,malha.sx,malha.tx,malha.ry,malha.sy,malha.ty,malha.rz,malha.sz,malha.tz)
 
    rhsHx = - curlEx + malha.LIFT @ (malha.Fscale * fluxHx) / 2.0
    rhsHy = - curlEy + malha.LIFT @ (malha.Fscale * fluxHy) / 2.0
    rhsHz = - curlEz + malha.LIFT @ (malha.Fscale * fluxHz) / 2.0

    rhsEx =   curlHx + malha.LIFT @ (malha.Fscale * fluxEx) / 2.0
    rhsEy =   curlHy + malha.LIFT @ (malha.Fscale * fluxEy) / 2.0
    rhsEz =   curlHz + malha.LIFT @ (malha.Fscale * fluxEz) / 2.0

    return rhsHx, rhsHy, rhsHz, rhsEx, rhsEy, rhsEz

def Maxwell3D(Hx,Hy,Hz,Ex,Ey,Ez,FinalTime,malha,CFL,pml:bool):
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
        Pz = np.zeros((malha.Np, malha.K))
        Mx = np.zeros((malha.Np, malha.K))
        My = np.zeros((malha.Np, malha.K))
        Mz = np.zeros((malha.Np, malha.K))

        # Calcula os mapas de absorção da PML
        sigmax, sigmay, sigmaz, _, _, _, = sigmas3D(malha)
        
        resPx = np.zeros((malha.Np, malha.K))
        resPy = np.zeros((malha.Np, malha.K))
        resPz = np.zeros((malha.Np, malha.K))
        resMx = np.zeros((malha.Np, malha.K))
        resMy = np.zeros((malha.Np, malha.K))
        resMz = np.zeros((malha.Np, malha.K))

    # Registradores residuais do RK (só precisamos de um para cada variável)
    resHx = np.zeros((malha.Np, malha.K))
    resHy = np.zeros((malha.Np, malha.K))
    resHz = np.zeros((malha.Np, malha.K))
    resEx = np.zeros((malha.Np, malha.K))
    resEy = np.zeros((malha.Np, malha.K))
    resEz = np.zeros((malha.Np, malha.K))

    dtscale = dtscale3D(malha.J,malha.sJ)
    dt = (CFL * np.min(dtscale)) / (malha.N**2) # !!! dtscale já pode ser um mínimo

    while time < FinalTime:

        for INTRK in range(5):
            t_local = time + rk4c[INTRK] * dt

            if apml == True:
                rhsHx, rhsHy, rhsHz, rhsEx, rhsEy, rhsEz, rhsPx, rhsPy, rhsPz, rhsMx, rhsMy, rhsMz = MaxwellRHS3D_PML(Hx,Hy,Hz,Ex,Ey,Ez,Px,Py,Pz,Mx,My,Mz,malha,t_local,sigmax,sigmay,sigmaz)
            else:
                rhsHx, rhsHy, rhsHz, rhsEx, rhsEy, rhsEz = MaxwellRHS3D_PEC(Hx,Hy,Hz,Ex,Ey,Ez,malha,t_local)

            # Atualiza o residual
            resHx = rk4a[INTRK] * resHx + dt * rhsHx
            resHy = rk4a[INTRK] * resHy + dt * rhsHy
            resHz = rk4a[INTRK] * resHz + dt * rhsHz
            resEx = rk4a[INTRK] * resEx + dt * rhsEx
            resEy = rk4a[INTRK] * resEy + dt * rhsEy
            resEz = rk4a[INTRK] * resEz + dt * rhsEz

            if apml == True:
                resPx = rk4a[INTRK] * resPx + dt * rhsPx
                resPy = rk4a[INTRK] * resPy + dt * rhsPy
                resPz = rk4a[INTRK] * resPz + dt * rhsPz
                resMx = rk4a[INTRK] * resMx + dt * rhsMx
                resMy = rk4a[INTRK] * resMy + dt * rhsMy
                resMz = rk4a[INTRK] * resMz + dt * rhsMz

            # Atualiza o campo principal
            Hx = Hx + rk4b[INTRK] * resHx
            Hy = Hy + rk4b[INTRK] * resHy
            Hz = Hz + rk4b[INTRK] * resHz
            Ex = Ex + rk4b[INTRK] * resEx
            Ey = Ey + rk4b[INTRK] * resEy
            Ez = Ez + rk4b[INTRK] * resEz

            if apml == True:
                Px += rk4b[INTRK] * resPx
                Py += rk4b[INTRK] * resPy
                Pz += rk4b[INTRK] * resPz
                Mx += rk4b[INTRK] * resMx
                My += rk4b[INTRK] * resMy
                Mz += rk4b[INTRK] * resMz

        time += dt
        print(f"Tempo atual: {time:.4e} / {FinalTime:.2e}") 
    
    return Hx, Hy, Hz, Ex, Ey, Ez, time
