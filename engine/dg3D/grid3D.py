import numpy as np

def GeometricFactors3D(x,y,z,Dr,Ds,Dt):
    '''Computa a métrica para o mapeamento local dos elementos'''

    xr = Dr @ x
    xs = Ds @ x
    xt = Dt @ x
    yr = Dr @ y
    ys = Ds @ y
    yt = Dt @ y
    zr = Dr @ z
    zs = Ds @ z
    zt = Dt @ z

    J = xr*(ys*zt-zs*yt)- yr*(xs*zt-zs*xt) + zr*(xs*yt-ys*xt)
    rx = (ys*zt- zs*yt)/J
    ry =-(xs*zt- zs*xt)/J
    rz = (xs*yt- ys*xt)/J
    sx =-(yr*zt- zr*yt)/J
    sy = (xr*zt- zr*xt)/J
    sz =-(xr*yt- yr*xt)/J
    tx = (yr*zs- zr*ys)/J
    ty =-(xr*zs- zr*xs)/J
    tz = (xr*ys- yr*xs)/J

    return rx, ry, rz, sx, sy, sz, tx, ty, tz, J 

def Normals3D(rx,ry,rz,sx,sy,sz,tx,ty,tz,J,Fmask,N,K):
    '''Computa as normais apontando para fora nos elementos das faces assim como os Jacobianos de superfície'''

    Nfp = int((N+1)*(N+2)/2)

    # O comando MATLAB Fmask(:) achata a matriz lendo por colunas (padrão Fortran).
    # Para replicar isso no NumPy, usamos flatten('F').
    fmask_flat = Fmask.flatten('F') 

    # Interpolar fatores geométricos para os nós das faces
    frx = rx[fmask_flat, :]
    fsx = sx[fmask_flat, :]
    ftx = tx[fmask_flat, :]

    fry = ry[fmask_flat, :]
    fsy = sy[fmask_flat, :]
    fty = ty[fmask_flat, :]

    frz = rz[fmask_flat, :]
    fsz = sz[fmask_flat, :]
    ftz = tz[fmask_flat, :]

    # Construir normais
    nx = np.zeros((int(4 * Nfp), K))
    ny = np.zeros((int(4 * Nfp), K))
    nz = np.zeros((int(4 * Nfp), K))

    # No Python, em vez de criar vetores de índices como o MATLAB faz (ex: fid1 = (1:Nfp)'),
    # usamos o objeto nativo 'slice' do Python. Ele é mais rápido e gasta menos memória.
    # Lembrando que Python começa no índice 0 e o último número é exclusivo [início : fim).
    fid1 = slice(0, Nfp)
    fid2 = slice(Nfp, 2 * Nfp)
    fid3 = slice(2 * Nfp, 3 * Nfp)
    fid4 = slice(3 * Nfp, 4 * Nfp)

    # Face 1
    nx[fid1, :] = -ftx[fid1, :]
    ny[fid1, :] = -fty[fid1, :]
    nz[fid1, :] = -ftz[fid1, :]

    # Face 2
    nx[fid2, :] = -fsx[fid2, :]
    ny[fid2, :] = -fsy[fid2, :]
    nz[fid2, :] = -fsz[fid2, :]

    # Face 3
    nx[fid3, :] = frx[fid3, :] + fsx[fid3, :] + ftx[fid3, :]
    ny[fid3, :] = fry[fid3, :] + fsy[fid3, :] + fty[fid3, :]
    nz[fid3, :] = frz[fid3, :] + fsz[fid3, :] + ftz[fid3, :]

    # Face 4
    nx[fid4, :] = -frx[fid4, :]
    ny[fid4, :] = -fry[fid4, :]
    nz[fid4, :] = -frz[fid4, :]

    # Normalização
    # O operador '.*' do MATLAB é simplesmente '*' no NumPy (multiplicação elemento a elemento).
    sJ = np.sqrt(nx**2 + ny**2 + nz**2)
    nx = nx / sJ
    ny = ny / sJ
    nz = nz / sJ

    sJ = sJ * J[fmask_flat, :]

    return nx, ny, nz, sJ

def tiConnect3D(EToV):
    """
    Algoritmo de conectividade de faces tetraédricas de Toby Isaac.
    Gera as matrizes Element-to-Element (EToE) e Element-to-Face (EToF).
    
    Entrada:
    EToV : Array (K, 4) contendo os índices (0-based) dos vértices de cada elemento.
    """
    Nfaces = 4
    K = EToV.shape[0]
    
    # Em Python (0-based), o número de nós é o valor máximo + 1
    Nnodes = np.max(EToV) + 1

    # Cria uma lista de todas as faces empilhadas
    # Tradução direta de [1,2,3], [1,2,4], [2,3,4], [1,3,4] do MATLAB para 0-based
    fnodes = np.vstack([
        EToV[:, [0, 1, 2]],
        EToV[:, [0, 1, 3]],
        EToV[:, [1, 2, 3]],
        EToV[:, [0, 2, 3]]
    ])

    # Ordena os nós de cada face. 
    # O MATLAB original subtraía 1 aqui. Como nosso EToV já é 0-based, ignoramos o -1.
    fnodes = np.sort(fnodes, axis=1)

    # Configura a conectividade padrão (o elemento aponta para si mesmo)
    # EToE: Matriz (K, 4) onde a linha 'i' é [i, i, i, i]
    EToE = np.repeat(np.arange(K)[:, None], Nfaces, axis=1)
    
    # EToF: Matriz (K, 4) onde cada linha é [0, 1, 2, 3]
    EToF = np.tile(np.arange(Nfaces), (K, 1))

    # Cria um ID (Hash) único para cada conjunto de três vértices
    face_id = fnodes[:, 0] * (Nnodes**2) + fnodes[:, 1] * Nnodes + fnodes[:, 2]

    # Achata as matrizes seguindo o padrão de colunas do MATLAB (Fortran-style 'F')
    EToE_flat = EToE.flatten('F')
    EToF_flat = EToF.flatten('F')
    linear_indices = np.arange(Nfaces * K) # Índices de 0 até (Nfaces*K - 1)

    # spNodeToNode = [Hash ID, Índice linear, ID do Elemento, ID da Face local]
    spNodeToNode = np.column_stack([face_id, linear_indices, EToE_flat, EToF_flat])

    # Ordena as linhas baseadas no número global da face (Hash na coluna 0)
    sorted_idx = np.argsort(spNodeToNode[:, 0])
    sorted_array = spNodeToNode[sorted_idx]

    # Encontra as correspondências (matches) na lista de faces ordenadas
    # Se o Hash da linha atual for igual ao Hash da próxima linha, é uma conexão!
    match_idx = np.where(sorted_array[:-1, 0] == sorted_array[1:, 0])[0]

    # Torna as conexões reflexivas (A aponta pra B, e B aponta pra A)
    matchL = np.vstack([sorted_array[match_idx, :], sorted_array[match_idx + 1, :]])
    matchR = np.vstack([sorted_array[match_idx + 1, :], sorted_array[match_idx, :]])

    # Insere as conexões usando os índices lineares
    # Coluna 1 = índices lineares; Coluna 2 = Elemento vizinho; Coluna 3 = Face vizinha
    target_indices = matchL[:, 1].astype(int)
    EToE_flat[target_indices] = matchR[:, 2].astype(int)
    EToF_flat[target_indices] = matchR[:, 3].astype(int)

    # Remonta as matrizes para o formato original (K x Nfaces), mantendo o padrão 'F'
    EToE = EToE_flat.reshape((K, Nfaces), order='F')
    EToF = EToF_flat.reshape((K, Nfaces), order='F')

    return EToE, EToF

def BuildMaps3D(K, Np, Nfp, Nfaces, Fmask, EToE, EToF, x, y, z, NODETOL=1e-10):
    """
    Constrói as tabelas de conectividade e contorno para os nós.
    
    Suposições:
    - K: número de elementos.
    - Np: nós por elemento.
    - Fmask, EToE, EToF já devem usar indexação 0-based.
    - x, y, z: Arrays (Np, K) com as coordenadas físicas.
    """
    # 1. O MATLAB acessa matrizes 2D usando um único índice (linear indexing). 
    # Para replicar isso perfeitamente, achatamos as coordenadas lendo por colunas ('F').
    x_flat = x.flatten('F')
    y_flat = y.flatten('F')
    z_flat = z.flatten('F')

    # 2. Numera os nós de volume sequencialmente (0 até K*Np - 1)
    # Equivalente ao: nodeids = reshape(1:K*Np, Np, K);
    nodeids = np.arange(K * Np).reshape((Np, K), order='F')

    vmapM = np.zeros((Nfp, Nfaces, K), dtype=int)
    vmapP = np.zeros((Nfp, Nfaces, K), dtype=int)

    # 3. Mapeia os índices globais dos nós pertencentes a cada face (vmapM)
    for k1 in range(K):
        for f1 in range(Nfaces):
            # Fmask aponta para o índice local do nó (0 a Np-1). nodeids pega o ID global.
            vmapM[:, f1, k1] = nodeids[Fmask[:, f1], k1]

    # 4. Encontra a conectividade entre os nós vizinhos (vmapP)
    for k1 in range(K):
        for f1 in range(Nfaces):
            # Elemento e face vizinhos
            k2 = EToE[k1, f1]
            f2 = EToF[k1, f1]

            vidM = vmapM[:, f1, k1]
            vidP = vmapM[:, f2, k2]

            # Pega as coordenadas físicas dos nós
            # Usamos [:, None] e [None, :] para criar um "produto externo" (broadcasting),
            # substituindo a necessidade do vetor 'tmp' do MATLAB.
            xM = x_flat[vidM][:, None]
            yM = y_flat[vidM][:, None]
            zM = z_flat[vidM][:, None]

            xP = x_flat[vidP][None, :]
            yP = y_flat[vidP][None, :]
            zP = z_flat[vidP][None, :]

            # Computa a matriz de distância Euclidiana ao quadrado
            D = (xM - xP)**2 + (yM - yP)**2 + (zM - zP)**2

            # Encontra os nós cujas coordenadas físicas batem (distância quase zero)
            idM, idP = np.where(np.abs(D) < NODETOL)

            # Salva o nó vizinho no mapa Plus (vmapP)
            vmapP[idM, f1, k1] = vmapM[idP, f2, k2]

    # 5. Achata as matrizes em vetores 1D (equivalente a vmapP(:) no MATLAB)
    vmapP_flat = vmapP.flatten('F')
    vmapM_flat = vmapM.flatten('F')

    # 6. Cria a lista de nós de fronteira (Boundary nodes)
    # Na fronteira física, o elemento não tem vizinho, então o algoritmo
    # de conectividade faz ele apontar para si mesmo (vmapP == vmapM).
    mapB = np.where(vmapP_flat == vmapM_flat)[0]
    vmapB = vmapM_flat[mapB]

    return vmapM_flat, vmapP_flat, vmapB, mapB