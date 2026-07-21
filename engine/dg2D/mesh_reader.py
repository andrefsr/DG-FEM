import numpy as np

def MeshReader2D(nome_arquivo):
    """
    Lê um arquivo .msh do Gmsh (Version 2 ASCII) e extrai 
    as coordenadas dos vértices (VX, VY) e a conectividade (EToV).
    """
    with open(nome_arquivo, 'r') as f:
        linhas = f.readlines()
    VX_list = []
    VY_list = []
    EToV_list = []
    BCTags_list = [] # Vai guardar as linhas das bordas
    
    i = 0
    while i < len(linhas):
        linha = linhas[i].strip()
        
        # ----------------------------------------------------
        # BLOCO 1: Lendo as Coordenadas dos Nós (VX e VY)
        # ----------------------------------------------------
        if linha == '$Nodes':
            i += 1
            num_nodes = int(linhas[i].strip())
            
            # Prepara as listas de coordenadas
            VX = np.zeros(num_nodes)
            VY = np.zeros(num_nodes)
            
            for _ in range(num_nodes):
                i += 1
                dados = linhas[i].split()
                node_id = int(dados[0]) - 1 # Subtrai 1 para o Python!
                
                VX[node_id] = float(dados[1])
                VY[node_id] = float(dados[2])
                
        # ----------------------------------------------------
        # BLOCO 2: Lendo a Conectividade (EToV)
        # ----------------------------------------------------
        elif linha == '$Elements':
            i += 1
            num_elements = int(linhas[i].strip())
            
            for _ in range(num_elements):
                i += 1
                dados = linhas[i].split()
                
                tipo_elemento = int(dados[1])
                num_tags = int(dados[2])
                
                # O Gmsh lista as tags antes dos nós. 
                # Geralmente a primeira tag é o ID do Grupo Físico que criamos no .geo
                tag_fisica = int(dados[3]) 
                
                # Pega os números dos nós pulando as tags iniciais (subtraindo 1 para o Python)
                nos = [int(n) - 1 for n in dados[3 + num_tags:]]
                
                if tipo_elemento == 2:
                    # Tipo 2 é um Triângulo! Salva na EToV
                    EToV_list.append(nos)
                    
                elif tipo_elemento == 1:
                    # Tipo 1 é uma Linha (Borda). Guardamos para montar o BCType depois
                    BCTags_list.append({"nos": nos, "tag": tag_fisica})
                    
        i += 1
        
    EToV = np.array(EToV_list)
    
    # K é o número total de triângulos
    K = EToV.shape[0]
    
    # Extrai as coordenadas x e y dos 3 vértices de cada triângulo
    x1, y1 = VX[EToV[:, 0]], VY[EToV[:, 0]]
    x2, y2 = VX[EToV[:, 1]], VY[EToV[:, 1]]
    x3, y3 = VX[EToV[:, 2]], VY[EToV[:, 2]]

    # Calcula o Determinante (Área geométrica direcional)
    J_geom = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

    # Encontra os índices dos triângulos que estão "do avesso" (J < 0)
    triangulos_invertidos = np.where(J_geom < 0)[0]

    if len(triangulos_invertidos) > 0:
        # Para inverter o sentido, basta trocar os nós 2 e 3 de lugar
        temp = EToV[triangulos_invertidos, 1].copy()
        EToV[triangulos_invertidos, 1] = EToV[triangulos_invertidos, 2]
        EToV[triangulos_invertidos, 2] = temp

    print(f"Malha carregada com sucesso!")
    print(f"Número de vértices: {len(VX)}")
    print(f"Número de Elementos (K): {K}")
    
    VX = np.array(VX, dtype=np.float64)
    VY = np.array(VY, dtype=np.float64)

    return VX, VY, EToV, BCTags_list

import numpy as np
import meshio

def MeshReader2D_v4(nome_arquivo):
    """
    Lê malhas do Gmsh (formato 4.1 ou superior) usando a biblioteca meshio.
    Retorna matrizes perfeitamente limpas para solvers DGTD.
    """
    print(f"Lendo malha 4.1 com meshio: {nome_arquivo}")
    malha = meshio.read(nome_arquivo)

    # 1. Extrai as coordenadas X e Y de todos os vértices
    # O meshio retorna um array 3D (X, Y, Z), pegamos só as duas primeiras colunas
    VX = malha.points[:, 0]
    VY = malha.points[:, 1]

    # 2. Extrai a matriz de conectividade dos triângulos (EToV)
    if "triangle" not in malha.cells_dict:
        raise ValueError("Nenhum triângulo encontrado! Verifique se a superfície foi gerada no Gmsh.")
    
    # O meshio já retorna a matriz EToV indexada em zero (padrão Python)
    EToV = malha.cells_dict["triangle"]

    # 3. Extrai as tags de contorno (BCTags) para as paredes
    # BCTags será uma matriz onde cada linha é: [No_1, No_2, Tag_Fisica]
    BCTags = []
    if "line" in malha.cells_dict:
        linhas = malha.cells_dict["line"] # Nós que formam as linhas
        
        # Tenta pegar as tags físicas definidas no Gmsh (ex: 1 para PEC)
        try:
            tags = malha.cell_data_dict["gmsh:physical"]["line"]
            # Une os nós da aresta com o número da tag física
            BCTags = np.column_stack((linhas, tags))
        except KeyError:
            print("Aviso: Linhas encontradas, mas sem 'Physical Groups' associados.")
            BCTags = linhas

    # ----------------------------------------------------------------------
    # A ARMADILHA DA BASE 1 vs BASE 0 (IMPORTANTE)
    # ----------------------------------------------------------------------
    # O formato MSH antigo e o código original do Hesthaven (MATLAB) começam 
    # a contar os nós a partir do número 1. O meshio e o Python contam do 0.
    # Se o seu arquivo StartUp2D.py usa a lógica de subtrair 1 internamente 
    # (ex: EToV = EToV - 1), o array lido pelo meshio vai ficar negativo e quebrar.
    #
    # Se o seu StartUp2D EXIGE que os índices comecem em 1, descomente as 2 linhas abaixo:
    #
    # EToV = EToV + 1
    # if len(BCTags) > 0: BCTags[:, 0:2] = BCTags[:, 0:2] + 1
    # ----------------------------------------------------------------------

    return VX, VY, EToV, BCTags
