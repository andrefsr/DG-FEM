import meshio

def MeshReader3D(nome_arquivo):

    # 1. Lê o arquivo da malha (lembre-se de usar a versão msh22 no Gmsh!)
    malha = meshio.read(nome_arquivo)

    # ==========================================
    # 2. Extraindo VX, VY, VZ
    # ==========================================
    # mesh.points retorna uma matriz (N_nos, 3) com as coordenadas
    VX = malha.points[:, 0]
    VY = malha.points[:, 1]
    VZ = malha.points[:, 2]

    # ==========================================
    # 3. Extraindo EToV (Element To Vertex)
    # ==========================================
    # O Gmsh exporta vários tipos de células (pontos, linhas, triângulos, tetraedros).
    # Para o volume, queremos apenas os tetraedros.
    EToV = malha.cells_dict["tetra"]

    # ==========================================
    # 4. Extraindo as Informações de Contorno (BCTags brutos)
    # ==========================================
    # No 3D, as fronteiras físicas são formadas por triângulos (as faces do cubo).
    # Extraímos a conectividade desses triângulos de borda:
    faces_contorno = malha.cells_dict["triangle"]

    # E extraímos os "Physical Groups" associados a cada um desses triângulos
    # (ex: 1 para "Tampa", 2 para "Paredes", conforme definido no .geo)
    tags_fisicas = malha.cell_data_dict["gmsh:physical"]["triangle"]

    print(f"Número de vértices (VX): {len(VX)}")
    print(f"Número de tetraedros (EToV): {EToV.shape[0]}")
    print(f"Número de faces de contorno: {faces_contorno.shape[0]}")

    return VX, VY, VZ, EToV #, BCTags