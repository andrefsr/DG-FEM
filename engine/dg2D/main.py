import numpy as np
import setup as stp
import Maxwell2D as Max
import operators2D as op2D
import mesh_reader as msh
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.tri as mtri

### Driver Script for solving the 2D vacuum Maxwell's equations on TM form

N = 6

VX, VY, EToV, BCTags = msh.MeshReader2D('engine/dg2D/cavidade quadrada.msh')

malha = stp.StartUp2D(N,EToV,VX,VY)

### Condições iniciais
Ez = np.sin(np.pi*malha.x)*np.sin(np.pi*malha.y)
#Ez = np.zeros((malha.Np,malha.K))
#Ez = np.exp(-(malha.x**2 + malha.y**2) / (0.1**2))
Hx = np.zeros((malha.Np,malha.K))
Hy = np.zeros((malha.Np,malha.K))

FinalTime = 0.5
#Hx, Hy, Ez = Max.Maxwell2D(Hx,Hy,Ez,FinalTime,malha)
Hx, Hy, Ez, pp, t = Max.Maxwell2D(Hx,Hy,Ez,FinalTime,malha)


###########################################################################################################################


print('N da malha', malha.N)
print('N real', N)

x_plot = malha.x.flatten(order='F')
y_plot = malha.y.flatten(order='F')
Ez_plot = Ez.flatten(order='F')

plt.figure(figsize=(8, 6))
plt.title(f'Campo Elétrico (Ez) em t = {FinalTime}')

grafico = plt.tricontourf(x_plot, y_plot, Ez_plot, levels=100, cmap='seismic')
plt.colorbar(grafico, label='Amplitude Ez')

plt.plot(VX[EToV].T, VY[EToV].T, color='k', linewidth=0.5, alpha=0.3)

plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.tight_layout()

plt.show()

node = True
if node == True:
    x_nos = malha.x.flatten(order='F')
    y_nos = malha.y.flatten(order='F')

    plt.figure(figsize=(8, 8))
    plt.title(f'Distribuição dos Nós no Nodal DG (Polinômio N = {N})')

    plt.triplot(VX, VY, EToV, color='gray', linewidth=0.8, alpha=0.5, label='Arestas dos Triângulos')

    plt.plot(x_nos, y_nos, 'o', markersize=3, color='red', label='Nós de Interpolação')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.legend(loc='upper right')

    plt.show()

ani = False
if ani == True:
    c0 = 299792458.0 # Velocidade da luz para o tempo físico

    # --- 2. CONFIGURAÇÃO DA FIGURA ESTÁTICA ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # A triangulação só precisa ser calculada uma vez!
    x_flat = malha.x.flatten(order='F')
    y_flat = malha.y.flatten(order='F')
    triangulacao = mtri.Triangulation(x_flat, y_flat)

    # --- 3. A FUNÇÃO DE ATUALIZAÇÃO (O "Motor" da Animação) ---
    def desenhar_quadro(frame):
        """
        Esta função é chamada automaticamente para cada quadro do vídeo.
        O argumento 'frame' é o índice atual da lista (0, 1, 2, ...)
        """
        ax.clear() # Limpa o desenho anterior
        
        # Pega o campo e o tempo exatos deste quadro
        Ez_atual = pp[frame].flatten(order='F')
        tempo_atual = t[frame]
        tempo_fisico_ns = (tempo_atual / c0) * 1e9
        
        # Desenha o novo contorno
        ax.tricontourf(triangulacao, Ez_atual, levels=50, cmap='seismic', vmin=-1.0, vmax=1.0)
        
        # Formatação do gráfico
        ax.set_title(f'Campo Ez - Tempo: {tempo_fisico_ns:.3f} ns')
        ax.set_aspect('equal')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Print para você acompanhar o progresso no terminal
        if frame % 10 == 0:
            print(f"Renderizando quadro {frame}/{len(pp)}...", end='\r')

    # --- 4. GERAÇÃO E SALVAMENTO DA ANIMAÇÃO ---
    print("Iniciando a renderização do vídeo...")

    # Pula de 20 em 20 índices: [0, 20, 40, 60...]
    indices_para_animar = np.arange(0, len(pp), 20) 

    # O FuncAnimation gerencia o loop automaticamente
    animacao = animation.FuncAnimation(
        fig, 
        desenhar_quadro, 
        frames=indices_para_animar, # Quantos quadros o vídeo terá
        interval=200,              # Milissegundos entre cada quadro (se for exibir na tela)
        blit=False                # Blit=False é mais seguro para tricontourf
    )

    # Salva como GIF usando o Pillow (já vem no Python)
    animacao.save('propagacao_pml.gif', writer='pillow', fps=10, dpi=100)

    print("\nRenderização concluída! Arquivo 'propagacao_pml.gif' salvo com sucesso.")
    plt.close(fig) # Limpa a memória

######## Solução teórica

Ez_analitico = np.sin(np.pi*malha.x)*np.sin(np.pi*malha.y)*np.cos(np.sqrt(2)*np.pi*FinalTime)

x_plot2 = malha.x.flatten(order='F')
y_plot2 = malha.y.flatten(order='F')
Ez_plot2 = Ez_analitico.flatten(order='F')

plt.figure(figsize=(8, 6))
plt.title(f'Campo Elétrico (Ez) analítico em t = {FinalTime}')

grafico = plt.tricontourf(x_plot2, y_plot2, Ez_plot2, levels=100, cmap='seismic')
plt.colorbar(grafico, label='Amplitude Ez')

plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.tight_layout()

plt.show()

# 2. Norma L-infinito (Erro Máximo Absoluto)
# Ótimo para ver se existe algum "pico" de erro escondido em algum triângulo
erro_Linf = np.max(np.abs(Ez - Ez_analitico))

# 3. Norma L2 Relativa (A mais usada em artigos científicos)
# Mostra o erro percentual global de energia na malha
erro_L2 = np.linalg.norm(Ez - Ez_analitico) / np.linalg.norm(Ez_analitico)

print(f"--- Análise de Erro (t = {FinalTime:.4f}) ---")
print(f"Norma L-infinito (Máx): {erro_Linf:.4e}")
print(f"Norma L2 Relativa:      {erro_L2:.4e}")