// 1. Variáveis
// O Gmsh tem a constante 'Pi' nativa embutida nele!
L = Pi;          
N_pontos = 81; // Mude aqui para refinar a malha no futuro

// 2. Pontos do Quadrado [-Pi, Pi]
Point(1) = {-L, -L, 0, 1.0};
Point(2) = { L, -L, 0, 1.0};
Point(3) = { L,  L, 0, 1.0};
Point(4) = {-L,  L, 0, 1.0};

// 3. Linhas (Paredes)
Line(1) = {1, 2}; // Parede Inferior
Line(2) = {2, 3}; // Parede Direita
Line(3) = {3, 4}; // Parede Superior
Line(4) = {4, 1}; // Parede Esquerda

// 4. Contorno e Superfície (O Vácuo interno)
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// 5. Estrutura da Malha (Força a criação de triângulos regulares)
Transfinite Curve {1, 2, 3, 4} = N_pontos;
Transfinite Surface {1};

// 6. GRUPOS FÍSICOS (Sem isso, o arquivo sai vazio!)
// O '1' e o '2' são as tags numéricas que o seu mesh_reader lê.
Physical Curve("Fronteira_PEC", 1) = {1, 2, 3, 4};
Physical Surface("Dominio_Vacuo", 2) = {1};