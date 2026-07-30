// PARA GERAR A MALHA: gmsh cavidade3D.geo -3 -format msh22
// Tamanho característico do elemento (Refinamento)
lc = 0.2; 

// ==========================================
// 1. PONTOS
// ==========================================
// Base do cubo (z = -1)
Point(1) = {-1, -1, -1, lc}; // Canto inf esq
Point(2) = { 1, -1, -1, lc}; // Canto inf dir
Point(3) = { 1,  1, -1, lc}; // Canto sup dir
Point(4) = {-1,  1, -1, lc}; // Canto sup esq

// Topo do cubo (z = 1)
Point(5) = {-1, -1,  1, lc}; // Canto inf esq
Point(6) = { 1, -1,  1, lc}; // Canto inf dir
Point(7) = { 1,  1,  1, lc}; // Canto sup dir
Point(8) = {-1,  1,  1, lc}; // Canto sup esq

// ==========================================
// 2. LINHAS
// ==========================================
// Linhas da base
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Linhas do topo
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

// Linhas verticais (Pilares conectando base e topo)
Line(9)  = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

// ==========================================
// 3. CONTORNOS E SUPERFÍCIES (As 6 faces do cubo)
// ==========================================
// O sinal de "menos" inverte a direção da linha para fechar o ciclo
Curve Loop(1) = {1, 2, 3, 4};       // Face Base (z=0)
Plane Surface(1) = {1};

Curve Loop(2) = {5, 6, 7, 8};       // Face Topo (z=1)
Plane Surface(2) = {2};

Curve Loop(3) = {1, 10, -5, -9};    // Face Frente (y=0)
Plane Surface(3) = {3};

Curve Loop(4) = {2, 11, -6, -10};   // Face Direita (x=1)
Plane Surface(4) = {4};

Curve Loop(5) = {3, 12, -7, -11};   // Face Fundo (y=1)
Plane Surface(5) = {5};

Curve Loop(6) = {4, 9, -8, -12};    // Face Esquerda (x=0)
Plane Surface(6) = {6};

// ==========================================
// 4. VOLUME
// ==========================================
// Agrupa as 6 faces para formar a "casca" do cubo
Surface Loop(1) = {1, 2, 3, 4, 5, 6};

// Preenche o interior com tetraedros
Volume(1) = {1};

// ==========================================
// 5. GRUPOS FÍSICOS (Tags para o Python)
// ==========================================
// Separamos a tampa das outras paredes para as condições de contorno
Physical Surface("Tampa", 1) = {2}; 
Physical Surface("Paredes", 2) = {1, 3, 4, 5, 6};

// Essencial no 3D: Você precisa agrupar o volume para que o 
// Gmsh exporte os nós e tetraedros internos, e não apenas a casca.
Physical Volume("Dominio_3D", 3) = {1};