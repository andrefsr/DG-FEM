
### Fase U(0,2pi)
### Amplitude propto k ^ -beta
### Tomar a inversa

import numpy as np
import matplotlib.pyplot as plt

Nx = Ny = 256

kx = np.fft.fftfreq(Nx)
ky = np.fft.fftfreq(Ny)

KX, KY = np.meshgrid(kx, ky)
K = np.sqrt(KX**2 + KY**2)

k0 = 0.05

beta = 2

A = np.zeros_like(K)
A[K>0] = (K[K>0])**(-2*beta)

phi = np.random.uniform(0, 2*np.pi, size=(Ny,Nx))

F = np.sqrt(A)*np.exp(1j*phi)

field = np.fft.ifft2(F).real

plt.imshow(field,cmap='viridis')
plt.colorbar()
plt.show()