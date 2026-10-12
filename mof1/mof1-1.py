import numpy as np
import matplotlib.pyplot as plt

N = 100
V = 0
L = 200 / 0.05292
deltaX = L / N
m = 0.067
au_to_meV = 27211.4

E_net_meV = np.linspace(0.1, 35, 1000)
psi_N_dist = []

for E_meV in E_net_meV:
    E_au = E_meV / au_to_meV
    
    psi_table = np.zeros(N + 1)
    psi_table[0] = 0
    psi_table[1] = 1
    
    for i in range(1, N):
        psi_table[i+1] = -(2 * m) * (E_au - V) * deltaX**2 * psi_table[i] - psi_table[i-1] + 2 * psi_table[i]

    psi_N_dist.append(psi_table[N])

plt.plot(E_net_meV, psi_N_dist, color='red', linestyle='-')
plt.axhline(0, color='black', linestyle='--') 
plt.ylim(-25, 25) 
plt.xlabel("Energia E [meV]")
plt.ylabel("Wartość Psi(L)")
plt.grid(True)
plt.show()