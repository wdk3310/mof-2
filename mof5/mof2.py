import numpy as np
import matplotlib.pyplot as plt

N = 200
dr_array = np.linspace(0.01, 1.0, 200)

E_1_0 = []
E_2_0 = []
E_2_1 = []
E_3_0 = []
E_3_1 = []
E_3_2 = []

for dr in dr_array:
    i = np.arange(1, N + 1)
    r = i * dr
    
    for l in [0, 1, 2]:
        diag = 1.0 / dr**2 + 0.5 * l * (l + 1) / r**2 - 1.0 / r
        off_diag = np.full(N - 1, -1.0 / (2 * dr**2))
        
        H = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        eigvals = np.linalg.eigvalsh(H)
        
        if l == 0:
            E_1_0.append(eigvals[0])
            E_2_0.append(eigvals[1])
            E_3_0.append(eigvals[2])
        elif l == 1:
            E_2_1.append(eigvals[0])
            E_3_1.append(eigvals[1])
        elif l == 2:
            E_3_2.append(eigvals[0])

plt.figure(figsize=(8, 5))
plt.plot(dr_array, E_1_0, label='l=0')
plt.axhline(-1.0 / (2 * 1**2), color='black', linestyle='--', label='Analityczne')
plt.title('Energia E_{n=1}')
plt.xlabel(r'$\Delta r$ [j.at.]')
plt.ylabel('Energia [j.at.]')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(dr_array, E_2_0, label='l=0')
plt.plot(dr_array, E_2_1, label='l=1')
plt.axhline(-1.0 / (2 * 2**2), color='black', linestyle='--', label='Analityczne')
plt.title('Energia E_{n=2}')
plt.xlabel(r'$\Delta r$ [j.at.]')
plt.ylabel('Energia [j.at.]')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(dr_array, E_3_0, label='l=0')
plt.plot(dr_array, E_3_1, label='l=1')
plt.plot(dr_array, E_3_2, label='l=2')
plt.axhline(-1.0 / (2 * 3**2), color='black', linestyle='--', label='Analityczne')
plt.title('Energia E_{n=3}')
plt.xlabel(r'$\Delta r$ [j.at.]')
plt.ylabel('Energia [j.at.]')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()


#3
dr_zad3 = 0.1
l_zad3 = 0

i_zad3 = np.arange(1, N + 1)
r_zad3 = i_zad3 * dr_zad3

diag_zad3 = 1.0 / dr_zad3**2 + 0.5 * l_zad3 * (l_zad3 + 1) / r_zad3**2 - 1.0 / r_zad3
off_diag_zad3 = np.full(N - 1, -1.0 / (2 * dr_zad3**2))
H_zad3 = np.diag(diag_zad3) + np.diag(off_diag_zad3, k=1) + np.diag(off_diag_zad3, k=-1)

eigvals_zad3, eigvecs_zad3 = np.linalg.eigh(H_zad3)

plt.figure(figsize=(10, 6))
plt.plot(r_zad3, eigvecs_zad3[:, 0], label=f'Stan podstawowy n=1 (E={eigvals_zad3[0]:.4f})')
plt.plot(r_zad3, eigvecs_zad3[:, 1], label=f'I stan wzbudzony n=2 (E={eigvals_zad3[1]:.4f})')
plt.plot(r_zad3, eigvecs_zad3[:, 2], label=f'II stan wzbudzony n=3 (E={eigvals_zad3[2]:.4f})')

plt.title(r'Pomocnicze funkcje falowe $\phi_{n,l=0}(r)$ dla $\Delta r = 0.1$')
plt.xlabel('Promień r [j.at.]')
plt.ylabel(r'Amplituda $\phi(r)$')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()

#=====4

plt.figure(figsize=(10, 6))
plt.plot(r_zad3, eigvecs_zad3[:, 0] / r_zad3, label=f'Stan podstawowy n=1 (E={eigvals_zad3[0]:.4f})')
plt.plot(r_zad3, eigvecs_zad3[:, 1] / r_zad3, label=f'I stan wzbudzony n=2 (E={eigvals_zad3[1]:.4f})')
plt.plot(r_zad3, eigvecs_zad3[:, 2] / r_zad3, label=f'II stan wzbudzony n=3 (E={eigvals_zad3[2]:.4f})')
plt.title(r'Docelowe funkcje falowe $\psi_{n,l=0}(r) = \phi(r)/r$ dla $\Delta r = 0.1$')
plt.xlabel('Promień r [j.at.]')
plt.ylabel(r'Amplituda $\psi(r)$')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()