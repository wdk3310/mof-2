import numpy as np
import matplotlib.pyplot as plt

N = 300
L_nm = 200.0
a_b = 0.05292
L = L_nm / a_b
dx = L / N
m = 0.067
hbar = 1.0
au_to_meV = 27211.4
V = np.zeros(N + 1) 
tolerance = 1e-6 / au_to_meV
alpha = 0.99 * (m * dx**2) / (hbar**2)
x_axis = np.linspace(0, L_nm, N + 1) 

def apply_Hamiltonian(Psi):
    H_Psi = np.zeros(N + 1)
    H_Psi[1:N] = -(hbar**2 / (2 * m)) * (Psi[2:N+1] + Psi[0:N-1] - 2 * Psi[1:N]) / (dx**2) + V[1:N] * Psi[1:N]
    return H_Psi

Psi_1 = np.random.uniform(-1, 1, N + 1)
Psi_1[0] = Psi_1[N] = 0.0
Psi_1 = Psi_1 / np.sqrt(np.sum(Psi_1**2) * dx)

E1_old = 0.0
while True:
    H_Psi1 = apply_Hamiltonian(Psi_1)
    Psi_1[1:N] = Psi_1[1:N] - alpha * H_Psi1[1:N]
    Psi_1 = Psi_1 / np.sqrt(np.sum(Psi_1**2) * dx)
    E1_new = np.sum(Psi_1 * apply_Hamiltonian(Psi_1)) * dx
    if abs(E1_new - E1_old) < tolerance:
        break
    E1_old = E1_new

if Psi_1[N//2] < 0:
    Psi_1 = -Psi_1

print(f"Energia E1: {E1_new * au_to_meV:.6f} meV")

Psi_2 = np.random.uniform(-1, 1, N + 1)
Psi_2[0] = Psi_2[N] = 0.0
Psi_2 = Psi_2 / np.sqrt(np.sum(Psi_2**2) * dx)

E2_old = 0.0
iteration = 0
energies_log = []
iterations_log = []

while True:
    iteration += 1

    H_Psi2 = apply_Hamiltonian(Psi_2)
    Psi_2[1:N] = Psi_2[1:N] - alpha * H_Psi2[1:N]
  
    c1 = np.sum(Psi_1 * Psi_2) * dx

    Psi_2 = Psi_2 - c1 * Psi_1
    
    Psi_2 = Psi_2 / np.sqrt(np.sum(Psi_2**2) * dx)
 
    E2_new = np.sum(Psi_2 * apply_Hamiltonian(Psi_2)) * dx
 
    if iteration % 100 == 0 or iteration == 1:
        iterations_log.append(iteration)
        energies_log.append(E2_new * au_to_meV)
        
    if abs(E2_new - E2_old) < tolerance:
        iterations_log.append(iteration)
        energies_log.append(E2_new * au_to_meV)
        break
        
    E2_old = E2_new

print(f"Energia E2: {E2_new * au_to_meV:.6f} meV, liczba iteracji: {iteration}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

error = np.abs(np.array(energies_log) - energies_log[-1]) + 1e-10 
ax1.plot(iterations_log, error, color='red', linewidth=2)
ax1.set_yscale('log')
ax1.set_title("Zbieznosc energii ")
ax1.set_xlabel("Liczba iteracji")
ax1.set_ylabel("Blad energii $\\Delta E$ [meV]")
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.plot(x_axis, Psi_1, label="Stan podstawowy $\\Psi_1$", linewidth=2)
ax2.plot(x_axis, Psi_2, label="Stan wzbudzony $\\Psi_2$", linewidth=2)
ax2.set_title("Funkcje falowe stanu 1 i 2")
ax2.set_xlabel("Poozenie x [nm]")
ax2.set_ylabel("Amplituda $\\Psi(x)$")
ax2.legend()
ax2.grid(True, linestyle='--')

plt.tight_layout()
plt.show()