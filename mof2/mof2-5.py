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

nu = 3
a_nm = 10.0
a = a_nm / a_b
x_axis = np.linspace(0, L, N + 1)         
x_axis_nm = np.linspace(0, L_nm, N + 1)  

V = -(hbar**2 / (2 * m * a**2)) * (nu * (nu + 1)) / (np.cosh((x_axis - L/2) / a))**2

tolerance = 1e-9 / au_to_meV

alpha = 0.9 * (m * dx**2) / (hbar**2)

def apply_Hamiltonian(Psi):
    H_Psi = np.zeros(N + 1)
    H_Psi[1:N] = -(hbar**2 / (2 * m)) * (Psi[2:N+1] + Psi[0:N-1] - 2 * Psi[1:N]) / (dx**2) + V[1:N] * Psi[1:N]
    return H_Psi

Psi_1 = np.random.uniform(-1, 1, N + 1)
Psi_1[0] = Psi_1[N] = 0.0
Psi_1 = Psi_1 / np.sqrt(np.sum(Psi_1**2) * dx)

E1_old = 0.0
iter1 = 0
iters_log_1, energies_log_1 = [], []

while True:
    iter1 += 1
    H_Psi1 = apply_Hamiltonian(Psi_1)
    
    # Czas urojony
    Psi_1[1:N] = Psi_1[1:N] - alpha * H_Psi1[1:N]
    # Normowanie
    Psi_1 = Psi_1 / np.sqrt(np.sum(Psi_1**2) * dx)
    
    E1_new = np.sum(Psi_1 * apply_Hamiltonian(Psi_1)) * dx
    
    if iter1 % 100 == 0 or iter1 == 1:
        iters_log_1.append(iter1)
        energies_log_1.append(E1_new * au_to_meV)
        
    if abs(E1_new - E1_old) < tolerance:
        iters_log_1.append(iter1)
        energies_log_1.append(E1_new * au_to_meV)
        break
    E1_old = E1_new

if Psi_1[N//2] < 0: Psi_1 = -Psi_1
E1_final = E1_new * au_to_meV

Psi_2 = np.random.uniform(-1, 1, N + 1)
Psi_2[0] = Psi_2[N] = 0.0
Psi_2 = Psi_2 / np.sqrt(np.sum(Psi_2**2) * dx)

E2_old = 0.0
iter2 = 0
iters_log_2, energies_log_2 = [], []

while True:
    iter2 += 1
    H_Psi2 = apply_Hamiltonian(Psi_2)
    
    Psi_2[1:N] = Psi_2[1:N] - alpha * H_Psi2[1:N]
    
    c1 = np.sum(Psi_1 * Psi_2) * dx
    Psi_2 = Psi_2 - c1 * Psi_1

    Psi_2 = Psi_2 / np.sqrt(np.sum(Psi_2**2) * dx)
    
    E2_new = np.sum(Psi_2 * apply_Hamiltonian(Psi_2)) * dx
    
    if iter2 % 100 == 0 or iter2 == 1:
        iters_log_2.append(iter2)
        energies_log_2.append(E2_new * au_to_meV)
        
    if abs(E2_new - E2_old) < tolerance:
        iters_log_2.append(iter2)
        energies_log_2.append(E2_new * au_to_meV)
        break
    E2_old = E2_new

if Psi_2[N//2 - int(10/dx)] < 0: Psi_2 = -Psi_2
E2_final = E2_new * au_to_meV

E_exact_0_meV = -(hbar**2 / (2 * m * a**2)) * (nu - 0)**2 * au_to_meV
E_exact_1_meV = -(hbar**2 / (2 * m * a**2)) * (nu - 1)**2 * au_to_meV


print(f"Stan podstawowy:")
print(f"  Numerycznie czas urojony: {E1_final:.6f} meV")
print(f"  Analitycznie czas urojony   {E_exact_0_meV:.6f} meV")
print(f"Stan 1 wzbudzony:")
print(f"  Numerycznie czas urojony: {E2_final:.6f} meV")
print(f"  Analitycznie czas urojony:   {E_exact_1_meV:.6f} meV")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

error_1 = np.abs(np.array(energies_log_1) - energies_log_1[-1]) + 1e-12
error_2 = np.abs(np.array(energies_log_2) - energies_log_2[-1]) + 1e-12

ax1.plot(iters_log_1, error_1, label='Stan podstawowy $\\Psi_1$', linewidth=2, color='blue')
ax1.plot(iters_log_2, error_2, label='Stan wzbudzony $\\Psi_2$', linewidth=2, color='red')
ax1.set_yscale('log')
ax1.set_title("Zbieznosc iteracji")
ax1.set_xlabel("Liczba iteracji")
ax1.set_ylabel("Blad bezwzgledny energii $\\Delta E$ [meV]")
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

V_scaled = V / np.max(np.abs(V)) * np.max(Psi_1) * 0.5 

ax2.plot(x_axis_nm, Psi_1, label="Stan podstawowy $\\Psi_1$")
ax2.plot(x_axis_nm, Psi_2, label="Stan wzbudzony $\\Psi_2$")
ax2.plot(x_axis_nm, V_scaled, label="Potencjał $V(x)$", color='black', linestyle=':')
ax2.set_title("Funkcje falowe potencjału Poschla-Tellera")
ax2.set_xlabel("Polozenie x [nm]")
ax2.set_ylabel("Amplituda")
ax2.legend()
ax2.grid(True, linestyle='--')

plt.tight_layout()
plt.show()