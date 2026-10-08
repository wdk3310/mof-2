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

alpha_crit = (m * dx**2) / (hbar**2)

def apply_Hamiltonian(Psi, V, dx, m, hbar, N):
    H_Psi = np.zeros(N + 1)
    H_Psi[1:N] = -(hbar**2 / (2 * m)) * (Psi[2:N+1] + Psi[0:N-1] - 2 * Psi[1:N]) / (dx**2) \
                 + V[1:N] * Psi[1:N]
    return H_Psi

def test_convergence(alpha_multiplier, max_iter=40000):
    alpha = alpha_crit * alpha_multiplier
    
    Psi = np.random.uniform(-1, 1, N + 1)
    Psi[0] = 0.0
    Psi[N] = 0.0
    Psi = Psi / np.sqrt(np.sum(Psi**2) * dx)
    
    E_old = 0.0
    iterations_log = []
    energies_log = []
    
    for iteration in range(1, max_iter + 1):
        H_Psi = apply_Hamiltonian(Psi, V, dx, m, hbar, N)
        
        Psi_prime = np.zeros(N + 1)
        Psi_prime[1:N] = Psi[1:N] - alpha * H_Psi[1:N]
        Psi[1:N] = Psi_prime[1:N]
        
        Psi = Psi / np.sqrt(np.sum(Psi**2) * dx)
        
        H_Psi_new = apply_Hamiltonian(Psi, V, dx, m, hbar, N)
        E_new = np.sum(Psi * H_Psi_new) * dx
    
        if iteration % 100 == 0 or iteration == 1:
            iterations_log.append(iteration)
            energies_log.append(E_new * au_to_meV)
            
        if np.isnan(E_new) or abs(E_new * au_to_meV) > 10000:
            return iterations_log, energies_log, iteration, False
            
        # Sprawdzenie warunku zbieżności
        if abs(E_new - E_old) < tolerance:
            iterations_log.append(iteration)
            energies_log.append(E_new * au_to_meV)
            return iterations_log, energies_log, iteration, True
            
        E_old = E_new
        
    return iterations_log, energies_log, max_iter, False

multipliers = [0.5, 0.9, 0.99, 1.0, 1.02]

plt.figure(figsize=(10, 6))

for mult in multipliers:
    iters_log, engs_log, total_iters, converged = test_convergence(mult)
    
    status = "Zbieznosc" if converged else "Rozbieznosc"
    print(f"  -> {mult} * alpha: {status}. Ilosc iteracji: {total_iters}.")
    
    label_text = f"$\\alpha = {mult}\\alpha_{{kryt}}$"
    plt.plot(iters_log, engs_log, label=label_text)

plt.title("Zbieznosc oczekiwanej energii")
plt.xlabel("Liczba iteracji")
plt.ylabel("Oczekiwana energia E [meV]")
plt.ylim(0, 10) 
plt.xlim(0, 20000)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()