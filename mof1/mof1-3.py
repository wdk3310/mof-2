import numpy as np
import numba as nb
import matplotlib.pyplot as plt

L_nm = 200.0
a_b = 0.05292
L = L_nm / a_b
m = 0.067
hbar = 1.0
au_to_meV = 27211.4

N = 300
a_nm = 10.0
a = a_nm / a_b
dx = L / N

TOLERANCJA = 1e-12
x_axis = np.linspace(0, L, N + 1)
x_axis_nm = np.linspace(0, L_nm, N + 1)

@nb.njit
def get_Psi_N(E_meV, V):
    E_au = E_meV / au_to_meV
    Psi = np.zeros(N + 1)
    Psi[0] = 0.0
    Psi[1] = 1.0

    for i in range(1, N):
        Psi[i+1] = -(2 * m / hbar**2) * (E_au - V[i]) * dx**2 * Psi[i] - Psi[i-1] + 2 * Psi[i]
        
    return Psi[N]

def get_full_Psi(E_meV, V):
    E_au = E_meV / au_to_meV
    Psi = np.zeros(N + 1)
    Psi[0] = 0.0
    Psi[1] = 1.0

    for i in range(1, N):
        Psi[i+1] = -(2 * m / hbar**2) * (E_au - V[i]) * dx**2 * Psi[i] - Psi[i-1] + 2 * Psi[i]
        
    C = np.sum(Psi**2) * dx
    return Psi / np.sqrt(C)

def get_V(nu):
    return -(hbar**2 / (2 * m * a**2)) * (nu * (nu + 1)) / (np.cosh((x_axis - L/2) / a))**2

def znajdz_energie(V, max_stanow=7):
    energie_wlasne = []
    E_scan = np.linspace(-100, 40, 10000)
    przedzialy = []
    
    for i in range(len(E_scan) - 1):
        E_a = E_scan[i]
        E_b = E_scan[i+1]
        if get_Psi_N(E_a, V) * get_Psi_N(E_b, V) < 0:
            przedzialy.append((E_a, E_b))
            if len(przedzialy) == max_stanow:
                break
                
    for E_a, E_b in przedzialy:
        lewy = E_a
        prawy = E_b

        while (prawy - lewy) > TOLERANCJA:
            srodek = (lewy + prawy) / 2.0
            if get_Psi_N(lewy, V) * get_Psi_N(srodek, V) < 0:
                prawy = srodek
            else:
                lewy = srodek

        energie_wlasne.append((lewy + prawy) / 2.0)
        
    return energie_wlasne

nu_vals = np.linspace(0, 4, 40)
energie_all = []

print("Szukanie energii dla roznych wartosci nu...")
for nu in nu_vals:
    V = get_V(nu)
    energie_all.append(znajdz_energie(V, 7))

energie_all = np.array(energie_all).T

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for i in range(7):
    plt.plot(nu_vals, energie_all[i], label=f'Stan {i}')

plt.title("En(nu)")
plt.xlabel(r"$\nu$")
plt.ylabel("Energia [meV]")
plt.grid(True)
plt.ylim(-60, 10)

print("Obliczanie funkcji falowych dla nu = 3...")
nu = 3
V = get_V(nu)
energie_nu3 = znajdz_energie(V, 4)

plt.subplot(1, 2, 2)
V_scaled = V * au_to_meV
plt.plot(x_axis_nm, V_scaled, 'k:', label='Potencjal V(x)')

for i, E in enumerate(energie_nu3):
    Psi = get_full_Psi(E, V)
    if Psi[N//2 - 10] < 0 and i % 2 == 0: 
        Psi = -Psi
    plt.plot(x_axis_nm, Psi * 15 + E, label=f'Stan {i} (E={E:.2f} meV)')

plt.title("Funkcje falowe - nu = 3")
plt.xlabel("Polozenie [nm]")
plt.ylabel("Energia / Amplituda")
plt.axhline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.ylim(-80, 5)

plt.tight_layout()
plt.show()