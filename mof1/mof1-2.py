import numpy as np
import numba as nb

L_nm = 200.0            
a_b = 0.05292    
L = L_nm / a_b        
m = 0.067           
hbar = 1.0                
au_to_meV = 27211.4        

TOLERANCJA = 0.000001  

@nb.njit
def get_Psi_N(E_meV, N):
    E_au = E_meV / au_to_meV
    dx = L / N

    Psi = np.zeros(N + 1)
    Psi[0] = 0.0
    Psi[1] = 1.0

    for i in range(1, N):
        Psi[i+1] = -(2 * m / hbar**2) * E_au * dx**2 * Psi[i] - Psi[i-1] + 2 * Psi[i]
        
    return Psi[N]

def znajdz_energie(N, max_stanow=5):
    energie_wlasne = []

    E_scan = np.linspace(0.01, 35, 2000)
    przedzialy = []
    
    for i in range(len(E_scan) - 1):
        E_a = E_scan[i]
        E_b = E_scan[i+1]
        if get_Psi_N(E_a, N) * get_Psi_N(E_b, N) < 0:
            przedzialy.append((E_a, E_b))
            if len(przedzialy) == max_stanow:
                break
                
    for E_a, E_b in przedzialy:
        lewy = E_a
        prawy = E_b

        while (prawy - lewy) > TOLERANCJA:
            srodek = (lewy + prawy) / 2.0

            if get_Psi_N(lewy, N) * get_Psi_N(srodek, N) < 0:
                prawy = srodek
            else:
                lewy = srodek

        energie_wlasne.append((lewy + prawy) / 2.0)
        
    return energie_wlasne

energie_100 = znajdz_energie(100)
energie_300 = znajdz_energie(300)

for i in range(1, len(energie_100) + 1):
    E_dokladne_au = (i**2 * hbar**2 * np.pi**2) / (2 * m * L**2)
    E_dokladne_meV = E_dokladne_au * au_to_meV
    
    E_100 = energie_100[i-1]
    E_300 = energie_300[i-1]
    
    print(f"Stan: {i:<2} | E: {E_dokladne_meV:<10.6f} [meV] | E N=100 {E_100:<10.6f} [meV] | E N=300 {E_300:<10.6f} [meV]")

print("\n \nBledy bezwzgledne:")
for i in range(1, len(energie_100) + 1):
    E_dokladne_au = (i**2 * hbar**2 * np.pi**2) / (2 * m * L**2)
    E_dokladne_meV = E_dokladne_au * au_to_meV
    
    blad_100 = abs(energie_100[i-1] - E_dokladne_meV)
    blad_300 = abs(energie_300[i-1] - E_dokladne_meV)
    
    print(f"Stan {i}: Błąd dla N=100: {blad_100:.6f} meV | Błąd dla N=300: {blad_300:.6f} meV")