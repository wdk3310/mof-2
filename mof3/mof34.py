import numpy as np
import matplotlib.pyplot as plt

m = 0.067      
ab = 0.05292    
Eh = 27211.4      
dt = 1.0         
n_steps = 10000 

N = 201
x_nm = np.linspace(-100, 100, N)
x_au = x_nm / ab               
dx_au = x_au[1] - x_au[0]

F_Vnm = 0.75                    
F_au = F_Vnm * ab / Eh        
V = F_au * x_au                   

sigma_nm = 10
sigma_au = sigma_nm / ab
k_nm = 0.1
k_au = k_nm * ab

Psi0 = np.exp(-x_au**2 / (2 * sigma_au**2)) * np.exp(1j * k_au * x_au)
Psi0[0] = Psi0[-1] = 0.0          

C_norm = np.sqrt(np.sum(np.abs(Psi0)**2) * dx_au)
Psi0 = Psi0 / C_norm                 

def apply_H(Psi):
    H_Psi = np.zeros_like(Psi, dtype=complex)
    for i in range(1, N - 1):
        d2Psi = (Psi[i+1] - 2*Psi[i] + Psi[i-1]) / (dx_au**2)
        H_Psi[i] = -(1 / (2 * m)) * d2Psi + V[i] * Psi[i]
    return H_Psi

H_Psi0 = apply_H(Psi0)
E_avg = np.sum(np.conj(Psi0) * H_Psi0) * dx_au
E_avg_real = np.real(E_avg)     

Psi1 = Psi0 * np.exp(-1j * E_avg_real * dt)
Psi1[0] = Psi1[-1] = 0.0             

Psi_prev = np.copy(Psi0)
Psi_curr = np.copy(Psi1)

density_map = []
time_map = []

density_map.append(np.abs(Psi0)**2)
time_map.append(0.0)

for step in range(2, n_steps + 1):
    H_Psi = apply_H(Psi_curr)

    Psi_next = Psi_prev + (2 * dt / 1j) * H_Psi
    Psi_next[0] = Psi_next[-1] = 0.0  

    Psi_prev = np.copy(Psi_curr)
    Psi_curr = np.copy(Psi_next)

    if step % 100 == 0:
        density_map.append(np.abs(Psi_curr)**2)
        time_map.append(step * dt)

plt.figure(figsize=(10, 6))
plt.pcolormesh(x_nm, time_map, density_map, cmap='inferno', shading='auto')
plt.colorbar(label='Gestosc prawdopodobienstwa $|\Psi(x, t)|^2$')
plt.title(f'Ewolucja paczki falowej, F = {F_Vnm} meV/nm')
plt.xlabel('Polozenie x [nm]')
plt.ylabel('Czas t [j.at.]')
plt.tight_layout()
plt.show()

#=================  4  =============
t_values_au = np.zeros(n_steps + 1)
x_avg_quant_nm = np.zeros(n_steps + 1)
x_class_nm = np.zeros(n_steps + 1)

def calc_x_avg(Psi):
    return np.sum(x_au * np.abs(Psi)**2) * dx_au

t_values_au[0] = 0.0
x_avg_quant_nm[0] = calc_x_avg(Psi0) * ab
x_class_nm[0] = 0.0  

t_values_au[1] = dt
x_avg_quant_nm[1] = calc_x_avg(Psi1) * ab
x_class_nm[1] = (k_au/m * dt - F_au * dt**2 / (2*m)) * ab

Psi_prev = np.copy(Psi0)
Psi_curr = np.copy(Psi1)

for step in range(2, n_steps + 1):
    H_Psi = apply_H(Psi_curr)

    Psi_next = Psi_prev + (2 * dt / 1j) * H_Psi
    Psi_next[0] = Psi_next[-1] = 0.0
    
    current_t = step * dt
    t_values_au[step] = current_t
    
    x_avg_quant_nm[step] = calc_x_avg(Psi_next) * ab
    
    x_class_nm[step] = (k_au/m * current_t - F_au * current_t**2 / (2*m)) * ab
    
    Psi_prev = np.copy(Psi_curr)
    Psi_curr = np.copy(Psi_next)

plt.figure(figsize=(10, 6))
plt.plot(t_values_au, x_avg_quant_nm, label=r'Kwantowa wartosc: oczekiwana $\langle x \rangle(t)$', linewidth=2)
plt.plot(t_values_au, x_class_nm, label=r'Klasyczny tor $x(t)$', linestyle='--', linewidth=2, color='red')
plt.title(f'F = {F_Vnm} V/nm')
plt.xlabel('Czas [j.at.]')
plt.ylabel('Polozenie [nm]')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()

#=================  5  =============

F_Vnm = 0.0                        
F_au = F_Vnm * ab / Eh             
V = F_au * x_au   

H_Psi0 = apply_H(Psi0)
E_avg = np.sum(np.conj(Psi0) * H_Psi0) * dx_au
E_avg_real = np.real(E_avg)     

Psi1 = Psi0 * np.exp(-1j * E_avg_real * dt)
Psi1[0] = Psi1[-1] = 0.0             

Psi_prev = np.copy(Psi0)
Psi_curr = np.copy(Psi1)

density_map_0 = []
time_map_0 = []

density_map_0.append(np.abs(Psi0)**2)
time_map_0.append(0.0)

for step in range(2, n_steps + 1):
    H_Psi = apply_H(Psi_curr)

    Psi_next = Psi_prev + (2 * dt / 1j) * H_Psi
    Psi_next[0] = Psi_next[-1] = 0.0  

    Psi_prev = np.copy(Psi_curr)
    Psi_curr = np.copy(Psi_next)

    if step % 100 == 0:
        density_map_0.append(np.abs(Psi_curr)**2)
        time_map_0.append(step * dt)

plt.figure(figsize=(10, 6))
plt.pcolormesh(x_nm, time_map_0, density_map_0, cmap='inferno', shading='auto')
plt.colorbar(label='Gestosc prawdopodobienstwa $|\Psi(x, t)|^2$')
plt.title(f'Ewolucja paczki falowej, F = {F_Vnm} meV/nm')
plt.xlabel('Polozenie x [nm]')
plt.ylabel('Czas t [j.at.]')
plt.tight_layout()
plt.show()

t_values_au = np.zeros(n_steps + 1)
x_avg_quant_nm = np.zeros(n_steps + 1)
x_class_nm = np.zeros(n_steps + 1)

def calc_x_avg(Psi):
    norm_sq = np.sum(np.abs(Psi)**2) * dx_au
    return (np.sum(x_au * np.abs(Psi)**2) * dx_au) / norm_sq

t_values_au[0] = 0.0
x_avg_quant_nm[0] = calc_x_avg(Psi0) * ab
x_class_nm[0] = 0.0 

t_values_au[1] = dt
x_avg_quant_nm[1] = calc_x_avg(Psi1) * ab
x_class_nm[1] = (k_au/m * dt - F_au * dt**2 / (2*m)) * ab

Psi_prev = np.copy(Psi0)
Psi_curr = np.copy(Psi1)

for step in range(2, n_steps + 1):
    H_Psi = apply_H(Psi_curr)

    Psi_next = Psi_prev + (2 * dt / 1j) * H_Psi
    Psi_next[0] = Psi_next[-1] = 0.0

    current_t = step * dt
    t_values_au[step] = current_t

    x_avg_quant_nm[step] = calc_x_avg(Psi_next) * ab
 
    x_class_nm[step] = (k_au/m * current_t - F_au * current_t**2 / (2*m)) * ab
    
    Psi_prev = np.copy(Psi_curr)
    Psi_curr = np.copy(Psi_next)

plt.figure(figsize=(10, 6))
plt.plot(t_values_au, x_avg_quant_nm, label=r'Kwantowa wartosc: oczekiwana $\langle x \rangle(t)$', linewidth=2)
plt.plot(t_values_au, x_class_nm, label=r'Klasyczny tor $x(t)$', linestyle='--', linewidth=2, color='red')
plt.title(f'F = {F_Vnm} V/nm')
plt.xlabel('Czas [j.at.]')
plt.ylabel('Polozenie [nm]')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.show()