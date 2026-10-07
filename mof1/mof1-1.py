import numpy as np
import matplotlib.pyplot as plt

N = 100
V = 0
L = 200 / 0.05292
deltaX = L/N
m = 0.067 
E_max = 35 *10**-3
#planck = 1
#x_net = np.arrange(0, N + deltaX, deltaX)
E_net = np.linspace(0, E_max, N)
V = 0


#psi_i+1
def psi(i_in, Ei):
    i = i_in - 1 
    if i == 0:
        return 0 
    if i == 1:
        return 1
    
    return -(2*m) * (Ei * V) * deltaX**2 * psi_table[i] - psi_table[i-2] + 2*psi_table[i]



psi_N_dist = []
for Ei in E_net:
    psi_table = [0, 1]
    for i in range(2, N+1):
        psi_table.append(psi(i, Ei))

    psi_N_dist.append(psi(101, Ei))
    
plt.plot(E_net, psi_N_dist)
plt.show()

'''
def C():
    psi_sum = 0
    for i in range(N+1):
        psi_sum += psi()
'''