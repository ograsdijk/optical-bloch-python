"""
System in Λ configuration, states |A>, |B> with energies  𝜔𝐴 ,  𝜔𝐵  and a
common excited state |E> with energy  𝜔𝐸
"""

import time
import pprint
import numpy as np
from sympy import Symbol
import matplotlib.pyplot as plt
from optical_bloch.utils.general import flatten
from optical_bloch import Hamiltonian, Dissipator, BlochEquations

wa = Symbol(u'ω_A', real = True)
wb = Symbol(u'ω_E', real = True)
we = Symbol(u'ω_E', real = True)

wp = Symbol(u'ω_p', real = True)
Wp = Symbol(u'Ω_p', real = True)
wc = Symbol(u'ω_c', real = True)
Wc = Symbol(u'Ω_c', real = True)

dp = Symbol(u'Δ_p', real = True)
dc = Symbol(u'Δ_c', real = True)

ham = Hamiltonian(3)
ham.addEnergies([wa, wb, we])
ham.addCoupling(0,2,Wp,wp)
ham.addCoupling(1,2,Wc,wc)
ham.eqnTransform()
ham.defineZero(we)
ham.defineStateDetuning(0,2,dp)
ham.defineStateDetuning(1,2,dc)

ga = Symbol(u'Γ_A', real = True)
gb = Symbol(u'Γ_B', real = True)
dis = Dissipator(3)
dis.addDecay(2,0,ga)
dis.addDecay(2,1,gb)

pp = pprint.PrettyPrinter(indent=4)
print('\nHamiltonian : ')
pp.pprint(ham.hamiltonian)
print('\nTranformed Hamiltonian : ')
pp.pprint(ham.transformed)

print('\nTransform matrix : ')
pp.pprint(ham.T)

print('\nDensity matrix : ')
pp.pprint(dis.density_matrix)

print('\nBranching : ')
pp.pprint(dis.branching)

print('\nDissipator : ')
pp.pprint(dis.dissipator)

replacements = [(dp, 0),
                (Wc, 2),
                (Wp, 2),
                (dc, 0),
                (gb, ga),
                (ga, 1/2)]

bloch = BlochEquations(3, dis.density_matrix, ham.transformed, dis.dissipator)

tstart = time.time()
sol = bloch.solveSteadyStateSymbolic(replacements)
print(f"\n{time.time() - tstart:.3f}s to solve 3 level steady-state optical bloch equations")
print("Steady-state solution :\n")
for key in flatten([bloch.density_matrix_steady_state[i,i:] for i in \
                    range(bloch.density_matrix_steady_state.rows)]):
    try:
        print(f"{str(key):<15} : {sol[key]:.3f}")
    except:
        print(f"{str(key):<15} : {str(sol[key])}")

y0 = np.zeros(bloch.levels**2, dtype = complex)
y0[0] = 1

tstart = time.time()
sol = bloch.solveNumeric(replacements, [0,40], y0, method = 'BDF')
print(f"\n{time.time() - tstart:.3f}s to solve 3 level optical bloch equations")

fig, ax = plt.subplots(figsize = (10,8))
ax.plot(sol.t, sol.y[0].real, label = r'$\rho_{00}$', lw = 2)
ax.plot(sol.t, sol.y[4].real, label = r'$\rho_{11}$', lw = 2)
ax.plot(sol.t, sol.y[8].real, label = r'$\rho_{22}$', lw = 2)
ax.legend(fontsize = 13)
ax.set_xlabel('time')
ax.set_ylabel('population')
plt.show()
