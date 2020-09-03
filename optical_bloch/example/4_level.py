import time
import numpy as np
from sympy import Symbol
import matplotlib.pyplot as plt
from optical_bloch.utils.general import flatten
from optical_bloch import Hamiltonian, Dissipator, BlochEquations

wa = Symbol(u'ω_A', real = True)
wb = Symbol(u'ω_B', real = True)
we1 = Symbol(u'ω_{E1}', real = True)
we2 = Symbol(u'ω_{E2}', real = True)

wc1 = Symbol(u'ω_{c1}', real = True)
Wc1 = Symbol(u'Ω_{c1}', real = True)
wp = Symbol(u'ω_p', real = True)
Wp = Symbol(u'Ω_p', real = True)
wc2 = Symbol(u'ω_{c2}', real = True)
Wc2 = Symbol(u'Ω_{c2}', real = True)

dp = Symbol(u'Δ_p', real = True)
dc1 = Symbol(u'Δ_{c1}', real = True)
dc2 = Symbol(u'Δ_{c2}', real = True)

ham = Hamiltonian(4)
ham.addEnergies([wa, wb, we1, we2])
ham.addCoupling(0,2,Wc1,wc1)
ham.addCoupling(1,2,Wp,wp)
ham.addCoupling(1,3,Wc2,wc2)
ham.eqnTransform()
ham.defineZero(we2)
ham.defineStateDetuning(0,2,dc1)
ham.defineStateDetuning(1,2,dp)
ham.defineStateDetuning(1,3,dc2)

ga = Symbol(u'Γ_A', real = True)
gb1 = Symbol(u'Γ_{B1}', real = True)
gb2 = Symbol(u'Γ_{B2}', real = True)
dis = Dissipator(4)
dis.addDecay(2,0,ga)
dis.addDecay(2,1,gb1)
dis.addDecay(3,1,gb2)

bloch = BlochEquations(4, dis.density_matrix, ham.transformed, dis.dissipator)

replacements = [(dc1,0),
                (dc2,0),
                (dp,0),
                (ga,1/2),
                (gb1,1/2),
                (gb2,1),
                (Wc1,2),
                (Wc2,2),
                (Wp,2)]


tstart = time.time()
sol = bloch.solveSteadyStateSymbolic(replacements)
print(f"\n{time.time() - tstart:.3f} s to solve 4 level steady-state optical bloch equations")
print("Steady-state solution :\n")
for key in flatten([bloch.density_matrix_steady_state[i,i:] for i in \
                    range(bloch.density_matrix_steady_state.rows)]):
    try:
        print(f"{str(key):<4} : {sol[key]:.3f}")
    except:
        print(f"{str(key):<4} : {str(sol[key])}")

replacements = [(dc2,0),
                (ga,1/2),
                (gb1,1/2),
                (gb2,1),
                (Wc1,2),
                (Wc2,2),
                (Wp,2)]
scan_ranges = [np.linspace(-10,10,751),np.linspace(-8,8,501)]
sol = bloch.solveSteadyStateNumeric(replacements, [dp,dc1], scan_ranges)

X,Y = np.meshgrid(*scan_ranges)
fig, ax = plt.subplots(figsize = (10,8))
cax = ax.pcolormesh(X,Y,sol[:,:,5].real.T, shading='auto')
cbar = fig.colorbar(cax)
cp = ax.contour(X,Y,sol[:,:,5].real.T, colors = 'k',
                linestyles = 'dashed')
ax.clabel(cp, inline = True, fontsize = 11, fmt = "%.2f")

ax.set_xlabel(f"${dp}$")
ax.set_ylabel(f"${dc1}$")

# Time evolution

replacements = [(dc1,0),
                (dc2,0),
                (dp,0),
                (ga,1/2),
                (gb1,1/2),
                (gb2,1),
                (Wc1,2),
                (Wc2,2),
                (Wp,2)]
#
y0 = np.zeros(bloch.levels**2, dtype = complex)
y0[0] = 1
#
tstart = time.time()
sol = bloch.solveNumeric(replacements, [0,20], y0, method = 'BDF')
print(f"\n{time.time() - tstart:.3f}s to solve 4 level optical bloch equations")

fig, ax = plt.subplots(figsize = (10,8))
ax.plot(sol.t, sol.y[0].real, label = r'$\rho_{00}$', lw = 2)
ax.plot(sol.t, sol.y[5].real, label = r'$\rho_{11}$', lw = 2)
ax.plot(sol.t, sol.y[10].real, label = r'$\rho_{22}$', lw = 2)
ax.plot(sol.t, sol.y[15].real, label = r'$\rho_{33}$', lw = 2)
ax.legend(fontsize = 13)
ax.set_xlabel('time')
ax.set_ylabel('population')

replacements = [(dc2,0),
                (ga,1/2),
                (gb1,1/2),
                (gb2,1),
                (Wc1,2),
                (Wc2,2),
                (Wp,2)]

sol = bloch.optimizeParametersNumeric(replacements, [0,20], y0, level = 1,
                                    parameters = (dp, dc1),
                                    bounds = [(-2.5,2.5), (-1,1)],
                                    method = 'BDF', optimize = 'minimum')
print(sol)
plt.show()
