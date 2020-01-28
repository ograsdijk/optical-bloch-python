import numpy as np
from .utils.math import commute
from sympy.matrices import zeros
from .utils.general import flatten
from sympy import Symbol, conjugate, simplify, Eq, solve

class BlochEquations:
    """
    Class for setting up the ODE system of optical Bloch equations.

    Methods:
    generateSteadyStateDensityMatrix()
    generateEquations()
    generateEquationsSteadyState()
    solveSteadyStateSymbolic(replacements)
    """

    def __init__(self, levels, density_matrix, hamiltonian, dissipator):
        """
        Initial parameters:
        levels          : number of levels of system
        density_matrix  : symbolic density matrix of system, 2D array or matrix
        hamiltonian     : symbolic hamiltonian of system, 2D array or matrix
        dissipator      : symbolic dissipator of system, 2D array or matrix
        """
        self.levels         = levels
        self.hamiltonian    = hamiltonian
        self.dissipator     = dissipator
        self.density_matrix = density_matrix

        # generate the ODE system of optical Bloch equations
        self.generateEquations()
        # generate the steady state density matrix
        self.generateSteadyStateDensityMatrix()
        # generate the stead state system of optical Bloch equations
        self.generateEquationsSteadyState()

    def generateSteadyStateDensityMatrix(self):
        """
        Generate the steady state density matrix; e.g. remove (t) from ρ₀₀(t),
        used for generating the steady state equations.

        Symbolic density matrix takes Hermitian properties into account.
        """
        density_matrix = zeros(*self.hamiltonian.shape)
        for i in range(self.hamiltonian.shape[0]):
            for j in range(i,self.hamiltonian.shape[0]):
                if i == j:
                    # \u03C1 is unicode for ρ, chr(0x2080+i) is unicode for
                    # subscript num(i), resulting in ρ₀₀ for example
                    density_matrix[i,j] = Symbol(u'\u03C1{0}{1}'.\
                                           format(chr(0x2080+i), chr(0x2080+j)))
                else:
                    density_matrix[i,j] = Symbol(u'\u03C1{0}{1}'.\
                                           format(chr(0x2080+i), chr(0x2080+j)))
                    density_matrix[j,i] = conjugate(Symbol(u'\u03C1{0}{1}'.\
                                          format(chr(0x2080+i), chr(0x2080+j))))
        self.density_matrix_steady_state = density_matrix

    def generateEquations(self):
        """
        Generate the system of ODEs for the optical Bloch equations.
        """
        self.equations = Eq(diff(self.density_matrix),
    simplify(-1j*commute(self.hamiltonian,self.density_matrix)+self.dissipator))

    def generateEquationsSteadyState(self):
        """
        Generate the steady state system of equations,
        e.g. dρ(t)/dt = 0 = -i[H,ρ]+L.
        """
        self.equations_steady_state = Eq(zeros(self.levels, self.levels),simplify(-1j*commute(self.hamiltonian,self.density_matrix_steady_state)+self.dissipator))
        for i in range(self.levels):
            for j in range(self.levels):
                self.equations_steady_state = self.equations_steady_state.\
                                               replace(self.density_matrix[i,j],
                                                       self.density_matrix_steady_state[i,j])
    def solveSteadyStateSymbolic(self, replacements):
        """
        Solve the steady state system of equations dρ(t)/dt = 0 = -i[H,ρ]+L.
        In principle can solve completely symbolically, but not guaranteed to
        give a solution in reasonable amount of time.
        With replacements numerical values can be substituted for symbolic
        variables in the equations to expedite solving.

        Parameters:
        replacements    : list of tuples, each tuple contains a symbolic
                          variable and the numeric replacement value for that
                          variable
        Returns:
        solution        : dictionary key, value pair where key is an element of
                          the density matrix and the value the solution for that
                          element
        """
        # taking the RHS of the steady state equations in order to add the
        # constraint Tr(ρ) = 1, needed to solve the system of equations
        eqns_rhs = self.equations_steady_state.rhs.subs(replacements)
        eqns_rhs = flatten(eqns_rhs.tolist())
        eqns_rhs.append(self.density_matrix_steady_state.trace()-1)
        # using the built in sympy solver, slow but can return symbolic results
        solution = solve(eqns_rhs, bloch.density_matrix_steady_state)
        return solution
