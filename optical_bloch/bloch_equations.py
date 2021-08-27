import itertools
import numpy as np
from tqdm import tqdm
from .utils.math import commute
from sympy.matrices import zeros
from .utils.general import flatten
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from sympy import Symbol, conjugate, simplify, Eq, solve, diff, \
                  linear_eq_to_matrix, lambdify, Function

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

    def generateSystem(self, replacements, full_output = False):
        eqns_rhs = self.equations.rhs.subs(replacements)

        # converting the symbolic functions ρ(t) to ρ in order to create the
        # matrix representing the linear equations (Ax=b)
        t = Symbol('t', real = True)
        for i in range(self.levels):
            for j in range(i,self.levels):
                tmp = Function(u'\u03C1{0}{1}'.format(chr(0x2080+i), 
                                                            chr(0x2080+j)))
                tmp1 = Symbol(u'\u03C1{0}{1}'.format(chr(0x2080+j), 
                                                            chr(0x2080+i)))
                eqns_rhs = eqns_rhs.subs(conjugate(tmp(t)), tmp1)

        for i in range(self.levels):
            for j in range(i,self.levels):
                tmp = Function(u'\u03C1{0}{1}'.format(chr(0x2080+i), 
                                                            chr(0x2080+j)))
                tmp1 = Symbol(u'\u03C1{0}{1}'.format(chr(0x2080+i), 
                                                            chr(0x2080+j)))
                eqns_rhs = eqns_rhs.subs(tmp(t), tmp1)

        syms = []
        for i in range(self.levels):
            for j in range(self.levels):
                syms.append(Symbol(u'\u03C1{0}{1}'.format(chr(0x2080+i), 
                                                                chr(0x2080+j))))

        # creating the matrix A (from Ax = b) for the ODE system
        matrix_eq = linear_eq_to_matrix(eqns_rhs, syms)[0]

        # check if there is still time dependence inside the matrix, return
        # a lambdified function if yes, else return a numpy array
        if t in matrix_eq.free_symbols:
            t_dependent = True
            matrix_eq = lambdify(t, matrix_eq, 'numpy')
        else:
            t_dependent = False
            matrix_eq = np.array(matrix_eq).astype(complex)

        if full_output:
            return matrix_eq, t_dependent, syms
        else:
            return matrix_eq, t_dependent

    def generateEquationsSteadyState(self):
        """
        Generate the steady state system of equations,
        e.g. dρ(t)/dt = 0 = -i[H,ρ]+L.
        """
        self.equations_steady_state = Eq(zeros(self.levels, self.levels), \
            simplify(-1j*commute(   self.hamiltonian,
                                    self.density_matrix_steady_state
                                ) + \
                    self.dissipator))
        for i in range(self.levels):
            for j in range(self.levels):
                self.equations_steady_state = self.equations_steady_state.\
                                replace(self.density_matrix[i,j], 
                                        self.density_matrix_steady_state[i,j])

    def generateSystemSteadyState(self, replacements, full_output = False):
        # taking the RHS of the steady state equations in order to add the
        # constraint Tr(ρ) = 1, needed to solve the system of equations
        eqns_rhs = self.equations_steady_state.rhs.subs(replacements)
        eqns_rhs = flatten(eqns_rhs.tolist())
        eqns_rhs[0] += self.density_matrix_steady_state.trace()-1

        for i in range(self.levels):
            for j in range(i,self.levels):
                tmp = Symbol(u'\u03C1{0}{1}'.format(chr(0x2080+i), 
                                                        chr(0x2080+j)))
                tmp1 = Symbol(u'\u03C1{0}{1}'.format(chr(0x2080+j), 
                                                        chr(0x2080+i)))
                for idx in range(len(eqns_rhs)):
                    eqns_rhs[idx] = eqns_rhs[idx].subs(conjugate(tmp), tmp1)
        syms = []
        for i in range(self.levels):
            for j in range(self.levels):
                syms.append(Symbol(u'\u03C1{0}{1}'.format(chr(0x2080+i), 
                                                            chr(0x2080+j))))
        matrix_eq = linear_eq_to_matrix(eqns_rhs, syms)       
        if full_output:
            return matrix_eq, syms
        else:
            return matrix_eq

    def solveSteadyStateSymbolic(self, replacements = []):
        """
        Solve the steady state system of equations dρ(t)/dt = 0 = -i[H,ρ]+L.
        In principle can solve completely symbolically, but not guaranteed to
        give a solution in reasonable amount of time.
        With replacements numerical values can be substituted for symbolic
        variables in the equations to expedite solving.

        Parameters:
        replacements    :   list of tuples, each tuple contains a symbolic
                            variable and the numeric replacement value for that
                            variable

        Returns:
        solution        :   dictionary key, value pair where key is an element 
                            of the density matrix and the value the solution for 
                            that element
        """
        # taking the RHS of the steady state equations in order to add the
        # constraint Tr(ρ) = 1, needed to solve the system of equations
        eqns_rhs = self.equations_steady_state.rhs.subs(replacements)
        eqns_rhs = flatten(eqns_rhs.tolist())
        eqns_rhs.append(self.density_matrix_steady_state.trace()-1)
        # using the built in sympy solver, slow but can return symbolic results
        return solve(eqns_rhs, self.density_matrix_steady_state)

    def solveSteadyStateNumeric(self, replacements, parameters_scan = None,
                                scan_ranges = None):
        """
        Solve the steady state system of equations dρ(t)/dt = 0 = -i[H,ρ]+L
        numerically. Allows for scanning multiple parameters returning an array
        of [i,j,...,n] where i,j,.. are the lengths of the scan_ranges and n is
        the number of parameters to solve for

        Parameters:
        replacements    :   list of tuples, each tuple contains a symbolic
                            variable and the numeric replacement value for that
                            variable
        parameters_scan :   list of parameters to scan during the solve
        scan_ranges     :   list with the scan ranges in the same order as
                            parameters_scan

        Returns:
        solution        :   [i,j,...,n] where i,j,.. are the lengths of the
                            scan_ranges and n is the number of parameters solved
                            for
        """
        matrix_eq, t_dependent, syms = self.generateSystemSteadyState(replacements, True)
        if t_dependent:
            raise AssertionError('No steady state solution for time dependent variables')

        if parameters_scan:
            y = np.zeros([*[len(r) for r in scan_ranges],
                            len(syms)], dtype = complex)
            a = lambdify(parameters_scan, matrix_eq[0], 'numpy')
            b = np.array(matrix_eq[1]).astype(complex)
            for indices in tqdm(itertools.product(*[range(len(r)) for r in scan_ranges]),
                                total = np.product([len(r) for r in scan_ranges])):
                param_values = [scan_ranges[idx][idy] for idx,idy in enumerate(indices)]
                y[indices] = np.linalg.solve(a(*param_values), b)[:,0]
            return y
        else:
            return np.linalg.solve(np.asarray(matrix_eq[0], dtype = complex),
                                    np.asarray(matrix_eq[1], dtype = complex))

    def solveNumeric(self, replacements, tspan, y0, max_step = 1e-1,
                    method = 'RK45'):
        """
        Solve the ODE system of equations dρ(t)/dt = -i[H,ρ]+L numerically.
        Allows for scanning multiple parameters returning an array of
        [i,j,...,n] where i,j,.. are the lengths of the scan_ranges and n is
        the number of parameters to solve for

        Parameters:
        replacements    :   list of tuples, each tuple contains a symbolic
                            variable and the numeric replacement value for that
                            variable
        tspan           :   start and stop time for ODE solver
        y0              :   initial conditions of ODE system
        max_step        :   maximum timestep of ODE solver
        method          :   method of ODE solver

        Returns:
        solution        :   [i,j,...,n] where i,j,.. are the lengths of the
                            scan_ranges and n is the number of parameters solved
                            for
        """
        matrix_eq, t_dependent = self.generateSystem(replacements)

        # ODE solver
        if t_dependent:
            fun = lambda t, rho: matrix_eq(t)@rho
        else:
            fun = lambda t, rho: matrix_eq@rho
        sol = solve_ivp(fun, tspan, y0, method, vectorized = True,
                        max_step = max_step)
        return sol

    def optimizeParametersNumeric(self, replacements, tspan, y0, level,
                                    parameters, bounds, max_step = 1e-1,
                                    method = 'RK45', optimize = "minimum"):
        """
        Use a differential evolution optimizer to find the parameters that get
        the minimum or maximum population in the specified level (ii) after 
        solving the system of ODEsdρ(t)/dt = -i[H,ρ]+L.

        Parameters:
        replacements    :   list of tuples, each tuple contains a symbolic
                            variable and the numeric replacement value for that
                            variable
        tspan           :   start and stop time for ODE solver
        y0              :   initial conditions of ODE system
        level           :   level (ii) to minimize or maximize
        parameters      :   list of parameters to optimize
        bounds          :   which range to search in
        max_step        :   maximum timestep of ODE solver
        method          :   method of ODE solver
        optimize        :   specify to find minimum or maximum

        Returns:
        solution        :   solution of the differential evolution optimizer
        """
        eqns_rhs = self.equations.rhs.subs(replacements)

        # converting the symbolic functions ρ(t) to ρ in order to create the
        # matrix representing the linear equations (Ax=b)
        t = Symbol('t', real = True)
        for i in range(self.levels):
            for j in range(i,self.levels):
                tmp = Function(u'\u03C1{0}{1}'.format(chr(0x2080+i), 
                                                        chr(0x2080+j)))
                tmp1 = Symbol(u'\u03C1{0}{1}'.format(chr(0x2080+j), 
                                                        chr(0x2080+i)))
                eqns_rhs = eqns_rhs.subs(conjugate(tmp(t)), tmp1)

        for i in range(self.levels):
            for j in range(i,self.levels):
                tmp = Function(u'\u03C1{0}{1}'.format(chr(0x2080+i), 
                                                                chr(0x2080+j)))
                tmp1 = Symbol(u'\u03C1{0}{1}'.format(chr(0x2080+i), 
                                                                chr(0x2080+j)))
                eqns_rhs = eqns_rhs.subs(tmp(t), tmp1)

        syms = []
        for i in range(self.levels):
            for j in range(self.levels):
                syms.append(Symbol(u'\u03C1{0}{1}'.format(chr(0x2080+i), 
                                                            chr(0x2080+j))))

        # creating the matrix A (from Ax = b) for the ODE system, has symbolic
        # variables specified in parameters in matrix
        matrix_eq = linear_eq_to_matrix(eqns_rhs, syms)[0]

        # turning matrix into a function with variables parameters
        a = lambdify(parameters, matrix_eq, 'numpy')

        # Set-up ODE solver function for differential evolution
        ode = lambda t, rho, param_values: a(*param_values)@rho
        if optimize == "minimum":
            funEvo = lambda x: solve_ivp(ode, tspan, y0, method, args = (x,),
                                    vectorized = True, max_step = max_step) \
                                        .y[self.levels*level + level,-1].real
        elif optimize == "maximum":
            funEvo = lambda x: -solve_ivp(ode, tspan, y0, method, args = (x,),
                                        vectorized = True, max_step = max_step)\
                                        .y[self.levels*level + level,-1].real
        else:
            raise ValueError('Specify optimize either as minimum or maximum.')

        sol = differential_evolution(funEvo, bounds = bounds, tol = 1e-4)
        return sol