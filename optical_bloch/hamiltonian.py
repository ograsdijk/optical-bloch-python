import numpy as np
from sympy import conjugate
from sympy.matrices import zeros, eye
from sympy.functions.elementary.exponential import exp as symb_exp
from sympy import symbols, Symbol, simplify, conjugate, solve, diff

class Hamiltonian:
    """
    Class for setting up the hamiltonian for a ODE system of Bloch equations.
    """
    def __init__(self, levels):
        """
        """
        # total number of states
        self.levels = levels

        # defining the symbolic Hamiltonian matrix
        self.hamiltonian = zeros(levels, levels)

        # defining the couplings and rabi rate matrices
        # couplings contains the frequencies between levels
        # rabi contains the rabi rates of the couplings between levels
        self.couplings = zeros(levels, levels)
        self.rabi = zeros(levels, levels)

        # level detunings list
        self.detunings = []

        # numer of state couplings from an initial level
        self.cpl = np.zeros([levels,1])

        # frequencies list
        self.frequencies = []

        # energies list
        self.energies = self.hamiltonian.diagonal()

        # by default no zero energy defined
        self.zero_energy = None

    def defineZero(self,zero_energy):
        """
        Defining the zero level energy

        Parameters:
        zero_energy : state Symbol() which to set to zero
        """
        if not zero_energy in self.energies.free_symbols:
            raise AssertionError('Specified energy not in any of the energy levels.')
        self.zero_energy = zero_energy
        self.transformed = self.transformed.subs(zero_energy, 0)

    def addEnergies(self,energies):
        """
        Adding energy levels to the Hamiltonian

        Parameters:
        energies : list of state Symbol() which define the energy levels
        """
        if not len(energies) == self.levels:
            raise AssertionError('Not all energies specified.')

        # energies are on the diagonal of the Hamiltonian matrix
        for idx, energy in enumerate(energies):
            self.hamiltonian[idx,idx] = energy
        self.energies = self.hamiltonian.diagonal()

    def addCoupling(self, initial_state, final_state, rabi_rate, omega):
        """
        Add a coupling between two states

        Parameters:
        initial_state  : initial coupled state
        final_state    : final coupled state
        rabi_rate      : rabi rate of coupling between initial and final, Symbol
        omega          : requency of coupling between initial and final, Symbol
        """
        if (initial_state > self.hamiltonian.shape[0]) or (final_state > self.hamiltonian.shape[0]):
            raise AssertionError('Specified state exceeds size of Hamiltonian')

        # setting the frequency and rabi rate of the coupling to the symbolic matrices
        self.couplings[initial_state, final_state] = omega
        self.rabi[initial_state, final_state]      = rabi_rate

        # adding the coupling frequency to the frequencies list if not already present
        if omega not in self.frequencies:
            self.frequencies.append(omega)

        # adding the appropriote terms to the symbolic Hamiltonian matrix
        t = Symbol('t', real = True)
        self.hamiltonian[initial_state, final_state] = -rabi_rate/2*symb_exp(1j*omega*t)
        self.hamiltonian[final_state, initial_state] = -conjugate(rabi_rate)/2*symb_exp(-1j*omega*t)
        # incrementing the number of couplings counter from a specific initial state
        self.cpl[initial_state] += 1

    def addPolyCoupling(self, initial_state, final_state, rabi_rate, omega):
        """
        """
        if (inital_state > self.levels) or (final_state > self.levels):
            raise AssertionError('Specified state exceeds size of Hamiltonian')

        self.couplings[initial_state, final_state] = omega
        self.rabi[initial_state, final_state]      = rabi_rate

        if omega not in self.frequencies:
            self.frequencies.append(omega)

        t = Symbol('t', real = True)
        self.hamiltonian[initial_state, final_state] -= rabi_rate/2*symb_exp(1j*omega*t)
        self.hamiltonian[final_state, initial_state] -= conjugate(rabi_rate)/2*symb_exp(-1j*omega*t)
        self.cpl[initial_state] += 1

    def defineStateDetuning(self, initial_state, final_state, detuning):
        """
        Adding a state detuning, requires that self.transformed is defined

        Paramters:
        initial_state : initial coupled state
        final_state   : final coupled state
        detuning      : state detuning, Symbol
        """
        if detuning in self.detunings:
            raise AssertionError('Detuning already defined.')

        # check if the coupling for which the state detuning is requested exists
        if (self.couplings[initial_state, final_state] == 0) and \
           (self.couplings[final_state, initial_state] == 0):
            raise AssertionError('No coupling between states')

        # couplings are defined as initial->final; grab non-zero frequency from initial->final or final->initial
        # (accounts for user error in specifying the detuning initial and final state)
        w = self.couplings[initial_state, final_state]
        if w == 0:
            w = self.couplings[final_state, initial_state]

        # adding the detuning the the transformed matrix
        self.transformed = self.transformed.subs(w,self.hamiltonian[final_state,final_state] - self.hamiltonian[initial_state, initial_state] - detuning)

        # if the zero_energy is added via the state detuning, set it to zero
        if self.zero_energy:
            self.transformed = self.transformed.subs(self.zero_energy, 0)

        # append the detuning to the detunings list
        self.detunings.append([self.hamiltonian[initial_state, initial_state],
                                                self.hamiltonian[final_state, final_state],
                                                detuning, w])

    def defineEnergyDetuning(self, inital_energy, final_energy, detuning, omega):
        """
        """
        if detuning in self.detunings:
            raise AssertionError('Detuning already defined.')

        if omega not in self.frequencies:
            raise AssertionError('Coupling does not exist')

        self.transformed = self.transformed.subs(omega, final_energy - initial_energy - detuning)

        if self.zero_energy:
            self.transformed = self.transformed.subs(self.zero-energy, 0)

        self.detunings.append([initial_energy, final_energy, detuning, omega])


    def eqnTransform(self):
        """
        Calculate the rotational wave approximation by solving a system of equations, only usable if
        the number of couplings does not exceed the number of states
        """
        A = symbols(f'a0:{self.levels}')

        Eqns = []
        for i in range(len(A)):
            for j in range(len(A)):
                if self.couplings[i,j] != 0:
                    Eqns.append(self.couplings[i,j] - (A[i] - A[j]))

        sol = solve(Eqns, A)
        free_params = [value for value in A if value not in list(sol.keys())]
        for free_param in free_params:
            for key,val in sol.items():
                sol[key] = val.subs(free_param, 0)


        T = zeros(*self.hamiltonian.shape)
        for i in range(self.hamiltonian.shape[0]):
            try:
                T[i,i] = symb_exp(1j*sol[Symbol(f'a{i}')]*Symbol('t', real = True))
            except KeyError:
                T[i,i] = 1
        self.T = T

        self.transformed = T.adjoint()@self.hamiltonian@T-1j*T.adjoint()@diff(T,Symbol('t', real = True))
        self.transformed = simplify(self.transformed)

        for i in range(self.levels):
            for j in range(i+1,self.levels):
                if self.transformed[i,j] != 0:
                    val = self.transformed[i,j]*2/self.rabi[i,j]
                    if val not in [-1,1]:
                        raise AssertionError('Could not find unitary transformation')  \

    def shortestCouplingPath(self, graph, initial_state):
        """
        """
        # find the indices of the states which are defined as the zero_energy
        indices_zero_energy = np.where(np.array(self.energies) == self.zero_energy)[1]
        if indices_zero_energy.size == 0:
            return 0

        # get the first succesful shortest path from the initial state to a state with zero_energy
        for idx in indices_zero_energy:
            shortest_path = nx.algorithms.shortest_path(graph, source=initial_state, target = idx, weight = 'weight')
            if shortest_path:
                break

        print(initial_state, shortest_path)

        # calculate the total coupling phase between the initial state and final state
        phase = 0
        for j in range(len(shortest_path)-1):
            start, stop = shortest_path[j:j+2]
            if self.couplings[start,stop] != 0:
                phase += self.couplings[start, stop]
            else:
                phase -= self.couplings[stop, start]
        return phase

    def generalTransform(self, graph):
        """
        """
        if not self.zero_energy:
            raise AssertionError('Zero energy has to be specified for this transformation method.')

        t = Symbol('t', real = True)

        T = eye(self.levels)
        for i in range(self.levels):
            phase = self.shortestCouplingPath(graph, i)
            T[i,i] = T[i,i]*symb_exp(1j*phase*t)
        T = simplify(T)

        self.transformed = T.adjoint()@self.hamiltonian@T-1j*T.adjoint()@diff(T,Symbol('t', real = True))

        if self.detunings:
            print(self.detunings)
            for i in range(len(self.detunings)):
                detuning = self.detunings[i]
                self.transformed = self.transformed.subs(detuning[3], detuning[1] - detuning[0] - detuning[2])

        if self.zero_energy:
            self.transformed = self.transformed.subs(self.zero_energy, 0)

        self.T = T
