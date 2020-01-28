from sympy.matrices import zeros
from .utils.math import commute, anti_commute
from sympy import Symbol, Function, Rational, sqrt, simplify

class Dissipator:
    """
    Class for setting up the dissipator for the optical Bloch equations.
    """
    def __init__(self, levels):
        """
        """
        self.density_matrix = zeros(levels, levels)
        self.dissipator = zeros(levels, levels)
        self.branching = zeros(levels, levels)
        self.decay_rate = zeros(levels, 1)
        self.levels = levels

        self.generateDensityMatrix()

    def generateDensityMatrix(self):
        """
        """
        self.density_matrix = zeros(self.levels, self.levels)
        t = Symbol('t', real = True)
        for i in range(self.levels):
            for j in range(i,self.levels):
                if i == j:
                    self.density_matrix[i,j] = Function(u'\u03C1{0}{1}'.format(chr(0x2080+i), chr(0x2080+j)))(t)
                else:
                    self.density_matrix[i,j] = Function(u'\u03C1{0}{1}'.format(chr(0x2080+i), chr(0x2080+j)))(t)
                    self.density_matrix[j,i] = conjugate(Function(u'\u03C1{0}{1}'.format(chr(0x2080+i), chr(0x2080+j)))(t))

    def addDecay(self, initial_state, final_state, gamma):
        """
        """
        if (initial_state >= self.levels) or (final_state >= self.levels):
            raise AssertionError('Specified state exceeds number of levels.')
        if initial_state == final_state:
            raise AssertionError('State cannot decay into itself.')

        G = zeros(self.levels, self.levels)
        G[final_state, initial_state] = sqrt(gamma)
        self.dissipator -= Rational(1/2) * anti_commute(G.T@G, self.density_matrix) - G@self.density_matrix@G.T
        self.dissipator = simplify(self.dissipator)

        # update decay rates
        decay_rate_old = self.decay_rate[initial_state]
        decay_rate_new = decay_rate_old + gamma
        self.decay_rate[initial_state] += gamma

        for i in range(self.levels):
            if i == final_state:
                self.branching[initial_state, i] = (self.branching[initial_state, i]*decay_rate_old+gamma)/decay_rate_new
            else:
                self.branching[initial_state, i] = (self.branching[initial_state, i]*decay_rate_old)/decay_rate_new
