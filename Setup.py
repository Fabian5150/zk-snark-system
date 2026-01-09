from py_ecc.bn128 import G1, G2, pairing, add, multiply, eq, curve_order, neg
import numpy as np
import random
import galois

class Setup:
    # finite field over which G1, G2 and G12 are defined
    GF = galois.GF(curve_order)

    """
    Returns a random interger between zero and the G1/G2-curve order (exclusive)
    """
    @staticmethod
    def __get_random_scalar():
        return random.randint(1, curve_order - 1) # Must not be 0

    """
    Calulates the structure reference string; powers of tau in a elliptic curve group
    Needs to be passed the max polynomial degree of the QAP,
    to calculate enough powers of tau
    """
    def __get_srs(self, generator):
        return [
            multiply(generator, self.tau**i)
            for i in range(self.poly_degree,-1,-1)
        ]

    """
    Calculates t(tau) for the auxilary polynomial t(x) = (x-1)(x-2)...(x-n)
    and returns the srs for it
    """
    def __build_aux_poly(self):
        t_xs = self.GF(np.arange(1, self.poly_degree + 1))
        tau_GF = self.GF(self.tau)
        t_tau_GF = np.prod(tau_GF - t_xs)

        return [
            multiply(G1, int((tau_GF**i) * t_tau_GF))
            for i in range(self.poly_degree - 3, -1, -1)
        ]

    """
    Takes the R1CS as a already interpolated QAP (2d array of coefficients)
    to construct the trusted setup
    """
    def __init__(
        self,
        out_poly: np.ndarray, # polynomials of the output side of the qap
        left_poly: np.ndarray, # polynomials of the left factor of the qap
        right_poly: np.ndarray, # polynomial of the right factor of the qap
    ):
        ### to be kept private
        # for powers of tau
        self.tau = self.__get_random_scalar()

        # for multiplication with the QAP matrices
        self.alpha = self.__get_random_scalar()
        self.beta = self.__get_random_scalar()
        ###

        poly_degree = len(out_poly[0])

        self.g1_srs = self.__get_srs(poly_degree + 1, G1, self.tau)
        self.g2_srs = self.__get_srs(poly_degree + 1, G2, self.tau)

        self.alpha_g1 = multiply(G1, self.alpha)
        self.beta_g1 = multiply(G1, self.beta)
        self.beta_g2 = multiply(G2, self.beta)

        self.t_tau_srs = self.__build_aux_poly()