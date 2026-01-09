from py_ecc.bn128 import G1, G2, pairing, add, multiply, eq, curve_order, neg
import numpy as np
import random

class Setup:
    """
    Returns a random interger between zero and the G1/G2-curve order (exclusive)
    """
    def __get_random_scalar():
        return random.randint(1, curve_order - 1) # Must not be 0

    """
    Calulates the structure reference string; powers of tau in a elliptic curve group
    Needs to be passed the max polynomial degree of the QAP,
    to calculate enough powers of tau
    """
    def __get_srs(self, poly_degree, generator, tau):
        return [multiply(generator, tau**i) for i in range(poly_degree,-1,-1)]

    """
    Takes the R1CS as a already interpolated QAP (2d array of coefficients)
    to construct the trusted setup
    """
    def __init__(
        self,
        out_poly: np.ndarray, # polynomials of the output side of the qap
        left_poly: np.ndarray, # polynomials of the left factor of the qap
        right_poly: np.ndarray, # polynomial of the right factor of the qap
        witness: np.ndarray
    ):
        ### to be kept private
        # for powers of tau
        tau = self.__get_random_scalar()

        # random scalars to be kept secret from the prover
        alpha = self.__get_random_scalar()
        beta = self.__get_random_scalar()

        # divisors for the public and the private parts of the witness
        gamma = self.__get_random_scalar()
        delta = self.__get_random_scalar()
        ###

        poly_degree = len(out_poly[0])

        self.g1_srs = self.__get_srs(poly_degree, G1, tau)
        self.g2_srs = self.__get_srs(poly_degree, G2, tau)

        self.alpa_g1 = multiply(alpha, G1)
        self.beta_g1 = multiply(beta, G1)
        self.beta_g2 = multiply(beta, G2)
        self.gamma_g1 = multiply(gamma, G1)
        self.delta_g1 = multiply(delta, G1)
        self.delta_g2 = multiply(delta, G2)


        # auxilary polynomial (t(x) = (x-1)(x-2)...(x-n))
        # as only t(tau) is needed, it can just be calculated directly
        # and used for the srs
        t_xs = np.arange(1,poly_degree+1)
        self.t_tau = np.prod([(self.tau-i)%curve_order for i in t_xs]) % curve_order
        
        self.t_tau_srs = self.__get_srs(poly_degree)