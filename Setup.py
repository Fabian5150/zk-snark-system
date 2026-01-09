from py_ecc.bn128 import G1, G2, multiply, curve_order
import numpy as np
import random
import galois

class Setup:
    # finite field over which G1, G2 and G12 are defined
    GF = galois.GF(curve_order)

    """
    Takes the R1CS as a already interpolated QAP (2d array of coefficients)
    to construct the trusted setup
    Polynomials of QAP must be given with coefficients of higher powers first
    """
    def __init__(
        self,
        out_polys: np.ndarray, # polynomials of the output side of the qap
        left_polys: np.ndarray, # polynomials of the left factor of the qap
        right_polys: np.ndarray, # polynomials of the right factor of the qap
    ):
        self.out_polys = out_polys
        self.left_polys = left_polys
        self.right_polys = right_polys
        
        ### to be kept private
        # for powers of tau
        self.tau = self.__get_random_scalar()
        self.tau_GF = self.GF(self.tau)

        # for multiplication with the QAP matrices
        self.alpha = self.__get_random_scalar()
        self.beta = self.__get_random_scalar()
        ###

        self.poly_degree = len(self.out_polys[0])

        self.g1_srs = self.__get_srs(self.poly_degree + 1, G1)
        self.g2_srs = self.__get_srs(self.poly_degree + 1, G2)

        self.alpha_g1 = multiply(G1, self.alpha)
        self.beta_g1 = multiply(G1, self.beta)
        self.beta_g2 = multiply(G2, self.beta)

        self.t_tau_srs = self.__build_aux_poly()
        self.psis = self.__evaluate_qap_polys()
    

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
        t_tau_GF = np.prod(self.tau_GF - t_xs)

        return [
            multiply(G1, int((self.tau_GF**i) * t_tau_GF))
            for i in range(self.poly_degree - 3, -1, -1)
        ]

    """
    Evaluates the QAP's polynomials at tau and constructs
    their linear combination as corresponding G1 curve point row wise
    G1(alpha * left_poly_i(tau) + beta * left_poly_i(tau) + out_poly_i(tau))
    """
    def __evaluate_qap_polys(self):
        left_eval = []
        right_eval = []
        out_eval = []

        for i in range(self.poly_degree):
            val_L = int(np.polyval(self.left_polys[i], self.tau)) % curve_order
            val_R = int(np.polyval(self.right_polys[i], self.tau)) % curve_order
            val_O = int(np.polyval(self.out_polys[i], self.tau)) % curve_order

            left_eval.append(val_L)
            right_eval.append(val_R)
            out_eval.append(val_O)

        
        return [
            multiply(
                G1,
                (self.alpha * right_eval[i] +
                self.beta  * left_eval[i] +
                out_eval[i]) % curve_order)
                for i in range(self.poly_degree
            )
        ]

    """
    Returns the necesarry parts of the setup for prover and verifier as dict
    """
    def get_setup(self):
        return {
            "alpha_g1": self.alpha_g1,
            "beta_g1": self.beta_g1,
            "beta_g2": self.beta_g2,
            "g1_srs": self.g1_srs,
            "g2_srs": self.g2_srs,
            "t_tau_srs": self.t_tau_srs,
            "psis": self.psis,
        }


    """
    Returns a random interger between zero and the G1/G2-curve order (exclusive)
    """
    @staticmethod
    def __get_random_scalar():
        return random.randint(1, curve_order - 1) # Must not be 0