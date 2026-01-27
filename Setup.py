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
        right_polys: np.ndarray, # polynomials of the right factor of the qap,
        tau=None, # for determinstic tesing
        alpha=None, # for determinstic tesing
        beta=None # for determinstic tesing
    ):
        self.out_polys = out_polys
        self.left_polys = left_polys
        self.right_polys = right_polys
        
        ### to be kept private
        # for powers of tau
        
        self.tau = tau if tau is not None else self.__get_random_scalar()
        self.tau_GF = self.GF(self.tau)

        # for multiplication with the QAP matrices
        self.alpha = alpha if alpha is not None else self.__get_random_scalar()
        self.beta = beta if beta is not None else self.__get_random_scalar()
        ###

        self.num_polys = len(self.out_polys)
        self.num_constraints = len(self.out_polys[0].coeffs)

        self.g1_srs = self.__get_srs(G1)
        self.g2_srs = self.__get_srs(G2)

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
            multiply(generator, int(pow(self.tau, i, curve_order)))
            for i in range(self.num_constraints - 1,-1,-1)
        ]

    """
    Calculates t(tau) for the auxilary polynomial t(x) = (x-1)(x-2)...(x-n)
    and returns the srs for it
    """
    def __build_aux_poly(self):
        t_xs = self.GF(np.arange(1, self.num_constraints + 1))
        t_tau_GF = np.prod(self.tau_GF - t_xs)

        return [
            multiply(G1, int((self.tau_GF**i) * t_tau_GF))
            for i in range(self.num_constraints - 2, -1, -1)
        ]

    """
    Evaluates the QAP's polynomials at tau and constructs
    their linear combination as corresponding G1 curve point row wise
    G1(alpha * left_poly_i(tau) + beta * left_poly_i(tau) + out_poly_i(tau))
    """
    def __evaluate_qap_polys(self):
        psis = []
        
        for i in range(self.num_polys):
            # Horner's polynomial evaluation algorithm with modular arithmetic
            def poly_eval_mod(poly_obj, x, mod):
                # Handle both galois.Poly objects and regular arrays
                coeffs = poly_obj.coeffs if hasattr(poly_obj, 'coeffs') else poly_obj # => accepts both galois poly objects and np coeff arrays
                res = 0
                for coeff in coeffs:
                    res = (res * x + int(coeff)) % mod
                return res
            
            val_left = poly_eval_mod(self.left_polys[i], self.tau, curve_order)
            val_right = poly_eval_mod(self.right_polys[i], self.tau, curve_order)
            val_out = poly_eval_mod(self.out_polys[i], self.tau, curve_order)
            
            # Psi_i = (alph*v_i(tau) + beta*u_i(tau) + w_i(tau))G_1
            combined = (
                self.alpha * val_right +
                self.beta * val_left +
                val_out
            ) % curve_order
            
            psis.append(multiply(G1, combined))
        
        return psis

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
    
if __name__ == "__main__":
    print("Small internal test")

    import numpy as np

    # Dummy-QAP
    left_polys = np.array([[1, 2], [3, 4]])
    right_polys = np.array([[5, 6], [7, 8]])
    out_polys = np.array([[9, 10], [11, 12]])

    setup = Setup(out_polys=out_polys,
                left_polys=left_polys,
                right_polys=right_polys)

    print("Trusted setup created!")
    print(setup.get_setup())