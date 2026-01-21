from py_ecc.bn128 import G1, G2, multiply, curve_order, add
import numpy as np

class Prover:
    """
    Takes as input the witness to prove with,
    aswell as the needed curve scalars and srs from the setup
    """
    def __init__(
        self,
        witness: list[int],
        alpha_g1,
        beta_g2,
        g1_srs,
        g2_srs,
        t_tau_srs,
        psis,
    ):
        self.witness = witness
        self.g1_srs = g1_srs
        self.g2_srs = g2_srs
        self.t_tau_srs = t_tau_srs
        self.psis = psis

        self.a_1 = self.__compute_AB_point(alpha_g1, g1_srs)
        self.a_1 = self.__compute_AB_point(beta_g2, g2_srs)

    """
    Can construct both the [A]_1 point (with alpha_g1 and srs_g1)
    and the [B]_2 point (with beta_g2 and srs_g2)
    """
    def __compute_AB_point(self, scalar, srs):
        acc = scalar

        for i, w_val in enumerate(self.witness):
            term = multiply(srs[i], w_val % curve_order)
            acc = add(acc, term)

        return acc
    
    def compute_C_point(self):
