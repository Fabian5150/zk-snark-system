from py_ecc.bn128 import multiply, curve_order, add
import numpy as np

class Prover:
    """
    Takes as input the witness to prove with,
    aswell as the needed curve scalars,
    evaluated qap polynomials and the two srs's from the setup
    """
    def __init__(
        self,
        witness: list[int],
        left_eval,
        right_eval,
        out_eval,
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
        self.left_eval = left_eval
        self.right_eval = right_eval
        self.out_eval = out_eval

        self.a_1 = self.__compute_AB_point(alpha_g1, g1_srs)
        self.a_1 = self.__compute_AB_point(beta_g2, g2_srs)

        self.h = self.__compute_h_tau()

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
    
    """
    Calculates the value of h(tau)t(tau)
    (as h = .../t, t_srs is not needed here, as it disapears)
    """
    def __compute_h_tau(self):
        sum_l = sum(int(a) * int(l_val) for a, l_val in zip(self.witness, self.left_eval))
        sum_r = sum(int(a) * int(r_val) for a, r_val in zip(self.witness, self.right_eval))
        sum_o = sum(int(a) * int(o_val) for a, o_val in zip(self.witness, self.out_eval))

        res = (sum_l * sum_r - sum_o) % curve_order

        return res
