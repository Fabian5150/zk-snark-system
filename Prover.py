from py_ecc.bn128 import multiply, G1, curve_order, add
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

        self.A_1 = self.__compute_AB_point(alpha_g1, g1_srs)
        self.B_2 = self.__compute_AB_point(beta_g2, g2_srs)

        self.h = self.__compute_h_tau()
        self.C_1 = self.__compute_C_point()

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
        sum_l = sum(w_val * l_val for w_val, l_val in zip(self.witness, self.left_eval))
        sum_r = sum(w_val * r_val for w_val, r_val in zip(self.witness, self.right_eval))
        sum_o = sum(w_val * o_val for w_val, o_val in zip(self.witness, self.out_eval))

        res = (sum_l * sum_r - sum_o) % curve_order

        return res
    
    """
    Using h, psis from the prover and the witness,
    computes the [C]_1
    """
    def __compute_C_point(self):
        psi_sum_point = None # representing the neutral element on ell. curve ("point at inf")

        for psi_point, w_val in zip(self.psis, self.witness):
            term = multiply(psi_point, w_val % curve_order)
            psi_sum_point = term if psi_sum_point is None else add(psi_sum_point, term)

        aux_term = multiply(G1, self.h % curve_order)

        C_1 = add(psi_sum_point, aux_term)

        return C_1
    
    """
    Returns the three curve points making up the proof
    """
    def get_proof(self):
        return {
            "A": self.A_1,
            "B": self.B_2,
            "C": self.C_1
        }

