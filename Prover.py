from py_ecc.bn128 import multiply, G1, curve_order, add
import numpy as np
import galois

class Prover:
    GF = galois.GF(curve_order)

    """
    Takes as input the witness to prove with,
    aswell as the needed curve scalars,
    evaluated qap polynomials and the two srs's from the setup
    """
    def __init__(
        self,
        witness: np.ndarray,
        left_polys: np.ndarray,
        right_polys: np.ndarray,
        out_polys: np.ndarray,
        alpha_g1,
        beta_g2,
        g1_srs,
        g2_srs,
        t_tau_srs,
        psis,
    ):
        self.witness = witness
        self.left_polys = left_polys
        self.right_polys = right_polys
        self.out_polys = out_polys
        
        self.alpha_g1 = alpha_g1
        self.beta_g2 = beta_g2
        self.g1_srs = g1_srs
        self.g2_srs = g2_srs
        self.t_tau_srs = t_tau_srs
        self.psis = psis

        self.A_1 = self.__compute_AB(self.alpha_g1, self.g1_srs, self.left_polys)
        self.B_2 = self.__compute_AB(self.beta_g2, self.g2_srs, self.right_polys)

        self.h_coeffs = self.__compute_h_coeffs()
        self.h = self.__compute_h_tau()
        self.C_1 = self.__compute_C_point()

    """
    Can construct both the [A]_1 point (with alpha_g1, srs_g1 and left_polys)
    and the [B]_2 point (with beta_g2, srs_g2 and right_polys)
    """
    def __compute_AB(self, scalar, srs, polys):        
        # Start with zero polynomial
        res_poly = None
        
        for i, witness_val in enumerate(self.witness):
            poly = polys[i]
            
            mul_poly = poly * self.GF(int(witness_val))
            
            if res_poly is None:
                res_poly = mul_poly
            else:
                res_poly = res_poly + mul_poly
        
        coeffs = res_poly.coeffs
        
        srs_len = len(srs)
        
        # Pad with 0-coefficients if necessary (galois shortens polynomials with 0-coefficients)
        if len(coeffs) < srs_len:
            coeffs = np.concatenate([np.zeros(srs_len - len(coeffs), dtype=coeffs.dtype), coeffs])
        
        acc = None
        for coeff, srs_element in zip(coeffs, srs):
            term = multiply(srs_element, int(coeff) % curve_order)
            acc = term if acc is None else add(acc, term)
        
        
        return add(scalar, acc) if acc is not None else scalar
    
    def __compute_h_coeffs(self):
        """
        Computes sum(witness_i * poly(x)) per given polynomial array
        """
        def compute_poly(polys):
            res_poly = None

            for i, witness_val in enumerate(self.witness):
                poly = polys[i]
                scaled = poly * self.GF(int(witness_val))
                res_poly = scaled if res_poly is None else res_poly + scaled

            return res_poly
            
        L_poly = compute_poly(self.left_polys)
        R_poly = compute_poly(self.right_polys)
        O_poly = compute_poly(self.out_polys)

        numerator = L_poly * R_poly - O_poly
        
        # Compute t(x)
        num_constraints = len(self.g1_srs)
        roots = self.GF(np.arange(1, num_constraints + 1))
        t_poly = galois.Poly([1], field=self.GF)

        for root in roots:
            t_poly = t_poly * galois.Poly([1, -root], field=self.GF)
        
        h_poly, remainder = divmod(numerator, t_poly)
        
        # Check for remainer (not 0 => invalid witness)
        if not np.all(remainder.coeffs == 0):
            raise ValueError("Invalid witness! (has devision remainder)")
        
        return h_poly.coeffs

    """
    Calculates the value of h(tau)t(tau)
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

