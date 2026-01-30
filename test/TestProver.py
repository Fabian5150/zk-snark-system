import unittest
import numpy as np
import galois
from py_ecc.bn128 import G1, G2, multiply, curve_order, pairing, add
import pickle
from utils import project_path

# for class import from parent dir (god forbid I could just use a file path for imports...)
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)
#

from Setup import Setup
from Prover import Prover


class TestProver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.GF = galois.GF(curve_order)
        
        with open(project_path("test", "qap_data.pkl"), "rb") as f:
            cls.data = pickle.load(f)
        
        # determinstic scalar values
        cls.tau = 7
        cls.alpha = 3
        cls.beta = 5
        
        # create setup
        cls.setup = Setup(
            out_polys=cls.data["out_polys"],
            left_polys=cls.data["left_polys"],
            right_polys=cls.data["right_polys"],
            tau=cls.tau,
            alpha=cls.alpha,
            beta=cls.beta
        )
        
        cls.setup_data = cls.setup.get_setup()
        cls.witness = cls.data["correct_witness"]
        
        
        # prover with correct witness
        cls.prover = Prover(
            witness=cls.witness,
            left_polys=cls.data["left_polys"],
            right_polys=cls.data["right_polys"],
            out_polys=cls.data["out_polys"],
            alpha_g1=cls.setup_data["alpha_g1"],
            beta_g2=cls.setup_data["beta_g2"],
            g1_srs=cls.setup_data["g1_srs"],
            g2_srs=cls.setup_data["g2_srs"],
            t_tau_srs=cls.setup_data["t_tau_srs"],
            psis=cls.setup_data["psis"],
        )
    
    """
    Checks if all h has the correct polynomial degree
    (must not be > #constraints - 2)
    """
    def test_00_h_polynomial_degree(self):
        max_degree = self.setup.num_constraints - 2
        
        h_degree = len(self.prover.h_coeffs) - 1
        
        self.assertLessEqual(h_degree, max_degree)
    
    """
    Checks if h was constrcuted correctly
    (h(x) * t(x) = L(x) * R(x) - o(x) must hold)
    """
    def test_01_h_divides_correctly(self):
        GF = self.GF
        
        def weighted_sum(polys, weights):
            result = None
            for poly, weight in zip(polys, weights):
                scaled = poly * GF(int(weight))
                result = scaled if result is None else result + scaled
            return result
        
        # weighted polynomial sum per polynomial matrix as needed in the h-calculation
        L_poly = weighted_sum(self.data["left_polys"], self.witness)
        R_poly = weighted_sum(self.data["right_polys"], self.witness)
        O_poly = weighted_sum(self.data["out_polys"], self.witness)
        
        # calculates t
        roots = GF(np.arange(1, self.setup.num_constraints + 1))
        t_poly = galois.Poly([1], field=GF)
        for root in roots:
            t_poly = t_poly * galois.Poly([1, -root], field=GF)
        
        h_poly = galois.Poly(self.prover.h_coeffs, field=GF)
        
        left_side = h_poly * t_poly
        right_side = L_poly * R_poly - O_poly
        
        self.assertTrue(np.all(left_side.coeffs == right_side.coeffs))
    
    """
    Checks if the prover's A point is correct
    by constructing it for the example and comaring
    TODO: Generalize this method, so it can be used for A and B
    """
    def test_02_A_computation_manual(self):
        GF = self.GF
        tau_gf = GF(self.tau)
        
        sum_val = GF(0)
        for i, w_val in enumerate(self.witness):
            poly = self.data["left_polys"][i]
            poly_at_tau = poly(tau_gf)
            sum_val += GF(int(w_val)) * poly_at_tau
        
        expected_A = add(
            self.setup_data["alpha_g1"],
            multiply(G1, int(sum_val))
        )
        
        self.assertEqual(self.prover.A_1, expected_A)
    
    """
    Checks if the prover's B point is correct
    by constructing it for the example and comaring
    """
    def test_03_B_computation_manual(self):
        GF = self.GF
        tau_gf = GF(self.tau)
        
        sum_val = GF(0)
        for i, w_val in enumerate(self.witness):
            poly = self.data["right_polys"][i]
            poly_at_tau = poly(tau_gf)
            sum_val += GF(int(w_val)) * poly_at_tau
        
        expected_B = add(
            self.setup_data["beta_g2"],
            multiply(G2, int(sum_val))
        )
        
        self.assertEqual(self.prover.B_2, expected_B)
    
    """
    Checks if the prover's C point is correct
    by constructing it for the example and comaring
    """
    def test_04_C_computation_manual(self):
        GF = self.GF
        tau_gf = GF(self.tau)
        
        # calculate the psi sum
        psi_sum = None
        for i, w_val in enumerate(self.witness):
            term = multiply(self.setup_data["psis"][i], int(w_val) % curve_order)
            psi_sum = term if psi_sum is None else add(psi_sum, term)
        
        self.assertIsNotNone(psi_sum)
        
        # calculate h(tau)t(tau)
        roots = GF(np.arange(1, self.setup.num_constraints + 1))
        t_at_tau = np.prod(tau_gf - roots)
        
        h_poly = galois.Poly(self.prover.h_coeffs, field=GF)
        h_at_tau = h_poly(tau_gf)
        
        h_t_product = int(h_at_tau * t_at_tau)
        h_t_point = multiply(G1, h_t_product % curve_order)
        
        # Combine both
        expected_C = add(psi_sum, h_t_point)
        
        
        self.assertEqual(self.prover.C_1, expected_C)
    
    """
    Check if a proof with the correct witness satisfies the example qap
    """
    def test_05_witness_satisfaction(self):
        def eval_poly(poly, x):
            res = 0
            for coeff in poly.coeffs:
                res = (res * x + int(coeff)) % curve_order
            
            return res
        
        for x in [1, 2, 3, 4]: # x values used for interpolation
            u_vals = [eval_poly(poly, x) for poly in self.data["left_polys"]]
            v_vals = [eval_poly(poly, x) for poly in self.data["right_polys"]]
            w_vals = [eval_poly(poly, x) for poly in self.data["out_polys"]]
            
            left = sum(int(self.witness[i]) * u_vals[i] for i in range(7)) % curve_order
            right = sum(int(self.witness[i]) * v_vals[i] for i in range(7)) % curve_order
            out = sum(int(self.witness[i]) * w_vals[i] for i in range(7)) % curve_order
            
            product = (left * right) % curve_order
            
            self.assertEqual(product, out)
    
    """
    Check if the prover correctly detects and stops to construct the proof,
    if the witness is inavlid
    """
    def test_06_false_witness_raises_error(self):
        false_witness = self.data["false_witness"]
        
        with self.assertRaises(ValueError) as context:
            Prover(
                witness=false_witness,
                left_polys=self.data["left_polys"],
                right_polys=self.data["right_polys"],
                out_polys=self.data["out_polys"],
                alpha_g1=self.setup_data["alpha_g1"],
                beta_g2=self.setup_data["beta_g2"],
                g1_srs=self.setup_data["g1_srs"],
                g2_srs=self.setup_data["g2_srs"],
                t_tau_srs=self.setup_data["t_tau_srs"],
                psis=self.setup_data["psis"],
            )
        
        error_msg = str(context.exception)
        self.assertIn("Invalid witness", error_msg)

def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestProver)
    
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    
    return res


if __name__ == '__main__':
    run_tests()