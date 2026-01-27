import numpy as np
import galois
from py_ecc.bn128 import G1, G2, multiply, curve_order, pairing, curve_order
import pickle
from utils import project_path
import unittest

# for class import from parent dir (god forbid I could just use a file path for imports...)
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)
#

from Setup import Setup

class TestSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.GF = galois.GF(curve_order)

        with open(project_path("test", "qap_data.pkl"), "rb") as f:
            cls.data = pickle.load(f)
        
        # determinstic scalar values
        cls.tau = 7
        cls.alpha = 3
        cls.beta = 5
        
        cls.setup = Setup(
            out_polys=cls.data["out_polys"],
            left_polys=cls.data["left_polys"],
            right_polys=cls.data["right_polys"],
            tau=cls.tau,
            alpha=cls.alpha,
            beta=cls.beta
        )
    
    """
    Checks whether the qap's length and polynimal degrees are extracted properly
    """
    def test_00_basic_dimensions(self):
        # witness: [1, out, x, y, v1, v2, v3]
        self.assertEqual(self.setup.num_polys, 7)

        self.assertEqual(self.setup.num_constraints, 4)
        self.assertEqual(self.setup.poly_degree, 3)
    
    """
    Checks whether the srs' contain enough powers of tau for all constrains
    """
    def test_01_srs_lengths(self):
        expected_srs_length = self.setup.num_constraints
        self.assertEqual(len(self.setup.g1_srs), expected_srs_length)
        self.assertEqual(len(self.setup.g2_srs), expected_srs_length)
        
        expected_t_length = self.setup.num_constraints - 1
        self.assertEqual(len(self.setup.t_tau_srs), expected_t_length)
        
        self.assertEqual(len(self.setup.psis), self.setup.num_polys)
    
    """
    Checks if the g1 srs contains all needed powers of tau for the example
    """
    def test_02_g1_srs_powers(self):
        expected = multiply(G1, (self.tau**3) % curve_order)
        self.assertEqual(self.setup.g1_srs[0], expected)
        
        expected = multiply(G1, (self.tau**2) % curve_order)
        self.assertEqual(self.setup.g1_srs[1], expected)
        
        expected = multiply(G1, self.tau)
        self.assertEqual(self.setup.g1_srs[2], expected)
        
        self.assertEqual(self.setup.g1_srs[3], G1)
    
    """
    Checks if the g2 srs contains all needed powers of tau for the example
    """
    def test_03_g2_srs_powers(self):
        expected = multiply(G2, (self.tau**3) % curve_order)
        self.assertEqual(self.setup.g2_srs[0], expected)
        
        self.assertEqual(self.setup.g2_srs[-1], G2)
    
    """
    Checks the proper computations of some of the powers of tau per srs
    """
    def test_04_t_tau_computation(self):
        tau_gf = self.GF(self.tau)
        t_values = self.GF(np.array([1, 2, 3, 4]))
        t_tau = np.prod(tau_gf - t_values)
        
        expected_last = multiply(G1, int(t_tau))
        self.assertEqual(self.setup.t_tau_srs[-1], expected_last)
        
        expected = multiply(G1, int(tau_gf * t_tau) % curve_order)
        self.assertEqual(self.setup.t_tau_srs[-2], expected)
        
        expected = multiply(G1, int(tau_gf**2 * t_tau) % curve_order)
        self.assertEqual(self.setup.t_tau_srs[0], expected)
    
    """
    Checks if all psi terms are correct
    """
    def test_05_psi_computations(self):
        for i in range (0, len(self.data["left_polys"])-1):
            u_poly = self.data["left_polys"][i]
            v_poly = self.data["right_polys"][i]
            w_poly = self.data["out_polys"][i]
            
            u_tau = self.poly_eval_mod(u_poly, self.tau, curve_order)
            v_tau = self.poly_eval_mod(v_poly, self.tau, curve_order)
            w_tau = self.poly_eval_mod(w_poly, self.tau, curve_order)
            
            psi_scalar = (self.alpha * v_tau + self.beta * u_tau + w_tau) % curve_order
            expected_psi = multiply(G1, psi_scalar)
            
            self.assertEqual(self.setup.psis[i], expected_psi)

    """
    Checks the consistency of the encrypted elements across the g1 and g2 group for alpha,
    which is necesarry for the verifier
    e([alpha]_1, G_2) = e(G_1, [alpha]_2)
    """
    def test_06_pairing_check_alpha(self):
        alpha_g2 = multiply(G2, self.alpha)
        
        left_pairing = pairing(G2, self.setup.alpha_g1)
        right_pairing = pairing(alpha_g2, G1)
        
        self.assertEqual(left_pairing, right_pairing)
    
    """
    Checks the consistency of the encrypted elements across the g1 and g1 group for beta,
    which is necesarry for the verifier
    e([beta]_1, G_2) = e(G_1, [beta]_2)
    """
    def test_07_pairing_check_beta(self):
        left_pairing = pairing(G2, self.setup.beta_g1)
        right_pairing = pairing(self.setup.beta_g2, G1)
        
        self.assertEqual(left_pairing, right_pairing)

    # Horner's method for polynomial evaluation with modular arithmetic
    @staticmethod
    def poly_eval_mod(poly_obj, x, mod):
        coeffs = poly_obj.coeffs if hasattr(poly_obj, 'coeffs') else poly_obj
        res = 0
        for coeff in coeffs:
            res = (res * x + int(coeff)) % mod
        return res


def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSetup)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    run_tests()