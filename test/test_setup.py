import numpy as np
import galois
from py_ecc.bn128 import curve_order
import pickle
from utils import project_path
import unittest
from py_ecc.bn128 import G1, G2, multiply, curve_order, pairing, add

# for class import from parent dir (god forbid I could just use a file path for imports...)
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)

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
    
    def test_01_basic_dimensions(self):
        # witness: [1, out, x, y, v1, v2, v3]
        self.assertEqual(self.setup.num_polys, 7)

        self.assertEqual(self.setup.num_constraints, 4)
        self.assertEqual(self.setup.poly_degree, 3)
    
    def test_02_srs_lengths(self):
        expected_srs_length = self.setup.num_constraints
        self.assertEqual(len(self.setup.g1_srs), expected_srs_length)
        self.assertEqual(len(self.setup.g2_srs), expected_srs_length)
        
        expected_t_length = self.setup.num_constraints - 1
        self.assertEqual(len(self.setup.t_tau_srs), expected_t_length)
        
        self.assertEqual(len(self.setup.psis), self.setup.num_polys)
    
    def test_03_basic_encrypted_values(self):
        expected_alpha_g1 = multiply(G1, self.alpha)
        self.assertEqual(self.setup.alpha_g1, expected_alpha_g1)
        
        expected_beta_g1 = multiply(G1, self.beta)
        self.assertEqual(self.setup.beta_g1, expected_beta_g1)
        
        expected_beta_g2 = multiply(G2, self.beta)
        self.assertEqual(self.setup.beta_g2, expected_beta_g2)
    
    def test_04_g1_srs_powers(self):
        n = self.setup.num_constraints
        
        expected = multiply(G1, (self.tau**3) % curve_order)
        self.assertEqual(self.setup.g1_srs[0], expected)
        
        expected = multiply(G1, (self.tau**2) % curve_order)
        self.assertEqual(self.setup.g1_srs[1], expected)
        
        expected = multiply(G1, self.tau)
        self.assertEqual(self.setup.g1_srs[2], expected)
        
        self.assertEqual(self.setup.g1_srs[3], G1)
    
    def test_05_g2_srs_powers(self):
        expected = multiply(G2, (self.tau**3) % curve_order)
        self.assertEqual(self.setup.g2_srs[0], expected)
        
        self.assertEqual(self.setup.g2_srs[-1], G2)
    
    def test_06_t_tau_computation(self):
        tau_gf = self.GF(self.tau)
        t_values = self.GF(np.array([1, 2, 3, 4]))
        t_tau = np.prod(tau_gf - t_values)
        
        expected_last = multiply(G1, int(t_tau))
        self.assertEqual(self.setup.t_tau_srs[-1], expected_last)
        
        expected = multiply(G1, int((tau_gf * t_tau) % curve_order))
        self.assertEqual(self.setup.t_tau_srs[-2], expected)
        
        expected = multiply(G1, int((tau_gf**2 * t_tau) % curve_order))
        self.assertEqual(self.setup.t_tau_srs[0], expected)
    
    def test_07_psi_computation_first_element(self):
        i = 0
        
        u_poly = self.data["left_polys"][i]
        v_poly = self.data["right_polys"][i]
        w_poly = self.data["out_polys"][i]
        
        u_tau = int(np.polyval(u_poly, self.tau)) % curve_order
        v_tau = int(np.polyval(v_poly, self.tau)) % curve_order
        w_tau = int(np.polyval(w_poly, self.tau)) % curve_order
        
        psi_scalar = (self.alpha * v_tau + self.beta * u_tau + w_tau) % curve_order
        expected_psi = multiply(G1, psi_scalar)
        
        self.assertEqual(self.setup.psis[i], expected_psi)
    
    def test_08_psi_computation_x_element(self):
        i = 2
        
        u_poly = self.data["left_polys"][i]
        v_poly = self.data["right_polys"][i]
        w_poly = self.data["out_polys"][i]
        
        u_tau = int(np.polyval(u_poly, self.tau)) % curve_order
        v_tau = int(np.polyval(v_poly, self.tau)) % curve_order
        w_tau = int(np.polyval(w_poly, self.tau)) % curve_order
        
        psi_scalar = (self.alpha * v_tau + self.beta * u_tau + w_tau) % curve_order
        expected_psi = multiply(G1, psi_scalar)
        
        self.assertEqual(self.setup.psis[i], expected_psi)
    
    def test_09_all_psis_non_trivial(self):
        identity = (0, 0, 0)
        
        for i, psi in enumerate(self.setup.psis):
            self.assertNotEqual(psi, identity)
    
    def test_10_pairing_check_alpha(self):
        alpha_g2 = multiply(G2, self.alpha)
        
        left_pairing = pairing(G2, self.setup.alpha_g1)
        right_pairing = pairing(alpha_g2, G1)
        
        self.assertEqual(left_pairing, right_pairing)
    
    def test_11_pairing_check_beta(self):
        left_pairing = pairing(G2, self.setup.beta_g1)
        right_pairing = pairing(self.setup.beta_g2, G1)
        
        self.assertEqual(left_pairing, right_pairing)
    
    def test_12_witness_evaluation(self):
        witness = self.data["correct_witness"]
        
        for constraint_x in [1, 2, 3, 4]:
            u_vals = [int(np.polyval(poly, constraint_x)) % curve_order 
                     for poly in self.data["left_polys"]]
            v_vals = [int(np.polyval(poly, constraint_x)) % curve_order 
                     for poly in self.data["right_polys"]]
            w_vals = [int(np.polyval(poly, constraint_x)) % curve_order 
                     for poly in self.data["out_polys"]]
            
            left = sum(int(witness[i]) * u_vals[i] for i in range(7)) % curve_order
            right = sum(int(witness[i]) * v_vals[i] for i in range(7)) % curve_order
            out = sum(int(witness[i]) * w_vals[i] for i in range(7)) % curve_order
            
            product = (left * right) % curve_order
            
            self.assertEqual(product, out)
    
    def test_13_full_setup_output(self):
        setup_output = self.setup.get_setup()
        
        required_keys = ["alpha_g1", "beta_g1", "beta_g2", "g1_srs", 
                        "g2_srs", "t_tau_srs", "psis"]
        
        for key in required_keys:
            self.assertIn(key, setup_output)


def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSetup)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    run_tests()