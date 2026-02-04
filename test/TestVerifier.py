import galois
from py_ecc.bn128 import multiply, curve_order
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
from Prover import Prover
from Verifier import Verifier

class TestVerifier(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.GF = galois.GF(curve_order)
        
        with open(project_path("test", "qap_data.pkl"), "rb") as f:
            cls.data = pickle.load(f)
        
        cls.setup = Setup(
            out_polys=cls.data["out_polys"],
            left_polys=cls.data["left_polys"],
            right_polys=cls.data["right_polys"],
        )
        
        cls.setup_data = cls.setup.get_setup()
    
    """
    Checks that a proof with correct witness verifies as valid
    """
    def test_00_correct_witness_verifies(self):
        correct_witness = self.data["correct_witness"]
        
        # Generate proof with correct witness
        prover = Prover(
            witness=correct_witness,
            left_polys=self.data["left_polys"],
            right_polys=self.data["right_polys"],
            out_polys=self.data["out_polys"],
            alpha_g1=self.setup_data["alpha_g1"],
            beta_g2=self.setup_data["beta_g2"],
            g1_srs=self.setup_data["g1_srs"],
            g2_srs=self.setup_data["g2_srs"],
            t_tau_srs=self.setup_data["t_tau_srs"],
            psis=self.setup_data["psis"]
        )
        
        proof = prover.get_proof()
        
        # Verify proof
        verifier = Verifier(
            A=proof["A"],
            B=proof["B"],
            C=proof["C"],
            alpha_1=self.setup_data["alpha_g1"],
            beta_2=self.setup_data["beta_g2"]
        )
        
        self.assertTrue(verifier.isValid)
    
    """
    Checks that a proof with incorrect witness fails verification
    """
    def test_01_false_witness_fails(self):
        false_witness = self.data["false_witness"]
        
        prover = Prover (
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
            allowFalseWitness=True
        )
        
        proof = prover.get_proof()
        
        verifier = Verifier(
            A=proof["A"],
            B=proof["B"],
            C=proof["C"],
            alpha_1=self.setup_data["alpha_g1"],
            beta_2=self.setup_data["beta_g2"]
        )
        
        self.assertFalse(verifier.isValid)
    
    """
    Checks that verification with manipulated proof fails
    """
    def test_02_manipulated_proof_fails(self):
        correct_witness = self.data["correct_witness"]
        
        # valid proof
        prover = Prover(
            witness=correct_witness,
            left_polys=self.data["left_polys"],
            right_polys=self.data["right_polys"],
            out_polys=self.data["out_polys"],
            alpha_g1=self.setup_data["alpha_g1"],
            beta_g2=self.setup_data["beta_g2"],
            g1_srs=self.setup_data["g1_srs"],
            g2_srs=self.setup_data["g2_srs"],
            t_tau_srs=self.setup_data["t_tau_srs"],
            psis=self.setup_data["psis"]
        )
        
        proof = prover.get_proof()
        
        # manipulated A point
        manipulated_A = multiply(proof["A"], 2)
        
        verifier = Verifier(
            A=manipulated_A,
            B=proof["B"],
            C=proof["C"],
            alpha_1=self.setup_data["alpha_g1"],
            beta_2=self.setup_data["beta_g2"]
        )
        
        self.assertFalse(verifier.isValid)
    
    """
    Checks that swapping proof components fails verification
    """
    def test_03_swapped_components_fail(self):
        correct_witness = self.data["correct_witness"]
        
        # Generate two different valid proofs
        prover1 = Prover(
            witness=correct_witness,
            left_polys=self.data["left_polys"],
            right_polys=self.data["right_polys"],
            out_polys=self.data["out_polys"],
            alpha_g1=self.setup_data["alpha_g1"],
            beta_g2=self.setup_data["beta_g2"],
            g1_srs=self.setup_data["g1_srs"],
            g2_srs=self.setup_data["g2_srs"],
            t_tau_srs=self.setup_data["t_tau_srs"],
            psis=self.setup_data["psis"]
        )
        
        proof1 = prover1.get_proof()
        
        # Create different setup for second proof
        setup2 = Setup(
            out_polys=self.data["out_polys"],
            left_polys=self.data["left_polys"],
            right_polys=self.data["right_polys"],
            tau=11,
            alpha=13,
            beta=17
        )
        setup2_data = setup2.get_setup()
        
        prover2 = Prover(
            witness=correct_witness,
            left_polys=self.data["left_polys"],
            right_polys=self.data["right_polys"],
            out_polys=self.data["out_polys"],
            alpha_g1=setup2_data["alpha_g1"],
            beta_g2=setup2_data["beta_g2"],
            g1_srs=setup2_data["g1_srs"],
            g2_srs=setup2_data["g2_srs"],
            t_tau_srs=setup2_data["t_tau_srs"],
            psis=setup2_data["psis"]
        )
        
        proof2 = prover2.get_proof()
        
        verifier = Verifier(
            A=proof1["A"],
            B=proof2["B"],  # From different proof
            C=proof1["C"],
            alpha_1=self.setup_data["alpha_g1"],
            beta_2=self.setup_data["beta_g2"]
        )
        
        self.assertFalse(verifier.isValid)
    

def run_tests():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestVerifier)
    
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    
    return res


if __name__ == "__main__":
    run_tests()