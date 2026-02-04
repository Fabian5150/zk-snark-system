import pickle
import galois
from py_ecc.bn128 import curve_order

from test.utils import project_path

"""
Runs the whole proving protocol with the example qap and some
command line output to follow along
"""

print("--- Importing Setup, Prover, Verifier ---")
from Setup import Setup
from Prover import Prover
from Verifier import Verifier

print(f"--- Building Galois field on {curve_order} ---")
GF = galois.GF(curve_order)

with open(project_path("test", "qap_data.pkl"), "rb") as f:
    data = pickle.load(f)

print("--- Constructing the Trusted Setup ---")

setup = Setup(
    out_polys=data["out_polys"],
    left_polys=data["left_polys"],
    right_polys=data["right_polys"],
)

print(f"Setup built with tau = {setup.tau}")

print("--- Constructing the Prover ---")

prover = Prover(
    witness=data["correct_witness"],
    left_polys=data["left_polys"],
    right_polys=data["right_polys"],
    out_polys=data["out_polys"],
    alpha_g1=setup.alpha_g1,
    beta_g2=setup.beta_g2,
    g1_srs=setup.g1_srs,
    g2_srs=setup.g2_srs,
    t_tau_srs=setup.t_tau_srs,
    psis=setup.psis,
)

print(f"Prover constructed with witness = {data["correct_witness"]} and\nA = {prover.A_1}\nB = {prover.B_2}\nC = {prover.C_1}")

print("--- Starting Verifier ---")

verifier = Verifier(
    A=prover.A_1,
    B=prover.B_2,
    C=prover.C_1,
    alpha_1=setup.alpha_g1,
    beta_2=setup.beta_g2
)

print(f"Verifier states that proof is: {verifier.isValid}")