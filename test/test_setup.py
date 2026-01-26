import numpy as np
import galois
from py_ecc.bn128 import curve_order
import pickle
from utils import project_path

# for class import from parent dir (god forbid I could just use a file path for imports...)
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)


from Setup import Setup

# must be initalized before pickle data import
GF = galois.GF(curve_order)

with open(project_path("test", "qap_data.pkl"), "rb") as f:
    data = pickle.load(f)

setup = Setup(
    out_polys=data["out_polys"],
    left_polys=data["left_polys"],
    right_polys=data["right_polys"]
)

trusted_setup = setup.get_setup()

print("Setup Output erhalten")

print(f"Setup erfolgreich erstellt!")
print(f"Anzahl Polynome: {setup.num_polys}")
print(f"Anzahl Constraints: {setup.num_constraints}")
print(f"Länge G1 SRS: {len(trusted_setup['g1_srs'])}")
print(f"Länge G2 SRS: {len(trusted_setup['g2_srs'])}")
print(f"Länge t(τ) SRS: {len(trusted_setup['t_tau_srs'])}")
print(f"Anzahl Ψ Elemente: {len(trusted_setup['psis'])}")