import numpy as np
import galois
from py_ecc.bn128 import curve_order
import pickle
from utils import project_path


# must be initalized before pickle data import
GF = galois.GF(curve_order)

with open(project_path("test", "qap_data.pkl"), "rb") as f:
    data = pickle.load(f)

def main():
    qap = data

    setup = Setup(
        out_polys=qap["out_polys"],
        left_polys=qap["left_polys"],
        right_polys=qap["right_polys"]
    )

    data = setup.get_setup()

    print("Trusted setup completed")
    print(f"Curve order: {curve_order}")
    print(f"Random tau: {setup.tau}")
    print(f"Random alpha: {setup.alpha}")
    print(f"Random beta: {setup.beta}")

    print("\nSRS G1 length:", len(data["g1_srs"]))
    print("SRS G2 length:", len(data["g2_srs"]))
    print("t(tau) SRS length:", len(data["t_tau_srs"]))
    print("Psi points:", len(data["psis"]))

if __name__ == "__main__":
    main()