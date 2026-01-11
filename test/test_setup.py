import numpy as np
import galois
from py_ecc.bn128 import curve_order
import pickle
from utils import project_path

# must be initalized before pickle data import
GF = galois.GF(curve_order)

with open(project_path("test", "qap_data.pkl"), "rb") as f:
    data = pickle.load(f)

print(data)