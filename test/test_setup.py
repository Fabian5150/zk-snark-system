import numpy as np
import galois
import pickle

curve_order = 79

# must be initalized before pickle data import
GF = galois.GF(curve_order)

with open("test/qap_data.pkl", "rb") as f:
    data = pickle.load(f)

print(data)