from py_ecc.bn128 import G1, G2, pairing, add, multiply, eq, curve_order, neg
import numpy as np
import random

def get_random_scalar():
    return random.randint(0, curve_order - 1)

# for powers of tau
tau = get_random_scalar()

# random scalars to be kept secret from the prover
alpha = get_random_scalar()
beta = get_random_scalar()

# divisors for the public and the private parts of the witness
gamma = get_random_scalar()
delta = get_random_scalar()

"""
Calulates the structure reference string; powers of tau in a elliptic curve group
Needs to be passed the max polynomial degree of the QAP,
to calculate enough powers of tau
"""
def get_srs(poly_degree, generator):
    return [multiply(generator, tau**i) for i in range(poly_degree,-1,-1)]
