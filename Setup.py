from py_ecc.bn128 import G1, G2, pairing, add, multiply, eq, curve_order, neg
import numpy as np
import random

class Setup:
    """
    Returns a random interger between zero and the G1/G2-curve order (exclusive)
    """
    def __get_random_scalar():
        return random.randint(0, curve_order - 1)

    """
    Calulates the structure reference string; powers of tau in a elliptic curve group
    Needs to be passed the max polynomial degree of the QAP,
    to calculate enough powers of tau
    """
    def __get_srs(self, poly_degree, generator):
        return [multiply(generator, self.tau**i) for i in range(poly_degree,-1,-1)]

    # for powers of tau
    tau = __get_random_scalar()

    # random scalars to be kept secret from the prover
    alpha = __get_random_scalar()
    beta = __get_random_scalar()

    # divisors for the public and the private parts of the witness
    gamma = __get_random_scalar()
    delta = __get_random_scalar()

    