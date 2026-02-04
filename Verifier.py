from py_ecc.bn128 import pairing, G2

"""
Takes as input the A, B and C points (aka the proof) from the prover
and the alpha and beta curve points from the setup
"""
class Verifier:
    def __init__(self,
        A,
        B,
        C,
        alpha_1,
        beta_2
    ):
        self.A = A
        self.B = B
        self.C = C
        self.alpha_1 = alpha_1
        self.beta_2 = beta_2

        self.isValid = self.verify()

    """
    Builds both sides of the verifing equation and checks if they're equal
    """
    def verify(self):
        left = pairing(self.B, self.A)

        right_1 = pairing(self.beta_2, self.alpha_1)
        right_2 = pairing(G2, self.C)

        return left == right_1 * right_2