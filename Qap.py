#Inputs L, R, O, a, curve_order/prime_p

from py_ecc.bn128 import G1, G2, curve_order
import numpy as np
import galois
import functools as ft

"""
For transforming a R1CS (given as its three matrices) into a QAP
"""
class QAP:

    def __init__(self, L, R, O, witness, c_order):
        self.curve_order = c_order
        self.L = (L + self.curve_order) % self.curve_order
        self.R = (R + self.curve_order) % self.curve_order
        self.O = (O + self.curve_order) % self.curve_order
        self.witness = witness

        self.GF = galois.GF(self.curve_order)


        self.L_galois = self.GF(L)
        self.R_galois = self.GF(R)
        self.O_galois = self.GF(O)

        self.U_polys = np.apply_along_axis(self.__interpolate_column, 0, self.L_galois)
        self.V_polys = np.apply_along_axis(self.__interpolate_column, 0, self.R_galois)
        self.W_polys = np.apply_along_axis(self.__interpolate_column, 0, self.O_galois)

        self.qap_1 = self.__inner_product_polynomials_with_witness(self.U_polys, witness)
        self.qap_2 = self.__inner_product_polynomials_with_witness(self.V_polys, witness)
        self.qap_3 = self.__inner_product_polynomials_with_witness(self.W_polys, witness)

        self.t = self.__get_t()
        self.h = (self.qap_1 * self.qap_2 - self.qap_3) // self.t

        self.__print_qap_check()
    
    def __interpolate_column(self, col):
        xs = self.GF(np.arange(1, self.L.shape[0]))
        return galois.lagrange_poly(xs, col)

    def __get_t(self):
        t_rows = self.GF(np.arange(1, self.L.shape[0]))
        t = 1
        for i in t_rows:
            t *= galois.Poly([1, self.curve_order - i], field=self.GF)
        return t
    
    def __print_qap_check(self):
        print("QAP formula true?", self.qap_1 * self.qap_2 == self.qap_3 + self.h * self.t)

    def __inner_product_polynomials_with_witness(self, polys, witness):
        mul_ = lambda x, y: x * y
        sum_ = lambda x, y: x + y
        return ft.reduce(sum_, map(mul_, polys, witness))