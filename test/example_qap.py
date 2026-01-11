import numpy as np
import galois
#from py_ecc.bn128 import curve_order
import pickle

# Creates a simple interpolated QAP with correct and incorrect withnesses for testing purposes
# Example taken from https://rareskills.io/post/r1cs-to-qap

# arithmetic circuit example:
# z=x^4-5y^2x^2$, as R1CS:
# v_1 = xx
# v_2 = v_1 * v_1$ (=> $x^4$)
# v_2 = -5yy$
# -v_2 + z = v_3 * v_1$ (=> $-5y^2*x^2$)

# --- R1CS ----

curve_order = 79

# mask matrices for output, left and right side
# for witness vector of form: 1, out, x, y, v1, v2, v3
L = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, curve_order-5, 0, 0, 0], # to get the additive inverse, negatives must be substracted from the field order
    [0, 0, 0, 0, 0, 0, 1],
])

R = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
])

O = np.array([
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, curve_order-1, 0],
])

GF = galois.GF(curve_order)

L_galois = GF(L)
R_galois = GF(R)
O_galois = GF(O)

x = GF(4)
y = GF(curve_order-2)
v1 = x * x
v2 = v1 * v1         # x^4
v3 = GF(curve_order-5)*y * y
out = v3*v1 + v2    # -5y^2 * x^2


witness = GF(np.array([1, out, x, y, v1, v2, v3]))
false_witness = GF(np.array([2,2,2,2,2,2,2]))

assert all(np.equal(
    np.matmul(L_galois, witness) * np.matmul(R_galois, witness),
    np.matmul(O_galois, witness)
)), "not equal"

assert all(np.not_equal(
    np.matmul(L_galois, false_witness) * np.matmul(R_galois, false_witness),
    np.matmul(O_galois, false_witness)
)), "equal, but should not"


# --- QAP ---
# as the r1cs above has 4 equations the x values for points
# to interpolate must be [1,2,3,4]

def interpolate_column(col):
    xs = GF(np.array([1,2,3,4]))
    return galois.lagrange_poly(xs, col)

U_polys = np.apply_along_axis(interpolate_column, 0, L_galois)
V_polys = np.apply_along_axis(interpolate_column, 0, R_galois)
W_polys = np.apply_along_axis(interpolate_column, 0, O_galois)

'''
print(f"""
    Interpolated Polynoms over the finite field: F_79:
    Left matrix:   {U_polys}
    Right matrix:  {V_polys}
    Output matrix: {W_polys}
    """
)
'''

def get_test_qap():
    return {
        "out_polys": W_polys,
        "left_polys": U_polys,
        "right_polys": V_polys,
        "correct_witness": witness,
        "false_witness": false_witness
    }

# as computation takes quite a bit of time, let's just do it once and export the results
with open("qap_data.pkl", "wb") as f:
    pickle.dump({
        "out_polys": W_polys,
        "left_polys": U_polys,
        "right_polys": V_polys,
        "correct_witness": witness,
        "false_witness": false_witness
    }, f)