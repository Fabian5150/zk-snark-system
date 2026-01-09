import numpy as np
import galois

# arithmetic circuit example:
# z=x^4-5y^2x^2$, as R1CS:
# v_1 = xx
# v_2 = v_1 * v_1$ (=> $x^4$)
# v_2 = -5yy$
# -v_2 + z = v_3 * v_1$ (=> $-5y^2*x^2$)

# --- R1CS ----

# mask matrices for output, left and right side
# for witness vector of form: 1, out, x, y, v1, v2, v3
L = np.array([
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, -5, 0, 0, 0],
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
    [0, 1, 0, 0, 0, -1, 0],
])

# valid example witness:
x = 4
y = -2
v1 = x * x
v2 = v1 * v1
v3 = -5 * y * y
z = v3 * v1 + v2

a = np.array([1, z, x, y, v1, v2, v3])

assert all(np.equal(
    (np.matmul(L, a) * np.matmul(R, a)),
    np.matmul(O, a)
))

# --- QAP ---