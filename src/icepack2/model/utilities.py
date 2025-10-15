import ufl
import firedrake


def get_test_function(u):
    z, = ufl.algorithms.extract_coefficients(u)
    Z = z.function_space()
    w = firedrake.TestFunction(Z)
    return firedrake.replace(u, {z: w})
