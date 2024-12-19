import ufl
from firedrake import (
    Constant, inner, tr, sym, grad, div, dx, ds, dS, avg, jump, FacetNormal, min_value
)
from ..constants import ice_density as ρ_I, water_density as ρ_W, gravity as g


def flow_law(**kwargs):
    field_names = ("thickness", "membrane_stress", "velocity", "test_function")
    h, M, u, N = map(kwargs.get, field_names)
    A, n = map(kwargs.get, ("flow_law_coefficient", "flow_law_exponent"))

    mesh = ufl.domain.extract_unique_domain(u)
    d = mesh.geometric_dimension()

    ε = sym(grad(u))
    M_2 = (inner(M, M) - tr(M) ** 2 / (d + 1)) / 2
    M_n = Constant(1.0) if float(n) == 1 else M_2 ** ((n - 1) / 2)
    return h * (A * M_n * (inner(M, N) - tr(M) * tr(N) / (d + 1)) - inner(ε, N)) * dx


def friction_law(**kwargs):
    τ, u, σ = map(kwargs.get, ("basal_stress", "velocity", "test_function"))
    K, m = map(kwargs.get, ("sliding_coefficient", "sliding_exponent"))
    τ_2 = inner(τ, τ)
    τ_m = Constant(1.0) if float(m) == 1 else τ_2 ** ((m - 1) / 2)
    return inner(K * τ_m * τ + u, σ) * dx


def calving_terminus(**kwargs):
    h, s, v = map(kwargs.get, ("thickness", "surface", "test_function"))
    outflow_ids = kwargs["outflow_ids"]

    mesh = ufl.domain.extract_unique_domain(v)
    ν = FacetNormal(mesh)

    # Compute the forces per unit length at the terminus from the glacier
    # and from the ocean (assuming that sea level is at z = 0)
    f_I = 0.5 * ρ_I * g * h**2
    d = min_value(0, s - h)
    f_W = 0.5 * ρ_W * g * d**2

    return (f_I - f_W) * inner(v, ν) * ds(outflow_ids)


def momentum_balance(**kwargs):
    field_names = (
        "velocity",
        "membrane_stress",
        "basal_stress",
        "thickness",
        "surface",
        "test_function",
    )
    u, M, τ, h, s, v = map(kwargs.get, field_names)

    ε = sym(grad(v))
    cell_balance = (-h * inner(M, ε) + inner(τ - ρ_I * g * h * grad(s), v)) * dx

    mesh = ufl.domain.extract_unique_domain(u)
    ν = FacetNormal(mesh)
    facet_balance = ρ_I * g * avg(h) * inner(jump(s, ν), avg(v)) * dS

    return cell_balance + facet_balance


