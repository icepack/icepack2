import ufl
from firedrake import (
    eq,
    conditional,
    Constant,
    inner,
    tr,
    sym,
    grad,
    div,
    dx,
    ds,
    dS,
    avg,
    jump,
    FacetNormal,
    min_value,
)
from . import utilities
from ..constants import ice_density as ρ_I, water_density as ρ_W, gravity as g


def flow_law(**kwargs):
    r"""Return the symbolic form of the flow law, i.e. the relation between the
    strain rate and membrane stress tensors"""
    field_names = ("thickness", "membrane_stress", "velocity")
    h, M, u = map(kwargs.get, field_names)
    N = utilities.get_test_function(M)
    A, n = map(kwargs.get, ("flow_law_coefficient", "flow_law_exponent"))

    mesh = ufl.domain.extract_unique_domain(u)
    d = mesh.geometric_dimension()

    ε = sym(grad(u))
    M_2 = (inner(M, M) - tr(M) ** 2 / (d + 1)) / 2
    M_n = conditional(eq(n, 1), Constant(1.0), M_2 ** ((n - 1) / 2))
    return h * (A * M_n * (inner(M, N) - tr(M) * tr(N) / (d + 1)) - inner(ε, N)) * dx


def friction_law(**kwargs):
    r"""Return the symbolic form of the sliding law, i.e. the relation between
    the basal sliding velocity and the basal drag vectors"""
    τ, u = map(kwargs.get, ("basal_stress", "velocity"))
    σ = utilities.get_test_function(τ)
    K, m = map(kwargs.get, ("sliding_coefficient", "sliding_exponent"))
    τ_2 = inner(τ, τ)
    τ_m = conditional(eq(m, 1), Constant(1.0), τ_2 ** ((m - 1) / 2))
    return inner(K * τ_m * τ + u, σ) * dx


def calving_terminus(**kwargs):
    r"""Return the symbolic form of the pressure exerted at the terminus of a
    glacier that flows into a water body"""
    h, s, u = map(kwargs.get, ("thickness", "surface", "velocity"))
    v = utilities.get_test_function(u)
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
    r"""Return the symbolic form of the constraint of momentum balance"""
    field_names = (
        "membrane_stress",
        "basal_stress",
        "thickness",
        "surface",
        "velocity",
    )
    M, τ, h, s, u = map(kwargs.get, field_names)
    v = utilities.get_test_function(u)

    ε = sym(grad(v))
    cell_balance = (-h * inner(M, ε) + inner(τ - ρ_I * g * h * grad(s), v)) * dx

    mesh = ufl.domain.extract_unique_domain(v)
    ν = FacetNormal(mesh)
    facet_balance = ρ_I * g * avg(h) * inner(jump(s, ν), avg(v)) * dS

    return cell_balance + facet_balance


def ice_shelf_momentum_balance(**kwargs):
    r"""Return the symbolic form of the constraint of momentum balance for the
    special case of floating ice shelves in hydrostatic balance

    Floating ice shelves are simpler because there is no basal shear stress
    and we assume the ice is hydrostatic, in which case the surface
    elevation is proportional to the thickness."""
    field_names = ("membrane_stress", "thickness", "velocity")
    M, h, u = map(kwargs.get, field_names)
    v = utilities.get_test_function(u)
    ε = sym(grad(v))

    ρ = ρ_I * (1 - ρ_I / ρ_W)
    return (-h * inner(M, ε) + 0.5 * ρ * g * h**2 * div(v)) * dx
