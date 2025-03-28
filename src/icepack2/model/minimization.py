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
from ..constants import ice_density as ρ_I, water_density as ρ_W, gravity as g


def viscous_power(**kwargs):
    r"""Return the viscous power dissipation"""
    # Get all the dynamical fields
    field_names = ("membrane_stress", "thickness")
    M, h = map(kwargs.get, field_names)

    # Get the parameters for the constitutive relation
    parameter_names = ("flow_law_coefficient", "flow_law_exponent")
    A, n = map(kwargs.get, parameter_names)

    mesh = ufl.domain.extract_unique_domain(M)
    d = mesh.geometric_dimension()

    M_2 = (inner(M, M) - tr(M) ** 2 / (d + 1)) / 2
    M_n = conditional(eq(n, 1), M_2, M_2 ** ((n + 1) / 2))
    return 2 * h * A / (n + 1) * M_n * dx


def friction_power(**kwargs):
    r"""Return the frictional power dissipation"""
    τ = kwargs["basal_stress"]
    parameter_names = ("sliding_coefficient", "sliding_exponent")
    K, m = map(kwargs.get, parameter_names)
    τ_2 = inner(τ, τ)
    τ_m = conditional(eq(m, 1), τ_2, τ_2 ** ((m + 1) / 2))
    return K / (m + 1) * τ_m * dx


def calving_terminus(**kwargs):
    r"""Return the power dissipation from the terminus boundary condition"""
    # Get all the dynamical fields and boundary conditions
    u, h, s = map(kwargs.get, ("velocity", "thickness", "surface"))
    outflow_ids = kwargs["outflow_ids"]

    # Get the unit outward normal vector to the terminus
    mesh = ufl.domain.extract_unique_domain(u)
    ν = FacetNormal(mesh)

    # Compute the forces per unit length at the terminus from the glacier
    # and from the ocean (assuming that sea level is at z = 0)
    f_I = 0.5 * ρ_I * g * h**2
    d = min_value(0, s - h)
    f_W = 0.5 * ρ_W * g * d**2

    return (f_I - f_W) * inner(u, ν) * ds(outflow_ids)


def momentum_balance(**kwargs):
    r"""Return the momentum balance constraint"""
    field_names = (
        "velocity", "membrane_stress", "basal_stress", "thickness", "surface"
    )
    u, M, τ, h, s = map(kwargs.get, field_names)
    f = kwargs.get("floating", Constant(1.0))
    ε = sym(grad(u))
    cell_balance = (-h * inner(M, ε) + inner(f * τ - ρ_I * g * h * grad(s), u)) * dx

    mesh = ufl.domain.extract_unique_domain(u)
    ν = FacetNormal(mesh)
    facet_balance = ρ_I * g * avg(h) * inner(jump(s, ν), avg(u)) * dS

    return cell_balance + facet_balance


def ice_shelf_momentum_balance(**kwargs):
    r"""Return the momentum balance constraint for floating ice shelves

    Floating ice shelves are simpler because there is no basal shear stress
    and we assume the ice is hydrostatic, in which case the surface
    elevation is proportional to the thickness.
    """
    field_names = ("velocity", "membrane_stress", "thickness")
    u, M, h = map(kwargs.get, field_names)
    ε = sym(grad(u))

    ρ = ρ_I * (1 - ρ_I / ρ_W)
    return (-h * inner(M, ε) + 0.5 * ρ * g * h**2 * div(u)) * dx
