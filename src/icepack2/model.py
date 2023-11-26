r"""Functions to calculate the Lagrangian for glacier momentum balance"""

import firedrake
from firedrake import Constant, inner, tr, sym, grad, dx, ds
import icepack


__all__ = ["viscous_power", "friction_power", "boundary", "constraint"]


# Physical constants
ρ_I = Constant(icepack.constants.ice_density)
ρ_W = Constant(icepack.constants.water_density)
g = Constant(icepack.constants.gravity)


def viscous_power(**kwargs):
    r"""Return the viscous power dissipation"""
    # Get all the dynamical fields
    field_names = ("velocity", "membrane_stress", "thickness")
    u, M, h = map(kwargs.get, field_names)

    # Get the parameters for the constitutive relation
    parameter_names = (
        "viscous_yield_strain", "viscous_yield_stress", "flow_law_exponent"
    )
    ε, τ, n = map(kwargs.get, parameter_names)
    A = ε / τ**n

    mesh = u.ufl_domain()
    d = mesh.geometric_dimension()

    M_2 = (inner(M, M) - tr(M) ** 2 / (d + 1)) / 2
    M_n = M_2 if float(n) == 1 else M_2 ** ((n + 1) / 2)
    return 2 * h * A / (n + 1) * M_n * dx


def boundary(**kwargs):
    r"""Return the power dissipation from the terminus boundary condition"""
    # Get all the dynamical fields and boundary conditions
    u, h, s = map(kwargs.get, ("velocity", "thickness", "surface"))
    outflow_ids = kwargs["outflow_ids"]

    # Get the unit outward normal vector to the terminus
    mesh = u.ufl_domain()
    ν = firedrake.FacetNormal(mesh)

    # Compute the forces per unit length at the terminus from the glacier
    # and from the ocean (assuming that sea level is at z = 0)
    f_I = 0.5 * ρ_I * g * h**2
    d = firedrake.min_value(0, s - h)
    f_W = 0.5 * ρ_W * g * d**2

    return (f_I - f_W) * inner(u, ν) * ds(outflow_ids)


def friction_power(**kwargs):
    r"""Return the frictional power dissipation"""
    τ = kwargs["basal_stress"]
    parameter_names = (
        "friction_yield_speed", "friction_yield_stress", "sliding_law_exponent"
    )
    u_c, τ_c, m = map(kwargs.get, parameter_names)
    K = u_c / τ_c ** m
    τ_2 = inner(τ, τ)
    τ_m = τ_2 if float(m) == 1 else τ_2 ** ((m + 1) / 2)
    return K / (m + 1) * τ_m * dx


def constraint(**kwargs):
    r"""Return the momentum balance constraint"""
    field_names = (
        "velocity", "membrane_stress", "basal_stress", "thickness", "surface"
    )
    u, M, τ, h, s = map(kwargs.get, field_names)
    f = kwargs.get("floating", Constant(1.0))
    ε = sym(grad(u))
    return (-h * inner(M, ε) + inner(f * τ - ρ_I * g * h * grad(s), u)) * dx