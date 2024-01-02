r"""Functions describing glacier mass and momentum balance"""

import ufl
import firedrake
from firedrake import (
    Constant, inner, tr, sym, grad, div, dx, ds, dS, avg, jump, max_value, min_value
)
from irksome import Dt
from .constants import ice_density as ρ_I, water_density as ρ_W, gravity as g


__all__ = [
    "viscous_power",
    "friction_power",
    "calving_terminus",
    "momentum_balance",
    "ice_shelf_momentum_balance",
    "mass_balance",
]


def viscous_power(**kwargs):
    r"""Return the viscous power dissipation"""
    # Get all the dynamical fields
    field_names = ("velocity", "membrane_stress", "thickness")
    u, M, h = map(kwargs.get, field_names)

    # Get the parameters for the constitutive relation
    parameter_names = ("flow_law_coefficient", "flow_law_exponent")
    A, n = map(kwargs.get, parameter_names)

    mesh = u.ufl_domain()
    d = mesh.geometric_dimension()

    M_2 = (inner(M, M) - tr(M) ** 2 / (d + 1)) / 2
    M_n = M_2 if float(n) == 1 else M_2 ** ((n + 1) / 2)
    return 2 * h * A / (n + 1) * M_n * dx


def calving_terminus(**kwargs):
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
    parameter_names = ("sliding_coefficient", "sliding_exponent")
    K, m = map(kwargs.get, parameter_names)
    τ_2 = inner(τ, τ)
    τ_m = τ_2 if float(m) == 1 else τ_2 ** ((m + 1) / 2)
    return K / (m + 1) * τ_m * dx


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
    ν = firedrake.FacetNormal(mesh)
    facet_balance = ρ_I * g * avg(h) * inner(jump(s, ν), avg(u)) * dS

    return cell_balance + facet_balance


def ice_shelf_momentum_balance(**kwargs):
    r"""Return the momentum balance constraint for floating ice shelves

    Floating ice shelves are simpler because there is no basal shear stress and
    we assume the ice is hydrostatic, in which case the surface elevation is
    proportional to the thickness.
    """
    field_names = ("velocity", "membrane_stress", "thickness")
    u, M, h = map(kwargs.get, field_names)
    ε = sym(grad(u))

    ρ = ρ_I * (1 - ρ_I / ρ_W)
    return (-h * inner(M, ε) + 0.5 * ρ * g * h**2 * div(u)) * dx


def mass_balance(**kwargs):
    r"""Return the mass balance equation"""
    field_names = ("thickness", "velocity", "accumulation", "test_function")
    h, u, a, φ = map(kwargs.get, field_names)
    h_inflow = kwargs.get("thickness_inflow", Constant(0.0))

    cell_balance = (Dt(h) * φ - inner(h * u, grad(φ)) - a * φ) * dx

    mesh = ufl.domain.extract_unique_domain(h)
    ν = firedrake.FacetNormal(mesh)
    f = h * max_value(0, inner(u, ν))

    outflow = f * φ * ds
    inflow = h_inflow * min_value(0, inner(u, ν)) * φ * ds
    boundary_balance = inflow + outflow

    facet_balance = jump(f) * jump(φ) * dS

    return cell_balance + facet_balance + boundary_balance
