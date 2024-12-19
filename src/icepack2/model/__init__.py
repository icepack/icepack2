import ufl
from firedrake import (
    max_value, min_value, jump, inner, grad, dx, ds, dS, FacetNormal, Constant
)
from irksome import Dt
from . import variational, minimization


def mass_balance(**kwargs):
    r"""Return the mass balance equation"""
    field_names = ("thickness", "velocity", "accumulation", "test_function")
    h, u, a, φ = map(kwargs.get, field_names)
    h_inflow = kwargs.get("thickness_inflow", Constant(0.0))

    cell_balance = (Dt(h) * φ - inner(h * u, grad(φ)) - a * φ) * dx

    mesh = ufl.domain.extract_unique_domain(h)
    ν = FacetNormal(mesh)
    f = h * max_value(0, inner(u, ν))

    outflow = f * φ * ds
    inflow = h_inflow * min_value(0, inner(u, ν)) * φ * ds
    boundary_balance = inflow + outflow

    facet_balance = jump(f) * jump(φ) * dS

    return cell_balance + facet_balance + boundary_balance
