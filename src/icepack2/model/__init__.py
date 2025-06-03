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
    h_in = kwargs.get("thickness_inflow", Constant(0.0))
    f_c = kwargs.get("frontal_ablation", Constant((0.0, 0.0)))

    cell_balance = (Dt(h) * φ - inner(h * u - f_c, grad(φ)) - a * φ) * dx

    mesh = ufl.domain.extract_unique_domain(h)
    ν = FacetNormal(mesh)
    F = max_value(0, inner(h * u - f_c, ν))

    outflow = F * φ * ds
    inflow = h_in * min_value(0, inner(u, ν)) * φ * ds
    boundary_balance = inflow + outflow

    facet_balance = jump(F) * jump(φ) * dS

    return cell_balance + facet_balance + boundary_balance
