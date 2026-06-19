import ufl
from firedrake import (
    max_value, min_value, jump, inner, dx, ds, dS, ds_v, dS_v, FacetNormal, Constant
)
from irksome import Dt
from . import variational, minimization, utilities
from icepack.calculus import grad, get_mesh_axes


def mass_balance(**kwargs):
    r"""Return the mass balance equation"""
    field_names = ("thickness", "velocity", "accumulation")
    h, u, a = map(kwargs.get, field_names)
    φ = utilities.get_test_function(h)
    h_inflow = kwargs.get("thickness_inflow", Constant(0.0))

    cell_balance = (Dt(h) * φ - inner(h * u, grad(φ)) - a * φ) * dx

    mesh = ufl.domain.extract_unique_domain(h)
    axes = get_mesh_axes(mesh)
    ν = FacetNormal(mesh)

    if axes in ["xy", "x"]:
        f = h * max_value(0, inner(u, ν))
        outflow = f * φ * ds
        inflow = h_inflow * min_value(0, inner(u, ν)) * φ * ds
        boundary_balance = inflow + outflow
        facet_balance = jump(f) * jump(φ) * dS
    elif axes in ["xz"]:
        f = h * max_value(0, u * ν[0])
        outflow = f * φ * ds_v
        inflow = h_inflow * min_value(0, u * ν[0]) * φ * ds_v
        boundary_balance = inflow + outflow
        facet_balance = jump(f) * jump(φ) * dS_v
    else:
        f = h * max_value(0, u[0] * ν[0] + u[1] * ν[1])
        outflow = f * φ * ds_v
        inflow = h_inflow * min_value(0, u[0] * ν[0] + u[1] * ν[1]) * φ * ds_v
        boundary_balance = inflow + outflow
        facet_balance = jump(f) * jump(φ) * dS_v

    return cell_balance + facet_balance + boundary_balance
