import ufl
from operator import itemgetter
from firedrake import (
    eq,
    conditional,
    Constant,
    inner,
    sqrt,
    tr,
    dx,
    ds_b,
    ds_v,
    dS,
    dS_v,
    avg,
    jump,
    FacetNormal,
    Identity,
    SpatialCoordinate,
    outer,
    min_value,
)
from ..constants import ice_density as ρ_I, water_density as ρ_W, gravity as g
from icepack.utilities import add_kwarg_wrapper
from icepack.calculus import grad, sym_grad, get_mesh_axes


def horizontal_strain_rate(**kwargs):
    r"""Calculate the horizontal strain rate with corrections for terrain-
    following coordinates"""
    u, h, s = itemgetter("velocity", "surface", "thickness")(kwargs)
    mesh = h.function_space().mesh()
    dim = mesh.geometric_dimension()
    ζ = SpatialCoordinate(mesh)[dim - 1]
    b = s - h
    v = -((1 - ζ) * grad(b) + ζ * grad(s)) / h
    du_dζ = u.dx(dim - 1)
    return sym_grad(u) + 0.5 * (outer(du_dζ, v) + outer(v, du_dζ))

def vertical_strain_rate(**kwargs):
    r"""Calculate the vertical strain rate with corrections for terrain-
    following coordinates"""
    u, h, s = itemgetter("velocity", "thickness", "surface")(kwargs)
    # Get the mesh from the surface elevation because sometimes h is a diff
    mesh = s.function_space().mesh()
    du_dζ = u.dx(mesh.geometric_dimension() - 1)
    return 0.5 * du_dζ / h


def C_operator_1storder(tensor, vector): 
    r"""Apply the script C operator to a tensor for Blatter-Pattyn approximation.
    
    The tensor shoud be 2x3."""
    return 1. / 2. * (tensor + tr(tensor) * Identity(tensor.geometric_dimension() - 1)), 1. / 2. * vector


def C_norm_1storder(tensor, vector):
    r"""Compute the C-norm of a tensor for Blatter-Pattyn approximation.
    
    The tensor shoud be 2x3."""
    t, v = C_operator_1storder(tensor, vector)
    return sqrt(inner(tensor, t) + inner(vector, v))


def viscous_power(**kwargs):
    r"""Return the symbolic form of the viscous power dissipation rate"""
    # Get all the dynamical fields
    field_names = ("membrane_stress_x", "membrane_stress_z", "thickness")
    Mx, Mz, h = map(kwargs.get, field_names)

    # Get the parameters for the constitutive relation
    parameter_names = ("flow_law_coefficient", "flow_law_exponent")
    A, n = map(kwargs.get, parameter_names)

    mesh = ufl.domain.extract_unique_domain(h)
    axes = get_mesh_axes(mesh)
    d = mesh.geometric_dimension()

    M_2 = (inner(Mx, Mx) + inner(Mz, Mz) - tr(Mx) ** 2 / d) / 2
    M_n = conditional(eq(n, 1), M_2, M_2 ** ((n + 1) / 2))
    return 2 * h * A / (n + 1) * M_n * dx


def bed_friction_power(**kwargs):
    r"""Return the symbolic form of the frictional power dissipation rate"""
    τ = kwargs["basal_stress"]
    parameter_names = ("sliding_coefficient", "sliding_exponent", "velocity")
    K, m, u = map(kwargs.get, parameter_names)
    τ_2 = inner(τ, τ)
    τ_m = conditional(eq(m, 1), τ_2, τ_2 ** ((m + 1) / 2))
    return (K / (m + 1) * τ_m + inner(τ, u)) * ds_b


def momentum_balance(**kwargs):
    r"""Return the symbolic form of the momentum balance constraint"""
    field_names = (
        "velocity", "membrane_stress_x", "membrane_stress_z", "thickness", "surface"
    )
    u, Mx, Mz, h, s = map(kwargs.get, field_names)
    ε_x = horizontal_strain_rate(velocity=u, thickness=h, surface=s)
    ε_z = vertical_strain_rate(velocity=u, thickness=h, surface=s)
    f = kwargs.get("floating", Constant(1.0))
    cell_balance = (-h * (inner(Mx, ε_x) + inner(Mz, ε_z)) - inner(ρ_I * g * h * grad(s), u)) * dx

    mesh = ufl.domain.extract_unique_domain(u)
    ν = FacetNormal(mesh)
    axes = get_mesh_axes(mesh)
    if axes in ["xy", "x"]:
        facet_balance = ρ_I * g * avg(h) * inner(jump(s, ν), avg(u)) * dS
    else:
        facet_balance = ρ_I * g * avg(h) * inner(jump(s, ν)[0], avg(u)[0]) * dS_v
        for dim in range(1, mesh.geometric_dimension() - 1):
            facet_balance += ρ_I * g * avg(h) * inner(jump(s, ν)[dim], avg(u)[dim]) * dS_v

    return cell_balance - facet_balance


def calving_terminus(**kwargs):
    r"""Return the symbolic form of the power dissipation at the terminus
    of a glacier that flows into a water body"""
    # Get all the dynamical fields and boundary conditions
    u, h, s = map(kwargs.get, ("velocity", "thickness", "surface"))
    outflow_ids = kwargs["outflow_ids"]

    # Get the unit outward normal vector to the terminus
    mesh = ufl.domain.extract_unique_domain(s)
    ν = FacetNormal(mesh)

    # Compute the forces per unit length at the terminus from the glacier
    # and from the ocean (assuming that sea level is at z = 0)
    f_I = 0.5 * ρ_I * g * h**2
    d = min_value(0, s - h)
    f_W = 0.5 * ρ_W * g * d**2

    return (f_I - f_W) * sum([inner(u[i], ν[i]) for i in range(mesh.geometric_dimension() - 1)]) * ds_v(outflow_ids)


class HybridModel:
    r"""Class for the dual form of the Blatter-Pattyn approximation.
    """
    def __init__(
        self,
        viscous_power=viscous_power,
        bed_friction_power=bed_friction_power,
        momentum_balance=momentum_balance,
        calving_terminus=calving_terminus,
    ):
        self.viscous_power = add_kwarg_wrapper(viscous_power)
        self.momentum_balance = add_kwarg_wrapper(momentum_balance)

        self.terms = [self.viscous_power, self.momentum_balance]
        if calving_terminus is not None:
            self.calving_terminus = add_kwarg_wrapper(calving_terminus)
            self.terms.append(self.calving_terminus)

        if bed_friction_power is not None:
            self.bed_friction_power = add_kwarg_wrapper(bed_friction_power)
            self.terms.append(self.bed_friction_power)

    def action(self, **kwargs):
        r"""Return the dual form of the Blatter-Pattyn approximation"""
        return sum(term(**kwargs) for term in self.terms)