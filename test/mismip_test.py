import firedrake
from firedrake import (
    sqrt, exp, max_value, inner, as_vector, Constant, dx
)
from icepack2.model import mass_Balance
from icepack2.model.variational import (
    momentum_balance, flow_law, friction_law, calving_terminus
)
from icepack2.constants import (
    gravity, ice_density, glen_flow_law, weertman_sliding_law
)


lx, ly = 640e3, 80e3


def mismip_bed_topography(x):
    x_c = Constant(300e3)
    X = x[0] / x_c

    B_0 = Constant(-150)
    B_2 = Constant(-728.8)
    B_4 = Constant(343.91)
    B_6 = Constant(-50.57)

    B_x = B_0 + B_2 * X**2 + B_4 * X**4 + B_6 * X**6

    f_c = Constant(4e3)
    d_c = Constant(500)
    w_c = Constant(24e3)
    Ly = Constant(ly)

    B_y = d_c * (
        1 / (1 + exp(-2 * (x[1] - Ly / 2 - w_c) / f_c)) +
        1 / (1 + exp(+2 * (x[1] - Ly / 2 + w_c) / f_c))
    )

    z_deep = Constant(-720)
    return max_value(B_x + B_y, z_deep)


def run_simulation(ny: int):
    nx = int(lx / ly) * ny
    mesh = firedrake.RectangleMesh(nx, ny, lx, ly, diagonal="crossed")

    # Make a bunch of finite elements and function spaces
    cg1 = firedrake.FiniteElement("CG", "triangle", 1)
    dg1 = firedrake.FiniteElement("DG", "triangle", 1)
    dg0 = firedrake.FiniteElement("DG", "triangle", 0)
    S = firedrake.FunctionSpace(mesh, cg1)
    Q = firedrake.FunctionSpace(mesh, dg1)
    V = firedrake.VectorFunctionSpace(mesh, cg1)
    Σ = firedrake.TensorFunctionSpace(mesh, dg0, symmetry=True)
    Z = V * Σ * V * Q

    # Make the bed topography
    x = firedrake.SpatialCoordinate(mesh)
    b = firedrake.Function(S).interpolate(mismip_bed_topography(x))

    # The spin-up phase of the experiment specifies a mass balance of 30 cm/yr
    a = Constant(0.3)

    # Make the initial thickness
    h_0 = firedrake.Function(Q).assign(Constant(100.0))

    # Fluidity of ice in yr⁻¹ MPa⁻³
    A = Constant(20.0)

    z = firedrake.Function(Z)
    u, M, τ, h = firedrake.split(z)
