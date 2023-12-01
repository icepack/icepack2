import numpy as np
import firedrake
from firedrake import interpolate, as_vector, max_value, Constant
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    glen_flow_law as n,
    weertman_sliding_law as m,
)
from icepack2 import model


Lx, Ly = Constant(20e3), Constant(20e3)
h0, dh = Constant(500.0), Constant(100.0)
u_inflow = Constant(100.0)

τ_c = Constant(0.1)
ε_c = Constant(0.01)
A = ε_c / τ_c**n

height_above_flotation = Constant(10.0)
d = Constant(-ρ_I / ρ_W * (h0 - dh) + height_above_flotation)
ρ = Constant(ρ_I - ρ_W * d**2 / (h0 - dh) ** 2)

# We'll arbitrarily pick this to be the velocity, then we'll find a
# friction coefficient and surface elevation that makes this velocity
# an exact solution of the shelfy stream equations.
def exact_u(x):
    Z = A * (ρ * g * h0 / 4) ** n
    q = 1 - (1 - (dh / h0) * (x / Lx)) ** (n + 1)
    du = Z * q * Lx * (h0 / dh) / (n + 1)
    return u_inflow + du


# With this choice of friction coefficient, we can take the surface
# elevation to be a linear function of the horizontal coordinate and the
# velocity will be an exact solution of the shelfy stream equations.
β = Constant(0.5)
α = Constant(β * ρ / ρ_I * dh / Lx)

def friction(x):
    return α * (ρ_I * g * (h0 - dh * x / Lx)) * exact_u(x) ** (-1 / m)


for degree in [1, 2]:
    errors, mesh_sizes = [], []
    k_min, k_max, num_steps = 5 - degree, 8 - degree, 9
    for nx in np.logspace(k_min, k_max, num_steps, base=2, dtype=int):
        mesh = firedrake.RectangleMesh(nx, nx, float(Lx), float(Ly), diagonal="crossed")
        x, y = firedrake.SpatialCoordinate(mesh)

        cg = firedrake.FiniteElement("CG", "triangle", degree)
        dg = firedrake.FiniteElement("DG", "triangle", degree - 1)
        Q = firedrake.FunctionSpace(mesh, cg)
        V = firedrake.VectorFunctionSpace(mesh, cg)
        Σ = firedrake.TensorFunctionSpace(mesh, dg, symmetry=True)
        # TODO: investigate using DG for this?
        T = firedrake.VectorFunctionSpace(mesh, cg)
        Z = V * Σ * T
        z = firedrake.Function(Z)
        z.sub(0).assign(Constant((u_inflow, 0)))

        u_exact = interpolate(as_vector((exact_u(x), 0)), V)

        h = interpolate(h0 - dh * x / Lx, Q)
        ds = (1 + β) * ρ / ρ_I * dh
        s = interpolate(d + h0 - dh + ds * (1 - x / Lx), Q)

        # TODO: adjust the yield stress so that this has a more sensible value
        C = interpolate(friction(x), Q)
        u_c = interpolate((τ_c / C)**m, Q)

        inflow_ids = (1,)
        outflow_ids = (2,)
        side_wall_ids = (3, 4)

        u, M, τ = firedrake.split(z)
        kwargs = {
            "velocity": u,
            "membrane_stress": M,
            "basal_stress": τ,
            "thickness": h,
            "surface": s,
            "outflow_ids": outflow_ids,
        }

        inflow_bc = firedrake.DirichletBC(Z.sub(0), Constant((u_inflow, 0)), inflow_ids)
        side_wall_bc = firedrake.DirichletBC(Z.sub(0).sub(1), 0, side_wall_ids)
        bcs = [inflow_bc, side_wall_bc]

        fns = [
            model.viscous_power,
            model.friction_power,
            model.calving_terminus,
            model.momentum_balance,
        ]

        lkwargs = {
            "flow_law_exponent": 1,
            "flow_law_coefficient": ε_c / τ_c,
            "sliding_exponent": 1,
            "sliding_coefficient": u_c / τ_c,
        }
        J_l = sum(fn(**kwargs, **lkwargs) for fn in fns)
        F_l = firedrake.derivative(J_l, z)
        firedrake.solve(F_l == 0, z, bcs=bcs)

        qdegree = max(8, degree ** n)
        fcparams = {
            "form_compiler_parameters": {"quadrature_degree": qdegree}
        }

        num_continuation_steps = 4
        for t in np.linspace(0.0, 1.0, num_continuation_steps):
            n_t = Constant((1 - t) + t * n)
            m_t = Constant((1 - t) + t * m)
            params = {
                "flow_law_exponent": n_t,
                "flow_law_coefficient": ε_c / τ_c ** n_t,
                "sliding_exponent": m_t,
                "sliding_coefficient": u_c / τ_c ** m_t,
            }
            J = sum(fn(**kwargs, **params) for fn in fns)
            F = firedrake.derivative(J, z)
            firedrake.solve(F == 0, z, bcs=bcs, **fcparams)

        u, M, τ = z.subfunctions
        error = firedrake.norm(u - u_exact) / firedrake.norm(u_exact)
        δx = mesh.cell_sizes.dat.data_ro.min()
        mesh_sizes.append(δx)
        errors.append(error)
        print(".", end="", flush=True)

    log_mesh_sizes = np.log2(np.array(mesh_sizes))
    log_errors = np.log2(np.array(errors))
    slope, intercept = np.polyfit(log_mesh_sizes, log_errors, 1)
    print(f"degree {degree}: log(error) ~= {slope:g} * log(dx) {intercept:+g}")
    assert slope > degree + 0.9
