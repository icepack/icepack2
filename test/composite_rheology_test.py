import numpy as np
import firedrake
from firedrake import (
    as_vector,
    max_value,
    Constant,
    Function,
    derivative,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
)
from icepack2.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    glen_flow_law,
    weertman_sliding_law,
)
from icepack2.model.minimization import viscous_power, ice_shelf_momentum_balance


sparams = {
    "solver_parameters": {
        "snes_type": "newtonls",
        "snes_max_it": 200,
        "snes_linesearch_type": "nleqerr",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}


def test_composite_rheology_floating():
    Lx, Ly = Constant(20e3), Constant(20e3)
    h0, dh = Constant(500.0), Constant(100.0)
    u_inflow = Constant(100.0)

    n1 = firedrake.Constant(1.0)
    n3 = firedrake.Constant(glen_flow_law)

    τ_c = Constant(0.1)
    ε_c = Constant(0.01)
    A1 = ε_c / τ_c ** n1
    A3 = ε_c / τ_c ** n3

    # This is an exact solution for the velocity of a floating ice shelf with
    # constant temperature and linearly decreasing thickness. See Greve and
    # Blatter for the derivation.
    def exact_δu(x, n, A):
        ρ = ρ_I * (1 - ρ_I / ρ_W)
        h = h0 - dh * x / Lx
        P = ρ * g * h / 4
        P_0 = ρ * g * h0 / 4
        δP = ρ * g * dh / 4
        return Lx * A * (P_0 ** (n + 1) - P ** (n + 1)) / ((n + 1) * δP)

    # We'll use the same perturbation to `u` throughout these tests.
    def perturb_u(x, y):
        px, py = x / Lx, y / Ly
        q = 16 * px * (1 - px) * py * (1 - py)
        return 60 * q * (px - 0.5)

    degree = 1
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
        Z = V * Σ
        z = Function(Z)
        z.sub(0).assign(Constant(u_inflow, 0))

        expr = u_inflow + exact_δu(x, n1, A1) + exact_δu(x, n3, A3)
        u_exact = Function(V).interpolate(as_vector((expr, 0)))

        h = Function(Q).interpolate(h0 - dh * x / Lx)
        inflow_ids = (1,)
        outflow_ids = (2,)
        side_wall_ids = (3, 4)

        inflow_bc = firedrake.DirichletBC(Z.sub(0), Constant((u_inflow, 0)), inflow_ids)
        side_wall_bc = firedrake.DirichletBC(Z.sub(0).sub(1), 0, side_wall_ids)
        bcs = [inflow_bc, side_wall_bc]

        u, M = firedrake.split(z)
        fields = {"velocity": u, "membrane_stress": M, "thickness": h}
        boundary_ids = {"outflow_ids": outflow_ids}

        L = (
            viscous_power(**fields, flow_law_exponent=n1, flow_law_coefficient=A1) +
            viscous_power(**fields, flow_law_exponent=n3, flow_law_coefficient=A3) +
            ice_shelf_momentum_balance(**fields, **boundary_ids)
        )

        F = derivative(L, z)
        qdegree = max(8, degree ** glen_flow_law)
        pparams = {"form_compiler_parameters": {"quadrature_degree": qdegree}}
        problem = NonlinearVariationalProblem(F, z, bcs, **pparams)
        solver = NonlinearVariationalSolver(problem, **sparams)

        num_continuation_steps = 5
        for exponent in np.linspace(1.0, glen_flow_law, num_continuation_steps):
            n3.assign(exponent)
            solver.solve()

        u, M = z.subfunctions
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
