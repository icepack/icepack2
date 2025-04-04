import pytest
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
from icepack2 import model


@pytest.mark.parametrize("degree", (1, 2))
@pytest.mark.parametrize("form", ("minimization", "variational"))
def test_convergence_rate_grounded(degree, form):
    Lx, Ly = Constant(20e3), Constant(20e3)
    h0, dh = Constant(500.0), Constant(100.0)
    u_inflow = Constant(100.0)

    n = firedrake.Constant(glen_flow_law)
    m = firedrake.Constant(glen_flow_law)

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
        T = firedrake.VectorFunctionSpace(mesh, dg)
        Z = V * Σ * T
        z = Function(Z)
        z.sub(0).assign(Constant((u_inflow, 0)))

        u_exact = Function(V).interpolate(as_vector((exact_u(x), 0)))

        h = Function(Q).interpolate(h0 - dh * x / Lx)
        ds = (1 + β) * ρ / ρ_I * dh
        s = Function(Q).interpolate(d + h0 - dh + ds * (1 - x / Lx))

        # TODO: adjust the yield stress so that this has a more sensible value
        C = Function(Q).interpolate(friction(x))
        u_c = Function(Q).interpolate((τ_c / C)**m)

        # Create the boundary conditions
        inflow_ids = (1,)
        outflow_ids = (2,)
        side_wall_ids = (3, 4)

        inflow_bc = firedrake.DirichletBC(Z.sub(0), Constant((u_inflow, 0)), inflow_ids)
        side_wall_bc = firedrake.DirichletBC(Z.sub(0).sub(1), 0, side_wall_ids)
        bcs = [inflow_bc, side_wall_bc]

        n = firedrake.Constant(1.0)
        m = firedrake.Constant(1.0)

        # Make the material parameters and input fields
        rheology = {
            "flow_law_exponent": n,
            "flow_law_coefficient": ε_c / τ_c ** n,
            "sliding_exponent": m,
            "sliding_coefficient": u_c / τ_c ** m,
        }

        u, M, τ = firedrake.split(z)
        fields = {
            "velocity": u,
            "membrane_stress": M,
            "basal_stress": τ,
            "thickness": h,
            "surface": s,
        }

        boundary_ids = {"outflow_ids": outflow_ids}

        qdegree = max(8, degree ** glen_flow_law)
        pparams = {"form_compiler_parameters": {"quadrature_degree": qdegree}}
        if form == "minimization":
            fns = [
                model.minimization.viscous_power,
                model.minimization.friction_power,
                model.minimization.calving_terminus,
                model.minimization.momentum_balance,
            ]

            def form_problem(rheology):
                L = sum(fn(**fields, **rheology, **boundary_ids) for fn in fns)
                F = derivative(L, z)
                J = derivative(F, z)
                problem = NonlinearVariationalProblem(F, z, bcs, J=J, **pparams)
                return problem
        elif form == "variational":
            v, N, σ = firedrake.TestFunctions(Z)
            fns = [
                (model.variational.flow_law, N),
                (model.variational.friction_law, σ),
                (model.variational.calving_terminus, v),
                (model.variational.momentum_balance, v),
            ]

            def form_problem(rheology):
                F = sum(
                    fn(**fields, **rheology, **boundary_ids, test_function=φ)
                    for fn, φ in fns
                )
                J = derivative(F, z)
                problem = NonlinearVariationalProblem(F, z, bcs, J=J, **pparams)
                return problem

        sparams = {
            "solver_parameters": {
                "snes_type": "newtonls",
                "snes_max_it": 200,
                "snes_linesearch_type": "nleqerr",
                "ksp_type": "gmres",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        }

        num_continuation_steps = 5
        λs = np.linspace(0.0, 1.0, num_continuation_steps)
        for λ in λs:
            n.assign((1 - λ) + λ * glen_flow_law)
            m.assign((1 - λ) + λ * weertman_sliding_law)
            problem = form_problem(rheology)
            solver = NonlinearVariationalSolver(problem, **sparams)
            solver.solve()

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


@pytest.mark.parametrize("degree", [1, 2])
def test_convergence_rate_floating(degree):
    Lx, Ly = Constant(20e3), Constant(20e3)
    h0, dh = Constant(500.0), Constant(100.0)
    u_inflow = Constant(100.0)

    n = firedrake.Constant(glen_flow_law)

    τ_c = Constant(0.1)
    ε_c = Constant(0.01)
    A = ε_c / τ_c**n

    # This is an exact solution for the velocity of a floating ice shelf with
    # constant temperature and linearly decreasing thickness. See Greve and
    # Blatter for the derivation.
    def exact_u(x):
        ρ = ρ_I * (1 - ρ_I / ρ_W)
        Z = A * (ρ * g * h0 / 4) ** n
        q = 1 - (1 - (dh / h0) * (x / Lx)) ** (n + 1)
        du = Z * q * Lx * (h0 / dh) / (n + 1)
        return u_inflow + du


    # We'll use the same perturbation to `u` throughout these tests.
    def perturb_u(x, y):
        px, py = x / Lx, y / Ly
        q = 16 * px * (1 - px) * py * (1 - py)
        return 60 * q * (px - 0.5)

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

        u_exact = Function(V).interpolate(as_vector((exact_u(x), 0)))

        h = Function(Q).interpolate(h0 - dh * x / Lx)
        inflow_ids = (1,)
        outflow_ids = (2,)
        side_wall_ids = (3, 4)

        inflow_bc = firedrake.DirichletBC(Z.sub(0), Constant((u_inflow, 0)), inflow_ids)
        side_wall_bc = firedrake.DirichletBC(Z.sub(0).sub(1), 0, side_wall_ids)
        bcs = [inflow_bc, side_wall_bc]

        fns = [
            model.minimization.viscous_power,
            #model.minimization.calving_terminus,
            model.minimization.ice_shelf_momentum_balance,
        ]

        rheology = {
            "flow_law_exponent": n,
            "flow_law_coefficient": ε_c / τ_c ** n,
        }

        u, M = firedrake.split(z)
        fields = {"velocity": u, "membrane_stress": M, "thickness": h}

        boundary_ids = {"outflow_ids": outflow_ids}
        L = sum(fn(**fields, **rheology, **boundary_ids) for fn in fns)
        F = derivative(L, z)

        # Regularize second derivative of the Lagrangian
        λ = firedrake.Constant(1e-3)
        regularized_rheology = {
            "flow_law_exponent": 1,
            "flow_law_coefficient": λ * ε_c / τ_c,
        }

        K = model.minimization.viscous_power(**fields, **regularized_rheology)
        J = derivative(derivative(L + K, z), z)

        qdegree = max(8, degree ** glen_flow_law)
        pparams = {"form_compiler_parameters": {"quadrature_degree": qdegree}}
        problem = NonlinearVariationalProblem(F, z, bcs, J=J, **pparams)
        solver = NonlinearVariationalSolver(problem)

        solver.solve()
        λ.assign(0.0)
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
