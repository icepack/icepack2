import pytest
import numpy as np
import sympy
from sympy import lambdify, simplify, diff
import firedrake
from firedrake import (
    as_vector,
    max_value,
    Constant,
    Function,
    derivative,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    DirichletBC,
)
from icepack2 import constants
from icepack2.model.minimization import (
    viscous_power,
    friction_power,
    calving_terminus,
    momentum_balance,
    ice_shelf_momentum_balance,
)


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


@pytest.mark.parametrize("degree", (1, 2))
def test_composite_rheology_floating(degree):
    lx, ly = 20e3, 20e3
    Lx, Ly = Constant(lx), Constant(ly)
    h0, dh = Constant(500.0), Constant(100.0)
    u_inflow = Constant(100.0)

    ρ_I = Constant(constants.ice_density)
    ρ_W = Constant(constants.water_density)
    g = Constant(constants.gravity)

    # Use a combination of two different rheology exponents.
    n_1 = firedrake.Constant(1.0)
    n_3 = firedrake.Constant(constants.glen_flow_law)

    τ_c = Constant(0.1)
    ε_c = Constant(0.01)
    A_1 = ε_c / τ_c ** n_1
    A_3 = ε_c / τ_c ** n_3

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

    errors, mesh_sizes = [], []
    k_min, k_max, num_steps = 5 - degree, 8 - degree, 9
    for nx in np.logspace(k_min, k_max, num_steps, base=2, dtype=int):
        mesh = firedrake.RectangleMesh(nx, nx, lx, ly, diagonal="crossed")
        x, y = firedrake.SpatialCoordinate(mesh)

        # Make some function spaces.
        cg = firedrake.FiniteElement("CG", "triangle", degree)
        dg = firedrake.FiniteElement("DG", "triangle", degree - 1)
        Q = firedrake.FunctionSpace(mesh, cg)
        V = firedrake.VectorFunctionSpace(mesh, cg)
        Σ = firedrake.TensorFunctionSpace(mesh, dg, symmetry=True)
        Z = V * Σ
        z = Function(Z)
        z.sub(0).assign(Constant(u_inflow, 0))

        expr = u_inflow + exact_δu(x, n_1, A_1) + exact_δu(x, n_3, A_3)
        u_exact = Function(V).interpolate(as_vector((expr, 0)))

        h = Function(Q).interpolate(h0 - dh * x / Lx)

        # Make the boundary conditions.
        inflow_ids = (1,)
        outflow_ids = (2,)
        side_wall_ids = (3, 4)

        inflow_bc = DirichletBC(Z.sub(0), Constant((u_inflow, 0)), inflow_ids)
        side_wall_bc = DirichletBC(Z.sub(0).sub(1), 0, side_wall_ids)
        bcs = [inflow_bc, side_wall_bc]

        # Make the model specification and solver.
        u, M = firedrake.split(z)
        fields = {"velocity": u, "membrane_stress": M, "thickness": h}
        boundary_ids = {"outflow_ids": outflow_ids}

        L = (
            viscous_power(**fields, flow_law_exponent=n_1, flow_law_coefficient=A_1) +
            viscous_power(**fields, flow_law_exponent=n_3, flow_law_coefficient=A_3) +
            ice_shelf_momentum_balance(**fields, **boundary_ids)
        )

        F = derivative(L, z)
        qdegree = max(8, degree ** constants.glen_flow_law)
        pparams = {"form_compiler_parameters": {"quadrature_degree": qdegree}}
        problem = NonlinearVariationalProblem(F, z, bcs, **pparams)
        solver = NonlinearVariationalSolver(problem, **sparams)

        # Solve the problem using a continuation method -- step the flow law
        # exponent from 1 to 3 in small intervals.
        num_steps = 5
        for exponent in np.linspace(1.0, constants.glen_flow_law, num_steps):
            n_3.assign(exponent)
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


def test_manufactured_solution():
    L, ρ_I, ρ_W, g = sympy.symbols("L ρ_I ρ_W g", real=True, positive=True)
    A_1, A_3 = sympy.symbols("A_1 A_3", real=True, positive=True)
    n_1, n_3 = sympy.symbols("n_1 n_3", integer=True, positive=True)

    def strain_rate(M, A_1, A_3, n_1, n_3):
        return A_1 * (M / 2)**n_1 + A_3 * (M / 2)**n_3

    def sliding_velocity(τ, K_1, K_3, m_1, m_3):
        return -K_1 * τ**m_1 - K_3 * τ**m_3

    def stress_balance(x, h, s, M, τ):
        return diff(h * M, x) + τ - ρ_I * g * h * diff(s, x)

    def boundary_condition(x, u, h, s, M):
        d = (s - h).subs(x, L)
        τ_c = (ρ_I * g * h**2 - ρ_W * g * d**2) / 2
        return (h * M - τ_c).subs(x, L)

    # Create the thickness and surface elevation
    x = sympy.symbols("x", real=True)
    h0, δh = sympy.symbols("h0 δh", real=True, positive=True)
    h = h0 - δh * x / L

    s0, δs = sympy.symbols("s0 δs", real=True, positive=True)
    s = s0 - δs * x / L

    # The variable `β` is the fraction of the driving stress that membrane
    # stress divergence will take up; this is chosen to eliminate some extra
    # terms and make the rest of the algebra simpler. See the website also.
    h_L = h0 - δh
    s_L = s0 - δs
    β = δh / δs * (ρ_I * h_L ** 2 - ρ_W * (s_L - h_L) ** 2) / (ρ_I * h_L**2)

    ρ = β * ρ_I * δs / δh
    P = ρ * g * h / 4
    δP = ρ * g * δh / 4
    P0 = ρ * g * h0 / 4

    M = ρ * g * h / 2
    τ = -(1 - β) * ρ_I * g * h * δs / L

    # The exact velocity field looks similar to the floating case by using a
    # careful choice of `β`.
    u_0 = sympy.symbols("u_0", real=True, positive=True)
    δu_1 = L * A_1 * (P0 ** (n_1 + 1) - P ** (n_1 + 1)) / ((n_1 + 1) * δP)
    δu_3 = L * A_3 * (P0 ** (n_3 + 1) - P ** (n_3 + 1)) / ((n_3 + 1) * δP)
    u = u_0 + δu_1 + δu_3

    # Make the linear part of the rheology a much smaller effect than the
    # nonlinear part -- at a stress of 100kPa, the nonlinear rheology gives
    # a strain rate of 10 m/yr/km, while the linear part only 1 m/yr/km.
    ε_1 = 0.001
    ε_3 = 0.01
    τ_c = 0.1

    values = {
        u_0: 100,
        h0: 500,
        δh: 100,
        s0: 150,
        δs: 90,
        L: 20e3,
        A_1: ε_1 / τ_c,
        A_3: ε_3 / τ_c ** constants.glen_flow_law,
        ρ_I: constants.ice_density,
        ρ_W: constants.water_density,
        n_1: 1,
        n_3: constants.glen_flow_law,
        g: constants.gravity,
    }

    # Check the momentum conservation law and boundary condition.
    τ_b = lambdify(x, τ.subs(values), "numpy")
    τ_d = lambdify(x, (-ρ_I * g * h * diff(s, x)).subs(values), "numpy")
    τ_m = lambdify(x, simplify(diff(h * M, x)).subs(values), "numpy")
    xs = np.linspace(0, values[L], 21)

    tolerance = 1e-8
    assert abs(boundary_condition(x, u, h, s, M).subs(values)) < tolerance

    stress_norm = np.max(np.abs(τ_d(xs)))
    assert np.max(np.abs(τ_m(xs) + τ_b(xs) + τ_d(xs))) < tolerance * stress_norm

    # Check the constitutive relation.
    ε_u = lambdify(x, simplify(diff(u, x).subs(values)), "numpy")
    ε_M = lambdify(
        x, simplify(strain_rate(M, A_1, A_3, n_1, n_3).subs(values)), "numpy"
    )
    assert np.max(np.abs(ε_u(xs) - ε_M(xs))) < tolerance * np.max(np.abs(ε_u(xs)))

    # Check the sliding law.
    α = 0.1
    m_1 = 1
    m_3 = constants.glen_flow_law
    K_1 = α * u / abs(τ) ** m_1
    K_3 = (u - K_1 * abs(τ) ** m_1) / abs(τ)**m_3

    U = lambdify(x, simplify(u).subs(values), "numpy")
    U_b = lambdify(
        x, simplify(sliding_velocity(τ, K_1, K_3, m_1, m_3).subs(values)), "numpy"
    )
    assert np.max(np.abs(U(xs) - U_b(xs))) < tolerance * np.max(np.abs(U(xs)))


@pytest.mark.parametrize("degree", (1, 2))
def test_composite_rheology_grounded(degree):
    lx, ly = 20e3, 20e3
    Lx, Ly = Constant(lx), Constant(ly)
    h0, dh = Constant(500.0), Constant(100.0)
    s0, ds = Constant(150.0), Constant(90.0)
    u_inflow = Constant(100.0)

    # See the previous test for explanations of all the setup.
    ρ_I = Constant(constants.ice_density)
    ρ_W = Constant(constants.water_density)
    g = Constant(constants.gravity)

    n_1 = firedrake.Constant(1.0)
    n_3 = firedrake.Constant(constants.glen_flow_law)

    m_1 = firedrake.Constant(1.0)
    m_3 = firedrake.Constant(constants.weertman_sliding_law)

    τ_c = Constant(0.1)
    ε_1 = Constant(0.001)
    ε_3 = Constant(0.01)
    A_1 = ε_1 / τ_c ** n_1
    A_3 = ε_3 / τ_c ** n_3

    h_L = h0 - dh
    s_L = s0 - ds
    β = dh / ds * (ρ_I * h_L**2 - ρ_W * (s_L - h_L)**2) / (ρ_I * h_L**2)

    def exact_δu(x, n, A):
        ρ = β * ρ_I * ds / dh
        h = h0 - dh * x / Lx
        P = ρ * g * h / 4
        dP = ρ * g * dh / 4
        P0 = ρ * g * h0 / 4
        du = Lx * A * (P0 ** (n + 1) - P ** (n + 1)) / ((n + 1) * dP)
        return du

    errors, mesh_sizes = [], []
    k_min, k_max, num_steps = 5 - degree, 8 - degree, 9
    for nx in np.logspace(k_min, k_max, num_steps, base=2, dtype=int):
        mesh = firedrake.RectangleMesh(nx, nx, lx, ly, diagonal="crossed")
        x, y = firedrake.SpatialCoordinate(mesh)

        # Make some function spaces.
        cg = firedrake.FiniteElement("CG", "triangle", degree)
        dg = firedrake.FiniteElement("DG", "triangle", degree - 1)
        Q = firedrake.FunctionSpace(mesh, cg)
        V = firedrake.VectorFunctionSpace(mesh, cg)
        Σ = firedrake.TensorFunctionSpace(mesh, dg, symmetry=True)
        T = firedrake.VectorFunctionSpace(mesh, dg)
        Z = V * Σ * T
        z = Function(Z)
        z.sub(0).assign(Constant((u_inflow, 0)))

        # Make the exact velocity, thickness, surface elevation, and sliding
        # coefficients.
        u_expr = u_inflow + exact_δu(x, n_1, A_1) + exact_δu(x, n_3, A_3)
        u_exact = Function(V).interpolate(as_vector((u_expr, 0)))

        h = Function(Q).interpolate(h0 - dh * x / Lx)
        s = Function(Q).interpolate(s0 - ds * x / Lx)

        α = Constant(0.1)
        τ_expr = (1 - β) * ρ_I * g * h * ds / Lx
        K_1 = α * u_expr / τ_expr ** m_1
        K_3 = (u_expr - K_1 * τ_expr ** m_1) / τ_expr ** m_3

        # Create the boundary conditions.
        inflow_ids = (1,)
        outflow_ids = (2,)
        side_wall_ids = (3, 4)

        inflow_bc = DirichletBC(Z.sub(0), Constant((u_inflow, 0)), inflow_ids)
        side_wall_bc = DirichletBC(Z.sub(0).sub(1), 0, side_wall_ids)
        bcs = [inflow_bc, side_wall_bc]

        # Create the model specification and solvers.
        u, M, τ = firedrake.split(z)
        fields = {
            "velocity": u,
            "membrane_stress": M,
            "basal_stress": τ,
            "thickness": h,
            "surface": s,
        }

        rheology_1 = {
            "flow_law_exponent": n_1,
            "flow_law_coefficient": A_1,
            "sliding_exponent": m_1,
            "sliding_coefficient": K_1,
        }

        rheology_3 = {
            "flow_law_exponent": n_3,
            "flow_law_coefficient": A_3,
            "sliding_exponent": m_3,
            "sliding_coefficient": K_3,
        }

        boundary_ids = {"outflow_ids": outflow_ids}
        L = (
            viscous_power(**fields, **rheology_1) +
            viscous_power(**fields, **rheology_3) +
            friction_power(**fields, **rheology_1) +
            friction_power(**fields, **rheology_3) +
            calving_terminus(**fields, **boundary_ids) +
            momentum_balance(**fields)
        )

        F = derivative(L, z)
        qdegree = max(8, degree ** constants.glen_flow_law)
        pparams = {"form_compiler_parameters": {"quadrature_degree": qdegree}}
        problem = NonlinearVariationalProblem(F, z, bcs, **pparams)
        solver = NonlinearVariationalSolver(problem, **sparams)

        # Solve the problem using a continuation method.
        num_steps = 5
        for exponent in np.linspace(1.0, constants.glen_flow_law, num_steps):
            n_3.assign(exponent)
            m_3.assign(exponent)
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
