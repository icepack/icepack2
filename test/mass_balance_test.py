import numpy as np
from numpy import pi as π
import pytest
import firedrake
from firedrake import dx, inner, max_value, Constant, conditional
import irksome
from icepack2 import model

@pytest.mark.parametrize("degree", [0, 1, 2])
def test_convergence_rate(degree):
    errors, mesh_sizes = [], []
    k_min, k_max, num_steps = 5 - degree, 8 - degree, 9
    for nx in np.logspace(k_min, k_max, num_steps, base=2, dtype=int):
        mesh = firedrake.UnitSquareMesh(nx, nx, diagonal="crossed")
        x = firedrake.SpatialCoordinate(mesh)

        element = firedrake.FiniteElement("DG", "triangle", degree)
        if degree > 0:
            bernstein = firedrake.FiniteElement("Bernstein", "triangle", degree)
            element = firedrake.BrokenElement(bernstein)

        Q = firedrake.FunctionSpace(mesh, element)

        y = Constant((3/4, 1/2))
        r = Constant(1/8)
        expr = max_value(0, r**2 - inner(x - y, x - y))
        h = firedrake.project(expr, Q)

        c = Constant((1/2, 1/2))
        w = x - c
        u = firedrake.as_vector((-w[1], w[0]))

        problem = model.mass_balance(
            thickness=h,
            velocity=u,
            accumulation=Constant(0.0),
            test_function=firedrake.TestFunction(Q),
        )

        tableau = irksome.BackwardEuler()
        if degree > 0:
            tableau = irksome.RadauIIA(max(2, degree))

        δx = mesh.cell_sizes.dat.data_ro.max()
        u_max = 0.5
        δt = 0.5 * δx / u_max

        t = Constant(0.0)
        dt = firedrake.Constant(δt)
        final_time = 2 * π

        params = {
            "solver_parameters": {
                "snes_type": "ksponly",
                "ksp_type": "gmres",
                "pc_type": "bjacobi",
            },
        }
        solver = irksome.TimeStepper(problem, tableau, t, dt, h, **params)

        while float(t) < final_time:
            if float(t) + float(dt) > final_time:
                dt.assign(final_time - float(t))
            solver.advance()
            t.assign(float(t) + float(dt))

        mesh_sizes.append(δx)
        errors.append(firedrake.assemble(abs(h - expr) * dx))
        print(".", end="", flush=True)

    log_mesh_sizes = np.log2(np.array(mesh_sizes))
    log_errors = np.log2(np.array(errors))
    slope, intercept = np.polyfit(log_mesh_sizes, log_errors, 1)
    print(f"degree {degree}: log(error) ~= {slope:g} * log(dx) {intercept:+g}")
    assert slope > degree - 0.55


@pytest.mark.parametrize("degree", [0, 1])
def test_frontal_ablation(degree):
    errors, mesh_sizes = [], []
    k_min, k_max, num_steps = 5 - degree, 8 - degree, 9
    for nx in np.logspace(k_min, k_max, num_steps, base=2, dtype=int):
        mesh = firedrake.UnitSquareMesh(nx, nx, quadrilateral=True)
        x = firedrake.SpatialCoordinate(mesh)

        element = firedrake.FiniteElement("DQ", "quadrilateral", degree)
        # Try Bernstein?

        Q = firedrake.FunctionSpace(mesh, element)

        u_max = Constant(1.0)
        u = Constant((u_max, 0.0))

        h_in = Constant(1.0)
        x = firedrake.SpatialCoordinate(mesh)
        L = Constant(0.25)
        expr = conditional(x[0] < L, h_in, 0.0)
        h_0 = firedrake.Function(Q).project(expr)
        h = h_0.copy(deepcopy=True)

        a = Constant(0.0)

        U_c = Constant(0.5)
        x_c = Constant(0.5)
        u_c = firedrake.as_vector((conditional(x[0] >= x_c, U_c, 0), 0))

        h_exact = conditional(x[0] <= x_c, h_in, h_in * u_max / (u_max + U_c))

        ϕ = firedrake.TestFunction(Q)
        problem = model.mass_balance(
            thickness=h,
            velocity=u + u_c,
            accumulation=a,
            thickness_inflow=h_in,
            test_function=ϕ,
        )

        δx = mesh.cell_sizes.dat.data_ro.max()
        δt = 0.5 * δx / u_max

        t = Constant(0.0)
        dt = Constant(δt)
        final_time = 2.0

        lower, upper = firedrake.Function(Q), firedrake.Function(Q)
        upper.assign(np.inf)
        params = {
            "solver_parameters": {
                "snes_type": "vinewtonrsls",
                "snes_atol": 1e-12,
            },
            "stage_type": "value",
            "basis_type": "Bernstein",
            "bounds": ("stage", lower, upper),
        }

        tableau = irksome.BackwardEuler()
        solver = irksome.TimeStepper(problem, tableau, t, dt, h, **params)

        while float(t) < final_time:
            if float(t) + float(dt) > final_time:
                dt.assign(final_time - float(t))
            solver.advance()
            t.assign(t + dt)

        mesh_sizes.append(δx)
        errors.append(firedrake.assemble(abs(h - h_exact) * dx))
        print(".", end="", flush=True)

    log_mesh_sizes = np.log2(np.array(mesh_sizes))
    log_errors = np.log2(np.array(errors))
    slope, intercept = np.polyfit(log_mesh_sizes, log_errors, 1)
    print(f"degree {degree}: log(error) ~= {slope:g} * log(dx) {intercept:+g}")
