import numpy as np
from numpy import pi as π
import pytest
import firedrake
from firedrake import dx, inner, max_value, Constant
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
