import numpy as np
from numpy import pi as π
import firedrake
from firedrake import inner, grad, dx, ds, exp, min_value, max_value, Constant
import irksome
from icepack2.model import mass_balance
from icepack2.model.variational import momentum_balance, flow_law, friction_law
from icepack2.constants import gravity, ice_density, glen_flow_law


def form_momentum_balance(z, w, h, b, H, α, rheo1, rheo3):
    u, M, τ = z
    v, N, σ = w

    F_stress_balance = momentum_balance(
        velocity=u,
        membrane_stress=M,
        basal_stress=τ,
        thickness=h,
        surface=b + h,
        test_function=v,
    )

    F_glen_law = flow_law(
        velocity=u, membrane_stress=M, thickness=H, **rheo3, test_function=N
    )

    F_linear_law = α * flow_law(
        velocity=u, membrane_stress=M, thickness=H, **rheo1, test_function=N
    )

    F_weertman_drag = friction_law(
        velocity=u, basal_stress=τ, **rheo3, test_function=σ
    )

    F_viscous_drag = α * friction_law(
        velocity=u, basal_stress=τ, **rheo1, test_function=σ
    )

    return (
        F_stress_balance
        + F_glen_law
        + F_linear_law
        + F_weertman_drag
        + F_viscous_drag
    )


def run_simulation(refinement_level: int):
    radius = Constant(12e3)
    mesh = firedrake.UnitDiskMesh(refinement_level)
    mesh.coordinates.dat.data[:] *= float(radius)

    # Make a bunch of finite elements and function spaces
    cg1 = firedrake.FiniteElement("CG", "triangle", 1)
    dg1 = firedrake.FiniteElement("DG", "triangle", 1)
    dg0 = firedrake.FiniteElement("DG", "triangle", 0)
    S = firedrake.FunctionSpace(mesh, cg1)
    Q = firedrake.FunctionSpace(mesh, dg1)
    V = firedrake.VectorFunctionSpace(mesh, cg1)
    Σ = firedrake.TensorFunctionSpace(mesh, dg0, symmetry=True)
    Z = V * Σ * V * Q

    x = firedrake.SpatialCoordinate(mesh)

    # Make the bed topography
    B = Constant(4e3)
    r_b = Constant(150e3 / (2 * π))
    expr = B * exp(-inner(x, x) / r_b**2)
    b = firedrake.Function(S).interpolate(expr)

    # Make the mass balance field
    z_measured = Constant(1600.0)
    a_measured = Constant(-0.917 * 8.7)
    a_top = Constant(0.7)
    z_top = Constant(4e3)
    δa_δz = (a_top - a_measured) / (z_top - z_measured)
    a_max = Constant(0.7)

    def smb(z):
        return min_value(a_max, a_measured + δa_δz * (z - z_measured))

    # Make the initial thickness
    r_h = Constant(5e3)
    H = Constant(100.0)
    expr = H * firedrake.max_value(0, 1 - inner(x, x) / r_h**2)
    h_0 = firedrake.Function(Q).interpolate(expr)

    s = firedrake.Function(Q).interpolate(b + h_0)
    a = firedrake.Function(Q).interpolate(smb(s))

    # Fluidity of ice in yr⁻¹ MPa⁻³ at 0C
    A = Constant(158.0)

    # Make an initial guess for the velocity using SIA
    ρ_I = Constant(ice_density)
    g = Constant(gravity)
    n = Constant(glen_flow_law)

    # Compute the initial velocity using the dual form of SSA
    z = firedrake.Function(Z)
    u, M, τ, h = firedrake.split(z)

    τ_c = Constant(0.1)
    ε_c = Constant(A * τ_c ** n)

    K = h * A / (n + 2)
    U_c = Constant(100.0)
    u_c = K * τ_c ** n + U_c

    rheo3 = {
        "flow_law_exponent": n,
        "flow_law_coefficient": ε_c / τ_c ** n,
        "sliding_exponent": n,
        "sliding_coefficient": u_c / τ_c ** n,
    }

    α = firedrake.Constant(1e-4)
    rheo1 = {
        "flow_law_exponent": 1,
        "flow_law_coefficient": ε_c / τ_c,
        "sliding_exponent": 1,
        "sliding_coefficient": u_c / τ_c,
    }

    fields = {
        "velocity": u,
        "membrane_stress": M,
        "basal_stress": τ,
        "thickness": h,
        "surface": b + h,
    }

    degree = 1
    qdegree = max(8, degree ** glen_flow_law)
    pparams = {"form_compiler_parameters": {"quadrature_degree": qdegree}}

    sparams = {
        "solver_parameters": {
            "snes_monitor": None,
            "snes_type": "newtonls",
            "snes_max_it": 200,
            "snes_linesearch_type": "nleqerr",
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    }

    print("Initial momentum solve")
    v, N, σ, η = firedrake.TestFunctions(Z)
    F_momentum = form_momentum_balance((u, M, τ), (v, N, σ), h, b, H, α, rheo1, rheo3)
    F_mass = (h - h_0) * η * dx
    F = F_momentum + F_mass
    problem = firedrake.NonlinearVariationalProblem(F, z, **pparams)
    solver = firedrake.NonlinearVariationalSolver(problem, **sparams)

    num_continuation_steps = 5
    for exponent in np.linspace(1.0, 3.0, num_continuation_steps):
        n.assign(exponent)
        solver.solve()

    F_mass = mass_balance(thickness=h, velocity=u, accumulation=a, test_function=η)
    F = F_momentum + F_mass

    tableau = irksome.BackwardEuler()
    t = Constant(0.0)
    dt = Constant(1.0 / 6)

    lower = firedrake.Function(Z)
    upper = firedrake.Function(Z)
    lower.assign(-np.inf)
    upper.assign(+np.inf)
    lower.subfunctions[3].assign(0.0)
    bounds = ("stage", lower, upper)

    bparams = {
        "solver_parameters": {
            "snes_monitor": None,
            "snes_type": "vinewtonrsls",
            "snes_max_it": 200,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        "stage_type": "value",
        "basis_type": "Bernstein",
        "bounds": bounds,
    }

    solver = irksome.TimeStepper(F, tableau, t, dt, z, **bparams, **pparams)

    us = [z.subfunctions[0].copy(deepcopy=True)]
    hs = [z.subfunctions[3].copy(deepcopy=True)]

    print("Time-dependent solves")
    final_time = 300.0
    num_steps = int(final_time / float(dt))
    for step in range(num_steps):
        solver.advance()
        h = z.subfunctions[3]
        a.interpolate(smb(b + h))

        us.append(z.subfunctions[0].copy(deepcopy=True))
        hs.append(z.subfunctions[3].copy(deepcopy=True))

    return hs, us


def test_dome_problem():
    hs, us = run_simulation(4)
    volumes = [firedrake.assemble(h * dx) for h in hs]
    assert volumes[0] <= volumes[-1]
