import numpy as np
import firedrake
from firedrake import (
    sqrt, exp, min_value, max_value, inner, as_vector, Constant, dx
)
from irksome import BackwardEuler, TimeStepper
from icepack2.model import mass_balance
from icepack2.model.variational import (
    momentum_balance, flow_law, friction_law, calving_terminus
)
from icepack2.constants import (
    gravity, ice_density, water_density, glen_flow_law, weertman_sliding_law
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


def form_momentum_balance(z, w, h, s, f, H, α, rheo1, rheo3):
    u, M, τ = z
    v, N, σ = w

    F_stress_balance = momentum_balance(
        membrane_stress=M,
        basal_stress=τ,
        thickness=h,
        surface=s,
        test_function=v,
    )

    F_glen_law = flow_law(
        velocity=u, membrane_stress=M, thickness=H, **rheo3, test_function=N
    )

    F_linear_law = α * flow_law(
        velocity=u, membrane_stress=M, thickness=H, **rheo1, test_function=N
    )

    F_weertman_drag = friction_law(
        velocity=u, basal_stress=τ, floating=f, **rheo3, test_function=σ
    )

    F_viscous_drag = α * friction_law(
        velocity=u, basal_stress=τ, floating=f, **rheo1, test_function=σ
    )

    F_terminus = calving_terminus(
        thickness=h, surface=s, test_function=v, outflow_ids=(2,)
    )

    return (
        F_stress_balance
        + F_glen_law
        + F_linear_law
        + F_weertman_drag
        + F_viscous_drag
        + F_terminus
    )


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

    n = Constant(glen_flow_law)
    m = Constant(weertman_sliding_law)

    τ_c = Constant(0.1)
    ε_c = Constant(A * τ_c ** n)

    ρ_I = Constant(ice_density)
    ρ_W = Constant(water_density)
    g = Constant(gravity)

    # Friction coefficient in MPa (m yr⁻¹)⁻¹ᐟ³
    C = Constant(1e-2)
    K = Constant(C ** (-m)) ## TODO: just use 1e6
    u_c = Constant(K * τ_c ** m)

    rheo3 = {
        "flow_law_exponent": n,
        "flow_law_coefficient": ε_c / τ_c ** n,
        "sliding_exponent": m,
        "sliding_coefficient": u_c / τ_c ** m,
    }

    rheo1 = {
        "flow_law_exponent": 1,
        "flow_law_coefficient": ε_c / τ_c,
        "sliding_exponent": 1,
        "sliding_coefficient": u_c / τ_c,
    }

    z = firedrake.Function(Z)
    δu = Constant(90.0)
    Lx = Constant(lx)
    u_expr = firedrake.as_vector((δu * x[0] / Lx, 0))
    z.sub(0).interpolate(u_expr)
    z.sub(3).assign(h_0)
    u, M, τ, h = firedrake.split(z)
    s = max_value(b + h, (1 - ρ_I / ρ_W) * h)

    # TODO: Think very hard about the float mask
    p_I = ρ_I * g * h
    p_W = ρ_W * g * max_value(0, -(s - h))
    N = max_value(0, p_I - p_W)
    f = min_value(N / (2 * τ_c), 1.0)

    fields = {
        "velocity": u,
        "membrane_stress": M,
        "basal_stress": τ,
        "thickness": h,
        "surface": max_value(b + h, (1 - ρ_I / ρ_W) * h),
        "floating": f,
    }

    inflow_bc = firedrake.DirichletBC(Z.sub(0), 0, [1])
    side_wall_bc = firedrake.DirichletBC(Z.sub(0).sub(1), 0, [3, 4])
    bcs = [inflow_bc, side_wall_bc]

    degree = 1
    qdegree = max(8, degree ** glen_flow_law)
    pparams = {
        "form_compiler_parameters": {"quadrature_degree": qdegree}
    }

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
    α = firedrake.Constant(1e-4)
    H = Constant(500.0)

    v, N, σ, η = firedrake.TestFunctions(Z)

    F_momentum = form_momentum_balance((u, M, τ), (v, N, σ), h, s, f, H, α, rheo1, rheo3)
    F_mass = (h - h_0) * η * dx
    F = F_momentum + F_mass
    problem = firedrake.NonlinearVariationalProblem(F, z, **pparams, bcs=bcs)
    solver = firedrake.NonlinearVariationalSolver(problem, **sparams)

    num_continuation_steps = 5
    for r in np.linspace(0.0, 1.0, num_continuation_steps):
        n.assign((1 - r) + r * glen_flow_law)
        m.assign((1 - r) + r * weertman_sliding_law)
        solver.solve()

    print("Time-dependent solve")
    F_mass = mass_balance(
        thickness=h,
        velocity=u,
        accumulation=a,
        thickness_inflow=h_0,
        test_function=η,
    )

    t = Constant(0.0)
    timestep = 1.0
    dt = Constant(timestep)
    F = F_momentum + F_mass

    lower = firedrake.Function(Z)
    upper = firedrake.Function(Z)
    lower.assign(-np.inf)
    upper.assign(+np.inf)
    lower.subfunctions[3].assign(0.0)

    params = {
        "solver_parameters": {
            "snes_monitor": None,
            "snes_type": "vinewtonrsls",
            "snes_max_it": 200,
            "snes_linesearch_type": "nleqerr",
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        "stage_type": "value",
        "basis_type": "Bernstein",
        "bounds": ("stage", lower, upper),
    }
    method = BackwardEuler()
    solver = TimeStepper(F, method, t, dt, z, **params, **pparams, bcs=bcs)

    final_time = 1.0
    num_steps = int(final_time / timestep)
    for step in range(num_steps):
        solver.advance()

    u, M, τ, h = z.subfunctions
    firedrake.assemble((h - h_0) * dx)

    return z


def test_mismip():
    run_simulation(ny=20)
