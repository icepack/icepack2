import numpy as np
import firedrake
from firedrake import inner, grad, div, dx, ds
import irksome
from irksome import Dt
from icepack.constants import gravity as g, ice_density as ρ


def form(z, **inputs):
    # Get all the external data
    names = ("flow_law_exponent", "bed", "accumulation", "fluidity")
    n, b, a, A = map(inputs.__getitem__, names)

    # Get some test functions
    Z = z.function_space()
    w = firedrake.TestFunction(Z)
    η, v, r = firedrake.split(w)

    # Extract the solution variables
    h, u, q = firedrake.split(z)

    # Create the mass and momentum balance equations
    mass_balance = (Dt(h) + div(q) - a) * η * dx

    s = b + h
    P = ρ * g * h
    S_n = inner(grad(s), grad(s)) ** ((n - 1) / 2)
    momentum_balance = inner(u + 2 * A * P**n / (n + 2) * S_n * grad(s), v) * dx

    # The mass flux is the product of the thickness and velocity
    mass_flux = inner(h * u - q, r) * dx

    # The variational form is the sum of the flux definition and mass and
    # momentum balance
    return mass_balance + momentum_balance + mass_flux


def embed(F, u_, t, dt, tableau, bcs):
    # Create the form and RK stages
    old_form, old_stages, old_bcs, _, _ = irksome.getForm(
        F, tableau, t, dt, u_, bcs=bcs
    )
    old_test_fns = firedrake.split(old_form.arguments()[0])

    # Get the function space `Q` where the solution lives, the space `S` where
    # the stages live, and create a new function space `Z = Q x S` for both
    n = len(u_.subfunctions)
    Q = u_.function_space()
    S = old_stages.function_space()
    Z = Q * S

    # Create a function in the joint solution x stage space and extract the
    # solution and stage variables
    zs = firedrake.Function(Z)
    soln_stages = firedrake.split(zs)
    soln_stages_test = firedrake.TestFunctions(Z)

    u, new_stages = soln_stages[0:n], soln_stages[n:]
    v, new_test_fns = soln_stages_test[0:n], soln_stages_test[n:]

    # Create a new form by mapping the old stages and test functions to the new
    stage_dict = dict(zip(firedrake.split(old_stages), new_stages))
    test_fn_dict = dict(zip(old_test_fns, new_test_fns))
    new_form = firedrake.replace(old_form, {**stage_dict, **test_fn_dict})

    β = tableau.b
    u_ = firedrake.split(u_)
    s = np.array(new_stages, dtype=object).reshape((-1, n))
    du_dt = β @ s

    soln_form = sum(inner(u[k] - (u_[k] + dt * du_dt[k]), v[k]) * dx for k in range(n))
    form = soln_form + new_form

    return zs, form
