import numpy as np
import firedrake
from firedrake import inner, dx
import irksome

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
