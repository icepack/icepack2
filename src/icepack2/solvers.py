import firedrake

__all__ = ["try_regularized_solve"]

def try_regularized_solve(solver, state, parameter):
    state_initial = state.copy(deepcopy=True)
    converged = False
    while not converged:
        try:
            solver.solve()
            converged = True
        except firedrake.ConvergenceError as error:
            message = str(error)
            if "DIVERGED_MAX_IT" in message:
                parameter.assign(parameter / 2)
            elif "DIVERGED_DTOL" in message:
                state.assign(state_initial)
                parameter.assign(2 * parameter)

