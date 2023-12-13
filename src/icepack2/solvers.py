import firedrake

__all__ = ["try_regularized_solve"]

def try_regularized_solve(solver, state, parameter, verbose=False):
    state_initial = state.copy(deepcopy=True)
    converged = False
    while not converged:
        try:
            if verbose:
                print(float(parameter))
            solver.solve()
            converged = True
        except firedrake.ConvergenceError as error:
            message = str(error)
            if "DIVERGED_MAX_IT" in message:
                state_initial.assign(state)
                parameter.assign(parameter / 2)
            elif "DIVERGED_DTOL" in message:
                state.assign(state_initial)
                parameter.assign(3 * parameter / 2)

