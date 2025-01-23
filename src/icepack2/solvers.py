import numpy as np
import firedrake
from firedrake import inner, dx, derivative, action

__all__ = [
    "ConstrainedOptimizationProblem",
    "NewtonSolver",
]

class ConstrainedOptimizationProblem:
    def __init__(self, L, z, H=None, bcs=None, form_compiler_parameters=None):
        self.objective = L
        self.solution = z
        self.hessian = H or derivative(derivative(L, z), z)
        self.bcs = bcs
        self.form_compiler_parameters = form_compiler_parameters


class NewtonSolver:
    def __init__(self, problem, **kwargs):
        self.problem = problem

        z = self.problem.solution
        Z = z.function_space()
        w = firedrake.Function(Z)
        self.search_direction = w

        L = self.problem.objective
        F = derivative(L, z)
        H = self.problem.hessian

        bcs = None
        if self.problem.bcs:
            bcs = firedrake.homogenize(self.problem.bcs)
        self.bcs = bcs
        problem = firedrake.LinearVariationalProblem(H, -F, w, bcs)
        solver = firedrake.LinearVariationalSolver(problem)
        self.search_direction_solver = solver

        self.step_length = firedrake.Constant(0.0)
        self.search_direction_solver.solve()

        self.armijo = kwargs.pop("armijo", 1e-4)
        self.tolerance = kwargs.pop("tolerance", 1e-8)

        φ = firedrake.TrialFunction(Z)
        ψ = firedrake.TestFunction(Z)
        M = kwargs.get("mass_matrix", inner(φ, ψ) * dx)

        F = derivative(L, z)
        F_t = firedrake.replace(F, {z: z + self.step_length * w})
        f = firedrake.Function(Z)
        problem = firedrake.LinearVariationalProblem(M, F_t, f, bcs=self.bcs)
        solver = firedrake.LinearVariationalSolver(problem)
        self.mass_matrix = M
        self.residual = f
        self.riesz_map = solver

    def assemble(self, *args, **kwargs):
        kwargs["form_compiler_parameters"] = self.problem.form_compiler_parameters
        return firedrake.assemble(*args, **kwargs)

    def line_search(self):
        L = self.problem.objective
        z = self.problem.solution
        w = self.search_direction
        t = self.step_length

        M = self.mass_matrix
        f = self.residual
        t.assign(0.0)
        self.riesz_map.solve()
        rs = [self.assemble(action(action(M, f), f))]

        t.assign(1.0)
        self.riesz_map.solve()
        rs.append(self.assemble(action(action(M, f), f)))

        ts = [0.0, 1.0]
        slope = 2 * rs[-1]
        while rs[-1] > rs[0] - self.armijo * float(t) * slope:
            t.assign(t / 2)
            self.riesz_map.solve()
            rs.append(self.assemble(action(action(M, f), f)))
            ts.append(float(t))

        return ts, rs

    def reinit(self):
        self.search_direction_solver.solve()
        self.step_length.assign(0.0)

    def step(self):
        ts, rs = self.line_search()
        z = self.problem.solution
        w = self.search_direction
        t = self.step_length
        t.assign(ts[-1])
        z.assign(z + t * w)

        self.search_direction_solver.solve()

    def solve(self):
        self.reinit()

        L = self.problem.objective
        z = self.problem.solution
        F = derivative(L, z)
        f = self.assemble(F, bcs=self.bcs).riesz_representation()
        residuals = [firedrake.norm(f)**2]
        while residuals[-1] > self.tolerance:
            self.step()
            f.assign(self.assemble(F, bcs=self.bcs).riesz_representation())
            residuals.append(firedrake.norm(f)**2)

        return np.array(residuals)
