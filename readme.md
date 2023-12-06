# icepack, version 2

Don't use this unless I told you to

### What is this

I'm prototyping a potential new version of icepack.
We've relied heavily on first-order timestepping schemes with operator splitting to handle multiple kinds of physics.
This may be painting us into a sad corner for harder problems like paleo-ice sheet simulation.
What I want to change:

**Higher-order timestepping.**
We've relied almost exclusively on implicit Euler.
If we want higher-order schemes, and we don't want to implement them ourselves, we could use the [irksome](https://firedrakeproject.github.io/Irksome/) library.
This package includes methods like Radau-IIA or Lobatto-IIIC, which are L-stable -- we want this for dissipative problems like glacier dynamics.

**Linearly-implicit schemes.**
Rather than do a full nonlinear solve for all the Runge-Kutta stages in each timestep, you can do a single iteration of Newton's method.
Asymptotically in $\delta t$ this is just as good as the fully implicit scheme while keeping the same good stability properties.
It's also much easier to implement and less prone to undiagnosable convergence failures.
These are also called Rosenbrock methods.

**Approximate linearization.**
You don't have to use the exact linearization with a linearly-implicit scheme.
In principle we could just use the diagonal blocks for each variable, which will cost only a little more than doing operator splitting.
This may or may not have the same order in $\delta t$, I have to see.
We might be able to work around some of the degeneracies that occur at zero strain rate or thickness by perturbing the linearization.

**Variational inequalities.**
Rather than clamp the ice thickness from below at every timestep to make sure it stays positive, we can instead formulate the problem as a variational inequality with explicit bounds constraints.
Ed Bueler says that this means using only piecewise linear basis functions.
Rob Kirby would probably say to try Bernstein.

**Mixed elements.**
I've tried using elements for the velocity and thickness that are stable for mixed Poisson and they seem to work better in the zero-thickness limit than using, say, CG(1) for both.
We might need to explicitly include the flux as an unknown (note to self, ask Colin about this).

**Duality.**
The dual form of SSA includes the SIA as a special case.
This opens up the possibility of making a simple hybrid model that describes both flow regimes.


### Discretization

Using Irksome, try the implicit Euler and Radau-IIA methods.
If we try to use the PETSc variational inequality solver, then we need to use either the semi-smooth Newton (VINEWTONSSLS) or active set (VINEWTONRSLS) solvers.
But if we want to use a linearly implicit method then we want to specify a nonlinear solver type of KSPONLY, which effectively does Rosenbrock for us.
If we use the VI solvers then we have to do a fully implicit problem and it's not obvious how to solve only a linear complementarity problem instead.
