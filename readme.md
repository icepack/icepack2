# icepack, version 2

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


### Dynamics

The cast of characters:

| symbol | description | units
|--------|-------------|------
| $h$ | ice thickness | length
| $s$ | surface elevation | length
| $b$ | bed elevation | length
| $u$| velocity | length / time
| $\rho$ | density | mass / length${}^3$
| $g$ | gravity | length / time${}^2$
| $n$ | Glen flow law exponent | 
| $m$ | Weertman sliding law exponent |
| $A$ | fluidity factor | pressure${}^{-n}$ time${}^{-1}$
| $K$ | sliding coefficient | time${}^{-1}$
| $\dot a$ | accumulation - melt | length / time

We assume that $s = b + h$ throughout.
We're using a sliding coefficient $K$, which increases when the bed is slippery, rather than a friction coefficient $C$, which decreases.
In the literature, $C$ is used more frequently than $K$.
The relation between the two is that $K = C^{-m}$.
The sliding coefficient is more convenient because $K = 0$ represents ice frozen to the bed.

We'll first use the shallow ice approximation, which assumes that the glacier has a small aspect ratio and that the $x-z$ and $y-z$ components of the strain rate tensor are much larger than the $x-x$, $x-y$, and $y-y$ components.
Using these approximations, we can solve exactly for the velocity:
$$u = -Kh|\nabla s|^{m - 1}\nabla s -\frac{2}{n + 2}A(\rho gh)^nh|\nabla s|^{n - 1}\nabla s$$
Together with the mass balance equation
$$\frac{\partial h}{\partial t} + \nabla\cdot hu = \dot a - \dot m$$
we have a closed system of equations for the thickness and velocity.


### Discretization

Using Irksome, try the implicit Euler and Radau-IIA methods.
If we try to use the PETSc variational inequality solver, then we need to use either the semi-smooth Newton (VINEWTONSSLS) or active set (VINEWTONRSLS) solvers.
But if we want to use a linearly implicit method then we want to specify a nonlinear solver type of KSPONLY, which effectively does Rosenbrock for us.
If we use the VI solvers then we have to do a fully implicit problem and it's not obvious how to solve only a linear complementarity problem instead.

Try the following spatial discretizations:

| thickness  | velocity | flux
| -----------|----------|------
| CG1        | CG1      |  
| CG1        | CG1 + B3 |
| CG1        | CG1      | CG1 + B3
| CG1        | CG1 + B3 | CG1 + B3
| BB2        | BB2 + B4 |
| BB2        | BB2      | BB2 + B4
| BB2        | BB2 + B4 | BB2 + B4

where CG = continuous Galerkin, B = bubble, BB = Bernstein-Bezier.


### Test cases

**Flat bed.**
The simplest case possible is a radially symmetric domain with a flat bed ($b = 0$), a frozen ice base ($K = 0$), and no accumulation or melt ($\dot a = 0$).
This makes $h = s$.
We'll start by taking
$$h = h_0 \max\{0, 1 - |x|^2 / r^2\}$$
for some initial thickness $h_0$ and radius $r$ which is less than the radius of the domain.
We should then see the glacier diffuse out.
This won't settle into any steady state.

**Growing a glacier, part 1.**
Using again a flat bed and a frozen ice base, take the initial ice thickness to be zero but use the accumulation rate
$$\dot a = \dot a_0(1 - |x|^2) / r^2$$
which will tend to grow a glacier in the center of the domain that melts off beyond a distance $r$ from the center of the domain.
This can settle into an equilibrium and if we're smart we should be able to calculate exactly what it is.

**Growing a glacier, part 2.**
Add sliding to the previous test case.
The glacier should be wider and flatter.

**Mountain peak.**
Use instead a bed elevation
$$b = b_0\exp(-|x|^2/r^2).$$