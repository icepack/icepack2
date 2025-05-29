# icepack, version 2

Don't use this unless I told you to


## What is this

I'm prototyping a potential new version of icepack.
Some of the choices of design and solution strategy have painted us into a corner when it comes to simulating evolving termini and for doing long time integration.
This repo is for testing proposed changes that will fix those problems.

**Current status**: We have achieved almost all of the goals that we set out originally (see below).
The next step is to incorporate this code into the main icepack repository without breaking existing workflows.


## What have we done

### Higher-order timestepping

So far, we've relied exclusively on implicit Euler for timestepping the mass balance equation.
We then use an operator splitting scheme to separate out the solution of the momentum balance equation.
We will want higher-order schemes if we want to do simulations on paleoclimate time scales.
But even if we used a higher-order scheme for the mass balance equation, we would still be stuck with first-order convergence because of the operator splitting.

This package uses the [irksome](https://firedrakeproject.github.io/Irksome/) library to solve the coupled mass and momentum balance equation.
The solution is *fully coupled*: we simultaneously solve the mass and momentum balance equations.
In addition to the implicit Euler scheme, we also tested using the L-stable Radau-IIA family.
These methods are L-stable, which we want for dissipative problems like glacier dynamics.

### Variational inequalities

The ice thickness needs to be positive.
Without special care, it can go negative in the ablation zone.
So far, we have solved this by clamping the thickness from below at every timestep.

Rather than clamp the ice thickness, we can instead formulate the problem as a variational inequality with explicit bounds constraints.
Irksome has added the capacity to do bounds-constrained problems, which we now use.
This guarantees positivity at each timestep.

And quite a bit more too.
If we're using higher-order Runge-Kutta methods, we might also want positivity of the solution at each Runge-Kutta stage.
Irksome does even better than that.
The Radau-IIA scheme is a collocation method in time.
It is equivalent to fitting a polynomial through the solution in $t$ and requiring the ODE to be exact at a set of collocation times within each interval.
The relationship between positivity of a polynomial and its coefficients is not direct.
But there's no reason that we need to use the monomial or the Lagrange basis.
We can instead expand this polynomial in the Bernstein basis.
If we require the coefficients of the polynomial in the Bernstein basis to be positive, then the polynomial is positive *everywhere in the interval*.
(The converse is not true, i.e. there are positive polynomials with negative Bernstein coefficients.
But this effect goes away under refinement, i.e. if we expand it on smaller sub-intervals, eventually all the coefficients will be positive.)

We have tested this with both the backward Euler and Radau-IIA schemes.

### Duality

The conventional approach to ice flow modeling is to solve a nonlinear elliptic differential equation, called the shallow stream approximation or SSA, for the ice velocity $u$.
This equation is ill-posed whenever the ice thickness can go to 0.
The orthodox solution is to use implicit interface tracking schemes like the level set method.

Here we instead use the dual or mixed form of SSA, which explicitly introduces the membrane and basal stresses as unknowns.
The dual form remains solvable even when the ice thickness is 0.
We have used this to simulate both iceberg calving from marine-terminating glaciers and the advance and retreat of land-terminating mountain glaciers.


## What is to be done

### Linearly-implicit schemes

Rather than do a full nonlinear solve for all the Runge-Kutta stages in each timestep, you can do a single iteration of Newton's method.
Asymptotically in $\delta t$ this is just as good as the fully implicit scheme while keeping the same good stability properties.
It's also much easier to implement and less prone to undiagnosable convergence failures.
These are also called Rosenbrock methods.

I don't know how this should interact with variational inequalities.
In principle you could solve a linear complementarity problem in each timestep instead of a nonlinear complementarity problem.

### Approximate linearization

You don't have to use the exact linearization with a linearly-implicit scheme.
In principle we could just use the diagonal blocks for each variable, which will cost only a little more than doing operator splitting.
This may or may not have the same order in $\delta t$, I have to see.
We might be able to work around some of the degeneracies that occur at zero strain rate or thickness by perturbing the linearization.

### Nonlinear elimination preconditioning

This basically uses the operator splitting scheme as a preconditioner for the monolithically-coupled problem.
For thermal problems it's supposed to be a big improvement.

### Hybrid models

The dual form of SSA can behave like SIA in a certain limit.
Is it really a hybrid model that can do all stress regimes?
This proposition requires proof...

### Frontal ablation

The ability to solve the momentum balance equation at zero thickness means we can simulate episodic calving events.
Just make the ice thickness zero over some region.
This approach handles position-based calving laws well.
It will not do rate-based calving laws, where the terminus retreats continuously at some specificed rate.
Rate-based calving laws or frontal ablation are (in principle?) are a sink term + a modification to the flux in the mass balance equation.
