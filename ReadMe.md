Funpy
=========

Extends the NumPy API to work with functions. The code and its structure take
inspiration from chebfun.

Goal
-------

Develop a flexible package allowing easy and precise operations with functions.
Ideally all functions of the NumPy API should be appropriately implemented.

In addition, it is highly desirable to implement solid collocation support to
solve integro-differential equations.

Future: Extensions to 2D and 3D.


Status
--------

1. The core features appear to work.
2. **Warning:** The implementation surely has bugs.
3. Some core features are unit-tested, but testing is not extensive enough.
4. The Sympy support is slow and a mess.


Functions
-------------------

Currently one dimensional functions defined on bounded intervals are supported.
Both non-periodic (Chebyshev series), and periodic (Fourier series)
functions are supported.

```
# Define a function having two components on the interval [0, 1]
fun = Fun(op=[lambda x: np.cos(np.pi * x), lambda x: -np.cos(np.pi * x)], domain=[0, 1])

# Compute the integral
int = np.sum(fun)

# Compute the derivative
dfun = np.diff(fun)

# And of course all arithmetic operations are implemented.
fun2 = fun1 + fun1
```

Collocation
-----------------

Collocation to compute the solutions of differential equations is available.
Working implementations are

1. Ultraspherical collocation for non-periodic functions.
2. Fourier spectral collocation for periodic functions.

Value collocations may be implemented in the future.


**Example:**

Operators are defined as below. The strings of the operators are interpreted
using sympy. If the operator is nonlinear sympy is used to symbolically compute
the required Frechet derivative.

```
op     = ChebOp(functions=['u', 'v'],
                parameters={'I': I, 'b': b, 'gamma': gamma, 'epsilon': epsilon, 'm': m, 'D': D, 'K': K},
                domain=[0, 1])

# The equation defining the operator
op.eqn = ['epsilon^2 * diff(u, x, 2) + (b + gamma * u^m / (1 + u^m)) * v - I * u',
          'D *         diff(v, x, 2) - (b + gamma * u^m / (1 + u^m)) * v + I * u']

# Boundary conditions
op.bcs = [lambda u, v: np.diff(u)(0), lambda u, v: np.diff(u)(1),
          lambda u, v: np.diff(v)(0), lambda u, v: np.diff(v)(1)]

# Additional constraints e.g. mass constraints
op.cts = ['0.5 * (K - int(u) - int(v))',
          '0.5 * (K - int(u) - int(v))']
```

Nonlinear solvers
----------------

Three nonlinear solvers are available:

1. Classical Newton
2. QNERR (ERRor-based Quasi-Newton algorithm) see P. Deuflhard 2011
3. NLEQ-ERR (ERRor-based Damped Newton algorithm) see P. Deuflhard 2011


Continuation
-----------------

Several methods of continuation are implemented:

1. Deflation available to discover new solutions.
2. Newton-Gauss continuation.
3. Pseudo-arclength continuation (does it still work?)
