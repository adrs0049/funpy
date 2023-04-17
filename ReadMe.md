Funpy
=========

Extends the NumPy API to work with functions. The code and its structure are inspired 
by and translated from [Chebfun](https://www.chebfun.org/), and Approximation Theory and
Approximation Practice.

So far this has been a learning project, and a project used to teach and do a little bit of
research. Both the Chebyshev and Fourier spectral collocation methods work well enough to carry out
continuation for PDEs. The ultraspherical collocation method (introduced by Townsend and Olver) has
been used and tested the most.

But as with all software the current state leaves much to be desired.

Goal
-------

Develop a flexible package allowing easy and precise operations with functions.
Ideally all functions of the NumPy API should be appropriately implemented.

In addition, it is highly desirable to implement solid collocation support to
solve integro-differential equations.

Future: Clean up all the mess? Fix the many API issues that exist. Extensions to 2D and 3D?

Installation
--------

There are currently no working `setup.py`. Things will work when you place your folder in your
python path. The few cython extensions will be compiled automatically on import. The first import
may be slow for this reason.

### Dependencies

1. pylops
2. python 3.11 (due to strEnum)
3. Input/Output support uses HDF5 and requires h5py.


Status
--------

1. The core features appear to work.
2. **Warning:** Almost surely there are bugs lurking.
3. Some core features are unit-tested, but testing is not extensive enough.
4. The Sympy support is (still) slow. Can we use the new sympy C++ core?

ToDo
-------

1. Write a proper setup.py.
2. Better unit testing.
3. Object constructors are a mess, make more use of classmethods!
4. Improve module import, and have all the usual package things good to go i.e.\
   (`__doc__` strings) etc. 

5. Fix fallout from removing jax from colloc.
6. Improve `sympy` code, in particular fix the "compilation" of non-local equations.
7. Complete continuation code in this repository.
8. Clean-up the many bolted on features in the main operator class, collocators, and nonlinear solvers.
9. Lazy evaluations of arithmetic operations?
10. Check that the series truncation code works as expected, in particular is it scale invariant?
11. Lots of other things need fixing and improving!


Functions
-------------------

Currently one dimensional functions defined on bounded intervals both
non-periodic (Chebyshev series), and periodic (Fourier series) functions are
supported.

```
import numpy as np
import funpy as fp
from funpy import Fun
from funpy import plot
import matplotlib.pyplot as plt

# Define a function having two components on the interval [0, 1]
fun = Fun(op=[lambda x: np.cos(np.pi * x), lambda x: -np.cos(np.pi * x)], domain=[0, 1])

# Compute the define integral
int = np.sum(fun)

# Compute the anti-derivative
cfun = np.cumsum(fun)

# Compute the derivative
dfun = np.diff(fun)

# And of course all arithmetic operations are implemented.
fun2 = fun + fun

# Plot fun2
plot(fun2)
plt.show()
```

Periodic functions approximated using Fourier series are supported. A periodic function is
created by specifying a type when creating a Fun object. Periodic functions are currently
slower than their Chebyshev counter parts, since less of their operations are implemented
in cython. **Warning:** Periodic function support has many more rough edges, further
the memory layout appears to be less persistent in newer versions of numpy, which
confuses some of the cython extensions. This strongly suggest we finally need to change
the memory layout.

```
import numpy as np
import funpy as fp
from funpy import Fun

# Define a function having two components on the interval [0, 1]
fun = Fun(op=[lambda x: np.sin(x), lambda x: np.cos(x)], domain=[0, 2. * np.pi], type='trig')

# Compute the define integral
int = np.sum(fun)

# Compute the anti-derivative
cfun = np.cumsum(fun)

# Compute the derivative
dfun = np.diff(fun)

# And of course all arithmetic operations are implemented.
fun2 = fun + fun

```

Collocation
-----------------

Collocation to compute the solutions of differential equations is available.
Working implementations are:

1. Ultraspherical collocation for non-periodic functions.
2. Fourier spectral collocation for periodic functions.

Value collocations may be implemented in the future (some code exists).

**Example:**

An operator is defined as follows.

```
op     = ChebOp(functions=['u', 'v'],
                parameters={'I': I, 'b': b, 'gamma': gamma, 'epsilon': epsilon, 'm': m, 'D': D, 'K': K},
                domain=[0, 1])

# The equation defining the operator
op.eqn = ['epsilon^2 * diff(u, x, 2) + (b + gamma * u^m / (1 + u^m)) * v - I * u',
          'D *         diff(v, x, 2) - (b + gamma * u^m / (1 + u^m)) * v + I * u']

# Boundary conditions
# FIXME Currently same BCs applied at both sides of domain.
op.bcs = ['diff(u, x, 1)', 'diff(v, x, 1)']

# Additional constraints e.g. mass constraints
op.cts = ['0.5 * (K - int(u) - int(v))',
          '0.5 * (K - int(u) - int(v))']
```

**Technical information:**

1. The strings in `eqn` are interpreted using `sympy`, and thus must be valid
sympy expressions. For this reason parameters and function names
must be given to ChebOp upon construction (Future: Can we infer this somehow?).

2. The boundary conditions must be valid `funpy` operations. They are
   transformed into usable for by using the automatic differentiation framework
   `jax`. It would be really nice to get rid of this dependency.

3. Additional constraints can be specified in `cts`. For instance, mass
   constrains can be defined as in the example above. Note however, that
   currently **no** projections are applied. This may not be correct depending
   on the type of problem you are solving!

Nonlinear solvers
----------------

Three nonlinear solvers are available:

1. Classical Newton
2. QNERR (ERRor-based Quasi-Newton algorithm) see P. Deuflhard 2011
3. NLEQ-ERR (ERRor-based Damped Newton algorithm) see P. Deuflhard 2011
