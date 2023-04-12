from fun.fun import Fun
import eval_lib
import numpy as np

""" Clenshaw for scalar equations """
def clenshaw_scalar(x, c):
    bk1 = np.zeros_like(x)
    bk2 = np.zeros_like(x)
    x = 2*x
    n = c.shape[0] - 1
    for k in range(n, 1, -2):
        bk2 = c[k] + x * bk1 - bk2
        bk1 = c[k-1] + x * bk2 - bk1

    if n & 1:
        tmp = bk1
        bk1 = c[1] + x * bk1 - bk2
        bk2 = tmp
    # make sure this returns a vector that is inline with what clenshaw_vector would return
    return np.expand_dims(c[0] + 0.5 * x * bk1 - bk2, axis=1)

""" Clenshaw for vector equations """
def clenshaw_vector(x, c):
    N = x.size
    x = np.tile(np.expand_dims(x, axis=1), (1, c.shape[1]))
    bk1 = np.zeros((N, c.shape[1]))
    bk2 = np.zeros((N, c.shape[1]))
    e = np.ones((N, 1))
    x = 2*x
    n = c.shape[0] - 1
    for k in range(n, 1, -2):
        bk2 = e * c[None, k, :]   + x * bk1 - bk2
        bk1 = e * c[None, k-1, :] + x * bk2 - bk1

    if n & 1:
        tmp = bk1
        bk1 = e * c[None, 1, :] + x * bk1 - bk2
        bk2 = tmp

    return e * c[None, 0, :] + 0.5 * x * bk1 - bk2

def clenshaw(x, coeffs):
    if len(coeffs.shape) == 1 or coeffs.shape[1] == 1:
        return clenshaw_scalar(x, coeffs)
    else:
        return clenshaw_vector(x, coeffs)

import time
import statistics
import random

if __name__ == '__main__':

    functions = clenshaw, eval_lib.clenshaw_scalar
    times = {f.__name__: [] for f in functions}

    N = 100
    k = 100

    fun = Fun(op=[lambda x: np.cos(k * np.pi * x / 2)], simplify=False)
    xs = np.linspace(-1, 1, 1000)

    tol = 1e-10
    for _ in range(N):
        fun = Fun(op=[lambda x: np.cos(k * np.pi * x / 2)])
        fun.prolong(1000)
        coeffs = fun.coeffs.squeeze()

        t0 = time.time()
        r1 = clenshaw(xs, coeffs)
        t1 = time.time()
        times[clenshaw.__name__].append((t1 - t0) * 1000)

        t0 = time.time()
        r2 = eval_lib.clenshaw_scalar(xs, coeffs)
        t1 = time.time()
        times[eval_lib.clenshaw_scalar.__name__].append((t1 - t0) * 1000)

        # check correctness
        assert np.all(np.isclose(r1, r2)), ''

    for name, numbers in times.items():
        print('FUNCTION:', name, 'Used', len(numbers), 'times')
        print('\tMEDIAN', statistics.median(numbers))
        print('\tMEAN  ', statistics.mean(numbers))
        print('\tSTDEV ', statistics.stdev(numbers))
