from fun.fun import Fun, roots
import eval_lib
import numpy as np

import time
import statistics
import random

if __name__ == '__main__':

    functions = roots,
    times = {f.__name__: [] for f in functions}

    N = 100
    k = 25

    fun = Fun(op=[lambda x: np.cos(k * np.pi * x / 2)], simplify=False)
    xs = np.linspace(-1, 1, 1000)

    tol = 1e-10
    for _ in range(N):
        fun = Fun(op=[lambda x: np.cos(k * np.pi * x / 2)])
        fun.prolong(500)
        coeffs = fun.coeffs.squeeze()

        t0 = time.time()
        r = roots(fun)
        t1 = time.time()
        times[roots.__name__].append((t1 - t0) * 1000)

        # t0 = time.time()
        # r2 = eval_lib.clenshaw_scalar(xs, coeffs)
        # t1 = time.time()
        # times[eval_lib.clenshaw_scalar.__name__].append((t1 - t0) * 1000)

        # check correctness
        # assert np.all(np.isclose(r1, r2))

    print('r = ', r)
    print('r = ', len(r))

    for name, numbers in times.items():
        print('FUNCTION:', name, 'Used', len(numbers), 'times')
        print('\tMEDIAN', statistics.median(numbers))
        print('\tMEAN  ', statistics.mean(numbers))
        print('\tSTDEV ', statistics.stdev(numbers))
