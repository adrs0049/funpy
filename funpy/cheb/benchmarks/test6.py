from fun.fun import Fun
import cheb.ufuncs as ufuncs
import numpy as np

import time
import statistics
import random

if __name__ == '__main__':

    functions = ufuncs.power,
    times = {f.__name__: [] for f in functions}

    N = 1000
    k = 100

    fun = Fun(op=[lambda x: np.cos(k * np.pi * x / 2)], simplify=False)
    xs = np.linspace(-1, 1, 1000)
    #print(fun.values[0:10].T)

    tol = 1e-10
    for _ in range(N):
        fun = Fun(op=[lambda x: np.cos(k * np.pi * x / 2)])

        t0 = time.time()
        r2 = fun**2
        t1 = time.time()
        times[ufuncs.power.__name__].append((t1 - t0) * 1000)

    for name, numbers in times.items():
        print('FUNCTION:', name, 'Used', len(numbers), 'times')
        print('\tMEDIAN', statistics.median(numbers))
        print('\tMEAN  ', statistics.mean(numbers))
        print('\tSTDEV ', statistics.stdev(numbers))
