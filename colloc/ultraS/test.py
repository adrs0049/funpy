import numpy as np
import time
import statistics
import random

from funpy.colloc.ultraS.matrices import multmat
from funpy.fun import Fun

if __name__ == '__main__':

    functions = multmat
    lams = [0, 1, 2]
    times = {multmat.__name__ + '{0:d}'.format(lam): [] for lam in lams}

    N = 100
    k = 100

    # lam
    lam = 0

    tol = 1e-10
    for _ in range(N):
        fun = Fun(op=[lambda x: np.cos(k * np.pi * x / 2)])

        lam = 0
        t0 = time.time()
        r1 = multmat(k, fun, lam)
        t1 = time.time()
        times[multmat.__name__ + '{0:d}'.format(lam)].append((t1 - t0) * 1000)

        lam = 1
        t0 = time.time()
        r1 = multmat(k, fun, lam)
        t1 = time.time()
        times[multmat.__name__ + '{0:d}'.format(lam)].append((t1 - t0) * 1000)

        lam = 2
        t0 = time.time()
        r1 = multmat(k, fun, lam)
        t1 = time.time()
        times[multmat.__name__ + '{0:d}'.format(lam)].append((t1 - t0) * 1000)

    for name, numbers in times.items():
        print('FUNCTION:', name, 'Used', len(numbers), 'times')
        print('\tMEDIAN', statistics.median(numbers))
        print('\tMEAN  ', statistics.mean(numbers))
        print('\tSTDEV ', statistics.stdev(numbers))
