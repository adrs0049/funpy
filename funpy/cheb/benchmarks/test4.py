from fun.fun import Fun
from cheb.chebpts import chebpts
import simplify
import numpy as np
import scipy

def prolong_py(array, Nout):
    # If Nout < length(self) -> compressed by chopping
    # If Nout > length(self) -> coefficients are padded by zero
    Nin = array.shape[0]
    Ndiff = int(Nout - Nin)

    m = 1
    if len(array.shape) > 1:
        m = array.shape[1]

    if Ndiff == 0: # Do nothing
        return array

    if Ndiff > 0:  # pad with zeros
        z = np.zeros((Ndiff, m))
        array = np.vstack((array, z))

    if Ndiff < 0:
        m = max(Nout, 0)
        array = array[:m, :]

    return array

import time
import statistics
import random

if __name__ == '__main__':

    functions = prolong_py, simplify.prolong
    times = {f.__name__: [] for f in functions}

    N = 1000
    k = 100

    Nout = 150

    #fun = Fun(op=[lambda x: x**5])
    a = np.arange(k).reshape((100, 1), order='F').astype(np.double)

    for _ in range(N):
        t0 = time.time()
        r1 = prolong_py(a, Nout)
        t1 = time.time()
        times[prolong_py.__name__].append((t1 - t0) * 1000)

        #print(r1.flags.f_contiguous)

        t0 = time.time()
        r2 = simplify.prolong(a, Nout)
        t1 = time.time()
        times[simplify.prolong.__name__].append((t1 - t0) * 1000)

        #print(r2.flags.f_contiguous)

        # print('r1:', r1)
        # print('r2:', r2)

        # check correctness
        assert np.all(np.isclose(r1, r2))

    for name, numbers in times.items():
        print('FUNCTION:', name, 'Used', len(numbers), 'times')
        print('\tMEDIAN', statistics.median(numbers))
        print('\tMEAN  ', statistics.mean(numbers))
        print('\tSTDEV ', statistics.stdev(numbers))

    # speedup
    print('Speed Up: %.4f' % (statistics.mean(times[functions[0].__name__]) / statistics.mean(times[functions[1].__name__])))
