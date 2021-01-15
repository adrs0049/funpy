from fun.fun import Fun
from cheb.chebpts import chebpts
import transform
import numpy as np
import scipy
from scipy.fftpack import dct, idct

def polyfit_py(sampled):
    """
    Compute Chebyshev coefficients for values located on Chebyshev points.
    sampled: array; first dimension is number of Chebyshev points
    """
    asampled = np.asanyarray(sampled)
    if len(asampled.shape) == 1:
        asampled = np.expand_dims(asampled, axis=1)

    if len(asampled) == 1:
        return np.copy(asampled)

    N = asampled.shape[0]
    coeffs = scipy.fftpack.dct(asampled, 1, axis=0, overwrite_x=False)
    coeffs /= (N-1)
    coeffs[0, :] /= 2.
    coeffs[-1, :] /= 2.
    # TODO: figure out why this is required
    coeffs[1::2] *= -1
    return coeffs

def polyval_py(chebcoeff):
    """ compute the interpolation values at chebyshev points. """
    if len(chebcoeff.shape) == 1:
        chebcoeff = np.expand_dims(chebcoeff, axis=1)

    N = chebcoeff.shape[0]
    if N == 1:
        return np.copy(chebcoeff)

    data = chebcoeff / 2
    data[0, :] *= 2
    data[N-1, :] *= 2
    # TODO: figure out why this is required
    data[1::2] *= -1
    dctdata = scipy.fftpack.idct(data, 1, axis=0, overwrite_x=True)

    return dctdata

import time
import statistics
import random

if __name__ == '__main__':

    functions = polyval_py, transform.polyval, polyfit_py, transform.polyfit
    #functions = polyfit_py, transform.polyfit
    times = {f.__name__: [] for f in functions}

    N = 1000
    k = 100

    #fun = Fun(op=[lambda x: x**5])
    xs = np.linspace(-1, 1, 100)
    xj, _, _, _ = chebpts(k)
    fv = np.expand_dims(xj**5, axis=1)
    fv = np.reshape(fv, fv.shape, order='F')

    tol = 1e-10
    for _ in range(N):
        t0 = time.time()
        r1 = polyfit_py(fv)
        t1 = time.time()
        times[polyfit_py.__name__].append((t1 - t0) * 1000)

        t0 = time.time()
        #print('fv:', fv.squeeze())
        r2 = transform.polyfit(fv)
        t1 = time.time()
        times[transform.polyfit.__name__].append((t1 - t0) * 1000)

        # check correctness
        assert np.all(np.isclose(r1, r2))

        # Revert
        t0 = time.time()
        c1 = polyval_py(r1)
        t1 = time.time()
        times[polyval_py.__name__].append((t1 - t0) * 1000)

        t0 = time.time()
        c2 = transform.polyval(r2)
        t1 = time.time()
        times[transform.polyval.__name__].append((t1 - t0) * 1000)

        # check correctness
        assert np.all(np.isclose(c1, c2))

    for name, numbers in times.items():
        print('FUNCTION:', name, 'Used', len(numbers), 'times')
        print('\tMEDIAN', statistics.median(numbers))
        print('\tMEAN  ', statistics.mean(numbers))
        print('\tSTDEV ', statistics.stdev(numbers))
