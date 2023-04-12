from vprof import runner

import time
import statistics
import random


def main():
    from fun.fun import Fun
    import numpy as np

    N = 100
    k = 100

    for _ in range(N):
        fun = Fun(op=[lambda x: np.cos(k * np.pi * x / 2)])
        r2 = fun**2

runner.run(main, 'cmph', host='localhost', port='8000')
