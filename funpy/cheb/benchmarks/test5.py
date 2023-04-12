from fun.fun import Fun
from cheb.chebpy import chebtec
from cheb.chebpts import chebpts
import simplify
import numpy as np
import scipy

fun1 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
fun2 = chebtec(op=lambda x: np.ones_like(x), type='cheb')
fun3 = fun1 + fun2

print('fun1:', fun1.coeffs)
print('fun2:', fun2.coeffs)
print('fun3:', fun3.coeffs)



fun1 = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb')
fun2 = Fun(op=lambda x: np.cos(2 * np.pi * x), type='cheb')
fun3 = fun1 - fun2

print('\nDOUBLE TEST SUB')

fun1 = Fun(op=[lambda x: np.sin(2 * np.pi * x), lambda x: np.cos(2 * np.pi * x)], type='cheb')
fun2 = Fun(op=[lambda x: np.cos(2 * np.pi * x), lambda x: np.sin(4 * np.pi * x)], type='cheb')

print('fun1:', fun1.coeffs.flags.f_contiguous)
print('fun2:', fun2.coeffs.flags.f_contiguous)

print('SUB')
fun3 = fun1 - fun2


f = Fun(op=lambda x: np.sin(2 * np.pi * x), type='cheb', domain=[-1, 1])
g = np.ones_like(f)
print(g.coeffs[0] == 1)
