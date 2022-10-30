import numpy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from decimal import Decimal


def g1(x):
    return x * pow(np.e, x)


def g2(x):
    return x ** 2 * np.sin(3 * x)


def diff2(x, h, f):
    return (f(x + h / 2) - f(x - h / 2)) / h


g1_der_values = []
h_values = np.arange(10e-16, 1, 10e-4)
for i in h_values:
    g1_der_values.append(diff2(2, i, g1))
origin_g1_der = 3*pow(np.e, 2)
approximation_errors = []
for value in g1_der_values:
    approximation_errors.append(abs(value - origin_g1_der))


plt.plot(np.log(h_values), np.log(approximation_errors), label='Approximation Error')
plt.xlabel('x')
plt.ylabel('f(x')
plt.legend()
plt.show()



