import matplotlib
import numpy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol
from decimal import Decimal
import scipy.integrate as integrate


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
origin_g1_der = 3 * pow(np.e, 2)
approximation_errors = []
for value in g1_der_values:
    approximation_errors.append(abs(value - origin_g1_der))


# matplotlib.use('TkAgg')
# plt.loglog(h_values, approximation_errors, label='Approximation Error')
# plt.xlabel('x')
# plt.ylabel('f(x')
# plt.legend()


# plt.show()


def composite_simpson(a: float, b: float, number_of_points: int, f):
    h = (b - a) / number_of_points
    k1 = 0
    k2 = 0
    for i in range(1, number_of_points, 2):
        k1 += f(a + i * h)
        k2 += f(a + (i + 1) * h)
    return h / 3 * (f(a) + 4 * k1 + 2 * k2)


n_values = np.arange(3, 9999, 10)
origin_integral_value = (9 * pow(np.pi, 2) - 4) / 27
simpson_error_values = []
for i in n_values:
    simpson_error_values.append(abs(composite_simpson(0, np.pi, i, g2) - origin_integral_value))


# plt.loglog(n_values, simpson_error_values, label='Approximation Error Simpson')
# plt.xlabel('x')
# plt.ylabel('f(x')
# plt.legend()


# plt.show()

def gauss_func(x):
    return (63 * x ** 5 - 70 * x ** 3 + 15 * x) / 8


def mul_func(x, x_values):
    for i in range(0, 5):
        mul = 1
        for j in range(0,5):
            if i != j:
                mul *= (x-x_values[j])/(x_values[i]-x_values[j])


def gauss_quad(x):
    x_arg = Symbol('x')
    x_values = solve('(63 * x ** 5 - 70 * x ** 3 + 15 * x) / 8', x_arg)
    c = []



print(gauss_quad(2))
