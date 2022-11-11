import matplotlib.pyplot as plt
import numpy as np
import sympy
from sympy import *


def g1(x):
    return x * pow(np.e, x)


def g2(x):
    return x ** 2 * np.sin(3 * x)


def diff2(x, h, f):
    return (f(x + h / 2) - f(x - h / 2)) / h


g1_der_values = []
h_values = np.arange(10e-16, 1, 10e-2)
for i in h_values:
    g1_der_values.append(diff2(2, i, g1))
origin_g1_der = 3 * pow(np.e, 2)
approximation_errors = []
for value in g1_der_values:
    approximation_errors.append(abs(value - origin_g1_der))


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

plt.loglog(n_values, simpson_error_values, label='Approximation Error Simpson')
plt.xlabel('x')
plt.ylabel('f(x')
plt.legend()

plt.show()


def diff4(x, h, f):
    return (-f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) + f(x + 2 * h)) / (12 * h)


g1_der4_values = []
for i in h_values:
    g1_der4_values.append(diff4(2, i, g1))
approximation_errors4 = []
for value in g1_der4_values:
    approximation_errors4.append(abs(value - origin_g1_der))

h_values_o2 = [h_value ** 2 for h_value in h_values]
h_values_o4 = [h_value ** 4 for h_value in h_values]


def Lg5(x):
    return (63 * x ** 5 - 70 * x ** 3 + 15 * x) / 8


x_5 = [0, np.sqrt(5 - 2 * np.sqrt((10 / 7))) / (-3), np.sqrt(5 - 2 * np.sqrt((10 / 7))) / 3,
       np.sqrt(5 + 2 * np.sqrt((10 / 7))) / (-3),
       np.sqrt(5 + 2 * np.sqrt((10 / 7))) / 3]


def Lg0(x):
    return 1


def Lg1(x):
    return x


def Lg2(x):
    return (3 * x ** 2 - 1) / 2


x_2 = [-1 / (np.sqrt(3)), 1 / np.sqrt(3)]


def Lg3(x):
    return (5 * x ** 3 - 3 * x) / 2


x_3 = [0, -1 * np.sqrt(3 / 5), np.sqrt(3 / 5)]


def Lg4(x):
    return (35 * x ** 4 - 30 * x ** 2 + 3) / 8


x_4 = [-1 * np.sqrt(3 / 7 - (2 * np.sqrt(6 / 5)) / 7), np.sqrt(3 / 7 - (2 * np.sqrt(6 / 5)) / 7),
       -1 * np.sqrt(3 / 7 + (2 * np.sqrt(6 / 5)) / 7), np.sqrt(3 / 7 + (2 * np.sqrt(6 / 5)) / 7)]


def Lg6(x):
    return (231 * x ** 6 - 315 * x ** 4 + 105 * x ** 2 - 5) / 16


x_6 = [-0.23862, 0.23862, -0.66121, 0.66121, -0.93247, 0.93247]


def gauss_quad(f):
    c = []
    x = x_5
    x_arg = Symbol('x')
    for i in range(0, len(x)):
        mul = ""
        for j in range(0, len(x)):
            if i != j:
                mul += "(x-%s)/(%s-%s)*" % (x[j], x[i], x[j])
        mul = mul[:-1]
        ans = sympy.integrate(mul, (x_arg, -1, 1))
        c.append(ans)
    sum = 0
    for i in range(0, len(x)):
        sum += c[i] * f(x[i])
    return sum


print(gauss_quad(Lg5))

print("done!")
