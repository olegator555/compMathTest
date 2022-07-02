import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from decimal import Decimal


def f(x):
    return np.exp(-pow(x, 2))


n = 100

x_i = [-3 + 6 / n * i for i in range(0, n + 1)]


def l_i(x, numb, j):
    p1 = 1
    p2 = 1
    for i in range(0, numb + 1):
        if i != j:
            p1 *= x - x_i[i]
            p2 *= x_i[j] - x_i[i]
    return p1 / p2


def L_n(x, numb):
    s = 0
    for i in range(0, numb + 1):
        s += f(x) * l_i(x, numb, i)
    return s


m = 100
x = [-3 + 6 / m * i for i in range(0, m + 1)]
y = [f(i) for i in x]

L = [L_n(i, n) for i in x]


def spline_approx(x: list) -> list:
    if len(x) == 0:
        raise ValueError("Stub!")
    point_num = len(x) - 1
    step = x[1] - x[0]
    y = [f(element_x) for element_x in x]
    a = np.zeros((point_num + 1, point_num + 1))
    a.itemset(0, 0, 4 * step)
    a.itemset(point_num, point_num, 4 * step)
    for i in range(1, point_num):
        a.itemset(i, i, 4 * step)
        if i != 0:
            a.itemset(i, i - 1, step)
        if i != 99:
            a.itemset(i, i + 1, step)
    p = np.zeros(point_num + 1)
    p.itemset(0, 0)
    p.itemset(point_num, 0)
    for i in range(1, point_num - 1):
        p.itemset(i, 3 * (y[i - 1] - 2 * y[i] + y[i + 1]) / step)
    c = np.linalg.solve(a, p)

    b = [(3 * (c[i + 1] - c[i]) - step * step * (2 * c[i] + c[i + 1])) / 3 for i in range(0, point_num)]
    d = [(c[i + 1] - c[i]) / (3 * step) for i in range(0, point_num)]

    spl = np.zeros(point_num + 1)
    for i in range(1, point_num):
        spl.itemset(i, y[i - 1] + b[i] * step + c[i] * step ** 2 + d[i] * step ** 3)
    return spl


def x_values(point_num: int) -> list:
    return [-3 + 6 / point_num * i for i in range(0, point_num + 1)]


points_count = 200
spline_x = x_values(points_count)
spline_y = [f(x_element) for x_element in spline_x]
spline = spline_approx(spline_x)


def max_approximation_error(original_values: list, approx_values: list) -> float:
    if len(original_values) != len(approx_values):
        raise IndexError("Stub!")
    errors = [original_values[i] - approx_values[i] for i in range(0, len(original_values))]
    return max(errors)


def error_function(point_num: int) -> float:
    x_points = x_values(point_num)
    y_points = [f(i) for i in x_points]
    spline_values = spline_approx(x_points)
    error = max_approximation_error(y_points, spline_values)
    return error


plt.plot(spline_x, spline_y, label='initil f(x)')
plt.plot(spline_x, spline, label='spline')
plt.xlabel('x')
plt.ylabel('f(x')
plt.legend()

n_values = [i for i in range(10, 1000)]
error_values = [error_function(i) for i in range(10, 1000)]
plt.plot(n_values, error_values, label="error")
plt.legend()
plt.show()

plt.show()
plt.savefig('./compiledSources/lag.png')
print("Stub!")
