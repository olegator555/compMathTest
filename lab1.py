import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math


def f(x):
    return 1 / (1 + 25 * x ** 2)


number_of_points = 6  # initial nodes number
points_count = 60
a = -1
b = 1

x_grid_values = [a + (b - a) / number_of_points * i for i in range(0, number_of_points + 1)]


def l_i(x, n, j):
    p1 = 1
    p2 = 1
    for i in range(0, n + 1):
        if i != j:
            p1 *= x - x_grid_values[i]
            p2 *= x_grid_values[j] - x_grid_values[i]
    return p1 / p2


def L_n(x, n):
    s = 0
    for i in range(0, n + 1):
        s += f(x_grid_values[i]) * l_i(x, n, i)
    return s


x_ax = [a + (b - a) / points_count * i for i in range(0, points_count)]
f_initial = [f(x_element) for x_element in x_ax]

lagrange_interpolation = [L_n(i, number_of_points) for i in x_ax]

plt.plot(x_ax, f_initial, label='initil f(x)')
plt.plot(x_ax, lagrange_interpolation, label='Lagrange interp.')

plt.xlabel('x')
plt.ylabel('f(x')
plt.legend()
plt.show()
