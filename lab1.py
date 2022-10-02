import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 1 / (1 + 25 * x ** 2)


def x_grid_ch(n):
    ch = [-1]
    for i in range(1, n):
        ch.append(np.cos((2 * i - 1) / (2 * (n - 1)) * np.pi))
    ch.append(1)
    ch.sort()
    return ch


number_of_points = 5
points_count = number_of_points * 10
a = -1
b = 1

x_grid_values = [a + (b - a) / number_of_points * i for i in range(0, number_of_points + 1)]
x_grid_ch_values = x_grid_ch(number_of_points)


def l_i(x, n, j):
    p1 = 1
    p2 = 1
    for i in range(0, n + 1):
        if i != j:
            p1 *= x - x_grid_ch_values[i]
            p2 *= x_grid_ch_values[j] - x_grid_ch_values[i]
    return p1 / p2


def L_n(x, n):
    s = 0
    for i in range(0, n + 1):
        s += f(x_grid_ch_values[i]) * l_i(x, n, i)
    return s


x_ax = [a + (b - a) / points_count * i for i in range(0, points_count + 1)]
f_initial = [f(x_element) for x_element in x_ax]

lagrange_interpolation = [L_n(i, number_of_points) for i in x_ax]

plt.plot(x_ax, f_initial, label='initil f(x)')
plt.plot(x_ax, lagrange_interpolation, label='Lagrange - 5 points.')

plt.xlabel('x')
plt.ylabel('f(x')
plt.legend()
plt.show()
