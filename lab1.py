import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random


def f(x):
    return 1 / (1 + 25 * x ** 2)


def x_grid_ch(n):
    ch = [-1]
    for i in range(1, n):
        ch.append(np.cos((2 * i - 1) / (2 * (n - 1)) * np.pi))
    ch.append(1)
    ch.sort()
    return ch


number_of_points = 8
points_count = number_of_points * 10
a = -1
b = 1

x_grid_values = [a + (b - a) / number_of_points * i for i in range(0, number_of_points + 1)]
x_grid_ch_values = x_grid_ch(number_of_points)


def l_i(x, n, j, values: list):
    p1 = 1
    p2 = 1
    for i in range(0, n + 1):
        if i != j:
            p1 *= x - values[i]
            p2 *= values[j] - values[i]
    return p1 / p2


def L_n(x, n, values: list):
    s = 0
    for i in range(0, n + 1):
        s += f(values[i]) * l_i(x, n, i, values)
    return s


x_ax = [a + (b - a) / points_count * i for i in range(0, points_count + 1)]
f_initial = [f(x_element) for x_element in x_ax]

lagrange_interpolation = [L_n(i, number_of_points, x_grid_values) for i in x_ax]

matplotlib.use('TkAgg')
plt.plot(x_ax, f_initial, label='initil f(x)')
plt.plot(x_ax, lagrange_interpolation, label='Lagrange - 5 points.')

plt.xlabel('x')
plt.ylabel('f(x')
plt.legend()
plt.show()


def pade(n, m, x):
    sum1 = 0
    for j in range(0, m):
        sum1 += random.random() * (x ** j)
    sum2 = 0
    for k in range(1, n):
        sum2 += random.random() * (x ** k)
    return sum1 / (sum2 + 1)


x_points = [a + (b - a) / 100 * i for i in range(0, 100 + 1)]
x_points_ch = x_grid_ch(100)
rnd = []
for i in range(0, 100):
    n = random.randint(7, 15)
    m = random.randint(7, 15)
    rnd.append(pade(n, m, x_points[i]))

x_values = []
y_values = []
for i in range(0, number_of_points):
    x_values.append(x_points[16+i])
    y_values.append(rnd[16+i])

lagrange_interpolation_rnd = [L_n(i, number_of_points, x_grid_values) for i in x_values]
lagrange_interpolation_rnd_ch = [L_n(i, number_of_points, x_grid_ch_values) for i in x_values]
plt.plot(x_values, y_values, label='initil f(x)')
plt.plot(x_values, lagrange_interpolation_rnd, label='Lagrange uniform.')
plt.plot(x_values, lagrange_interpolation_rnd_ch, label='Lagrange ch.')


plt.xlabel('x')
plt.ylabel('f(x')
plt.legend()
plt.savefig("./2.png")
plt.show()

polynom = []
max_diffs = []
for N in range(1, 31):
    max_diff_uniform = 0
    max_diff_ch = 0
    for i in range(0, 100):
        uniform = L_n(x_points[i], N, x_points)
        ch = L_n(x_points[i], N, x_points_ch)
        diff_uniform = abs(x_points[i] - uniform)
        diff_ch = abs(x_points[i] - ch)
        polynom.append([x_points[i], uniform, ch, N, diff_uniform, diff_ch])
        if diff_uniform > max_diff_uniform:
            max_diff_uniform = diff_uniform
        if diff_ch > max_diff_ch:
            max_diff_ch = diff_ch
    max_diffs.append([N, max_diff_uniform, max_diff_ch])

n_points = [3, 8, 13, 18, 23]
diff_points = []
diff_points_ch = []
n_index = []
for point in n_points:
    n_index.append(point - 1)
    diff_points.append(max_diffs[point-1][1])
    diff_points_ch.append(max_diffs[point-1][2])

plt.semilogy(n_points, diff_points, label='unform diff f(x)')
plt.semilogy(n_points, diff_points_ch, label='ch diff f(x)')
plt.savefig("./3.png")

plt.xlabel('x')
plt.ylabel('f(x')
plt.legend()
plt.show()
print("Processed")
