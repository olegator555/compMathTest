import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math


# initial function data

def f(x):
    return np.exp(-pow(x, 2))


number_of_points = 5  # initial nodes number
a = -3
b = 3

x_grid_values = [a + (b - a) / number_of_points * i for i in range(0, number_of_points + 1)]


# spline function

def spline_coeff(x: list) -> list:
    if len(x) == 0:
        raise ValueError("Stub!")
    point_num = len(x) - 1
    step = x[1] - x[0]
    y = [f(element_x) for element_x in x]  # n+1
    a = np.zeros((point_num + 1, point_num + 1))
    a.itemset((0, 0), 1)  # ! здесь уравнение вида c_1 = 0. почемему коэффициент 4*step?
    a.itemset((point_num, point_num), 1)  # ! здесь уравнение вида c_{n+1} = 0, аналогично
    for i in range(1, point_num):
        a.itemset((i, i),
                  4)  # не верно, i-е уравнение для коэффициентов c_i имеет вид: step*c_i + 2*step*c_{i+1} + step*c_{i+2} = ...
        a.itemset(i, i - 1, 1)  # не верно
        a.itemset(i, i + 1, 1)  # не верно
    p = np.zeros(point_num + 1)
    p.itemset(0, 0)
    p.itemset(point_num, 0)
    for i in range(1, point_num - 1):  # ! разве -1?
        p.itemset(i, 3 * (y[i - 1] - 2 * y[i] + y[i + 1]) / step * 2)  # +
    c = np.linalg.solve(a, p)  # n+1

    # b = [(3 * (c[i + 1] - c[i]) / step - step * (2 * c[i] + c[i + 1])) / 3 for i in range(0, point_num)]
    b = [(y[i] - y[i - 1]) / step - step / 3 * (2 * c[i] + c[i + 1]) for i in range(1, point_num)]
    # d = [(c[i + 1] - c[i]) / (3 * step) for i in range(0, point_num)]
    d = [(c[i + 1] - c[i]) / (3 * step) for i in range(0, point_num - 1)]
    l = []
    l.append([y[i - 1] for i in (range(1, point_num))])  # размер массива = n-1, а коэффицентов n!
    l.append(b)
    l.append([c[i] for i in range(1, point_num)])
    l.append(d)
    return l


l = spline_coeff(x_grid_values)


def spline(x_value: float, origin_x: list):
    step = origin_x[1] - origin_x[0]
    ans = -1
    if a <= x_value <= b:
        for i in range(0, len(origin_x) - 2):
            if (x_value >= origin_x[i]) and (x_value < origin_x[i] + step):
                ans = i
            else:
                continue
            break

    # print(ans)
    return l[0][ans] + l[1][ans] * (x_value - origin_x[ans]) + l[2][ans] * ((x_value - origin_x[ans]) ** 2) + l[3][
        ans] * ((x_value - origin_x[ans]) ** 3)


# Lagrange polynomial

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


# interpolation eror

def x_values(point_num: int) -> list:
    return [a + (b - a) / point_num * i for i in range(0, point_num + 1)]


def max_approximation_error(number_of_points: int, a: float, b: float) -> float:  # create single error function
    points = [a + (b - a) / number_of_points * i for i in range(0, number_of_points + 1)]
    origin_y = [f(i) for i in points]
    l_y = L_n(origin_y, number_of_points)
    errors = [abs(points[i] - origin_y[i]) for i in range(0, number_of_points)]
    return max(errors)


# building plots

points_count = 200
x_ax = [a + (b - a) / points_count * i for i in range(0, points_count)]
f_initial = [f(x_element) for x_element in x_ax]
spline_values = []
for i in range(0, points_count):
    spline_values.append(spline(x_ax[i], x_grid_values))
lagrange_interpolation = [L_n(i, number_of_points) for i in x_ax]

plt.plot(x_ax, f_initial, label='initil f(x)')
plt.plot(x_ax, spline_values, label='spline')
plt.plot(x_ax, lagrange_interpolation, label='Lagrange interp.')

plt.xlabel('x')
plt.ylabel('f(x')
plt.legend()
plt.show()
