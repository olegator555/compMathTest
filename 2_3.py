import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

x = [-2, 2, 6]
y = [-2, 4, 5]
step = 4
point_num = len(x)
a = -2
b = 6


def spline_coeff() -> list:
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


l = spline_coeff()


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


points_count = 50
x_ax = [a + (b - a) / points_count * i for i in range(0, points_count)]
spline_values = []
for i in range(0, points_count):
    spline_values.append(spline(x_ax[i], x))

plt.plot(x_ax, spline_values, label='spline')
plt.xlabel('x')
plt.ylabel('f(x')
plt.legend()
plt.show()
