import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from decimal import Decimal


def f(x):
    return np.exp(-pow(x, 2))


n = 5

x_i = [-3 + 6 / n * i for i in range(0, n + 1)]
step = x_i[1] - x_i[0]


def origin_dy(x: float):
    return -2 * x * np.exp(-pow(x, 2))


def dy_dx(x_val: float, x_i: float):
    return (f(x_val + x_i / 2) - f(x_val - x_i / 2)) / x_i


def left_dy_dx(x_val: float, x_i: float):
    return (f(x_i) - f(x_val - x_i)) / x_i


def right_dy_dx(x_val: float, x_i: float):
    return (f(x_val + x_i) - f(x_i)) / x_i


def l_i(x, numb, j):
    p1 = 1
    p2 = 1
    for i in range(0, n + 1):
        if i != j:
            p1 *= x - x_i[i]
            p2 *= x_i[j] - x_i[i]
    return p1 / p2


def L_n(x, n):
    s = 0
    s_left = 0
    s_right = 0
    for i in range(0, n + 1):
        s += dy_dx(x, step) * l_i(x, n, i)
        s_left += left_dy_dx(x, step) * l_i(x, n, i)
        s_right += right_dy_dx(x, step) * l_i(x, n, i)
    return [s, s_left, s_right]


m = 100
x = [-3 + 6 / m * i for i in range(0, m + 1)]
y = [origin_dy(i) for i in x]

L = [L_n(i, n) for i in x]

plt.plot(x, y, label='initil f(x)')
plt.plot(x, [L[i][0] for i in range(0, len(L))], label='Lagrange_mid')
plt.plot(x, [L[i][1] for i in range(0, len(L))], label='Lagrange_left')
plt.plot(x, [L[i][2] for i in range(0, len(L))], label='Lagrange_rigth')

plt.xlabel('x')
plt.ylabel('f(x')
plt.legend()

plt.show()
plt.savefig('./compiledSources/lag.png')

a = np.zeros((m, m))
h = x_i[1] - x_i[0]
for i in range(1, m):
    a.itemset(i, i, 4 * h)
    if i != 0:
        a.itemset(i, i - 1, h)
    if i != 99:
        a.itemset(i, i + 1, h)
p = (y[i + 1] - 2 * y[i] + y[i - 1] for i in range(1, 99))

# div = max(L_n(x_i[i])) - f(x_i[i] for i in range(0, m))
