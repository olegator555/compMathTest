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
    for i in range(0, n + 1):
        if i != j:
            p1 *= x - x_i[i]
            p2 *= x_i[j] - x_i[i]
    return p1 / p2


def L_n(x, n):
    s = 0
    for i in range(0, n + 1):
        s += f(x) * l_i(x, n, i)
    return s


m = 100
x = [-3 + 6 / m * i for i in range(0, m + 1)]
y = [f(i) for i in x]

L = [L_n(i, n) for i in x]

plt.plot(x, y, label='initil f(x)')
plt.plot(x, L, label='Lagrange')

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

