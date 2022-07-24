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


def approximation_error(points_count: int, left_border: int, right_border: int) -> tuple:
    step = abs(left_border - right_border) / points_count
    x_values = [left_border + step * i for i in range(0, points_count)]
    origin_y = [origin_dy(x_value) for x_value in x_values]
    approx_y = [L_n(x_value, points_count) for x_value in x_values]
    errors_mid = [abs(approx_y[i][0] - origin_y[i]) for i in range(0, points_count)]
    errors_left = [abs(approx_y[i][1] - origin_y[i]) for i in range(0, points_count)]
    errors_right = [abs(approx_y[i][2] - origin_y[i]) for i in range(0, points_count)]
    max_mid_error = max(errors_mid)
    max_left_error = max(errors_left)
    max_right_error = max(errors_right)
    print(points_count)
    return max_mid_error, max_left_error, max_right_error


points = [i for i in range(1, n)]
errors_mid = [approximation_error(i, -3, 3)[0] for i in range(1, n)]
errors_left = [approximation_error(i, -3, 3)[1] for i in range(1, n)]
errors_right = [approximation_error(i, -3, 3)[2] for i in range(1, n)]

plt.plot(points, errors_mid, label='mid error')
plt.plot(points, errors_right, label='left error')
plt.plot(points, errors_left, label='right error')

plt.xlabel('x')
plt.ylabel('f(x')
plt.legend()

plt.show()
print("Processed!")
# div = max(L_n(x_i[i])) - f(x_i[i] for i in range(0, m))
