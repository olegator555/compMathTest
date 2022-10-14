import numpy as np


def f(x):
    return np.cos(x) ** 2


def composite_simpson(a: float, b: float, number_of_points: int):
    iter_count = 0
    h = (b - a) / number_of_points
    k1 = 0
    k2 = 0
    for i in range(1, number_of_points, 2):
        k1 += f(a + i * h)
        k2 += f(a + (i + 1) * h)
        iter_count += 1
    print(iter_count)
    return h / 3 * (f(a) + 4 * k1 + 2 * k2)


print(composite_simpson(-1 / 3, 1 / 3, 9))

print(0.6425183294879154/0.0446617792659777)

def composite_simpson_eps(a: float, b: float, eps):
    n = 1
    approx_1 = composite_simpson(a, b, n)
    approx_2 = eps * 100
    while abs(approx_2 - approx_1) > eps:
        approx_2 = approx_1
        n *= 2
        approx_1 = composite_simpson(a, b, n)
    print(f"n = ", n)
    return approx_1


print(composite_simpson_eps(-1 / 3, 1 / 3, 10e-8))
