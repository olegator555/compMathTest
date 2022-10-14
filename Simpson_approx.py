import numpy as np


def f(x):
    return np.cos(x) ** 2


def composite_simpson(a: float, b: float, number_of_points: int):
    h = (b - a) / number_of_points
    k1 = 0
    k2 = 0
    for i in range(1, number_of_points, 2):
        k1 += f(a + i*h)
        k2 += f(a + (i+1)*h)
    return h/3*(f(a) + 4*k1 + 2*k2)


print(composite_simpson(-1 / 3, 1 / 3, 3))


