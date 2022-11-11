import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from decimal import Decimal


def f(t, y):
    return 2 * y - t ** 2 + 1


def approx2(c, h, a, b):
    w = [c]
    n = int(abs(a - b) / h)
    print(f"n = ", n)
    for i in range(0, n):
        t = a + h * i
        print(f"t = ", t)
        w.append(w[i] + f(t, w[i]) + 1 * h ** 2)
    return w


print(approx2(1, 0.2, 0, 1))
