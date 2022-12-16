import numpy as np


def J(a):
    n = len(a)
    d = np.zeros((n, n))
    l = np.zeros((n, n))
    u = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                d.itemset((i, j), a[i, j])
            if i > j:
                l.itemset((i, j), -a[i, j])
            if i < j:
                u.itemset((i, j), -a[i, j])
    return d, l, u


def newton(x_0, f, J):


a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
x = J(a)
print(x)
