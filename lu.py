import numpy as np
import scipy as scipy


def lu_decomposition(a):
    n = len(a)
    l = np.zeros([n, n], float)
    u = np.zeros([n, n], float)
    u[1, 1] = a[1][1]
    for i in range(0, n):
        l[i, i] = 1
    for j in range(2, n):
        u[1, j] = a[1][j]
        l[j, 1] = a[j][1] / u[1, 1]
    for i in range(2, n):
        sum = 0
        for k in range(1, i - 1):
            sum += l[i, k] * u[k, i]
        u[i, i] = a[i][i] - sum
        for j in range(i + 1, n):
            sum = 0
            for k in range(1, i - 1):
                sum += l[i, k] * u[k, j]
            u[i, j] = a[i][j] - sum
            sum = 0
            for k in range(1, i - 1):
                sum += l[j, k] * u[k, j]
            l[i, j] = 1 / u[i, i] * (a[j][i] - sum)

    print("l = ", l)
    print("u = ", u)
    return l, u


def lu(a):
    n = len(a)
    u = a
    l = np.zeros([n, n], float)
    for i in range(0, n - 1):
        for j in range(i, n - 1):
            l[j][i] = u[j][i] / u[i][i]
    for k in range(1, n - 1):
        for i in range(k - 1, n - 1):
            for j in range(i, n - 1):
                l[j][i] = u[j][i] / u[i][i]

        for i in range(k, n - 1):
            for j in range(k - 1, n - 1):
                u[i][j] = u[i][j] - l[i][k - 1] * u[k - 1][j]

    print(l)
    print(u)
    return l, u


def lu1(a):
    u = np.zeros((len(a), len(a)))
    l = np.zeros((len(a), len(a)))
    m = np.zeros((len(a), len(a)))
    for i in range(0, len(a)):
        m.itemset(i, i, 1)
    for i in range(0, len(a)):
        for j in range(0, len(a)):
            if abs(a[j][i]) > 10e-10 and i > j:
                m.itemset((i, j), -(a[i][j] / a[j][i]))

    u = np.multiply(a, m)
    # l = np.linalg.inv(m)
    return u


a = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])
wrong = np.array(lu1(a))
print("lu processed")
correct = np.array(scipy.linalg.lu(np.array(a), permute_l=False))
print("done")
