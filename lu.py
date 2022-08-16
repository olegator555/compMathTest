import numpy as np


def lu_decomposition(a):
    n = len(a)
    l = np.zeros([n, n], float)
    u = np.zeros([n, n], float)
    u[1, 1] = a[1, 1]
    for i in range(0, n):
        l[i, i] = 1
    for j in range(2, n):
        u[1, j] = a[1, j]
        l[j, 1] = a[j, 1] / u[1, 1]
    for i in range(2, n):
        sum = 0
        for k in range(1, i - 1):
            sum += l[i, k] * u[k, i]
        u[i, i] = a[i, i] - sum
        for j in range(i + 1, n):
            sum = 0
            for k in range(1, i - 1):
                sum += l[i, k] * u[k, j]
            u[i, j] = a[i, j] - sum
            sum = 0
            for k in range(1, i - 1):
                sum += l[j, k] * u[k, j]
            l[i, j] = 1 / u[i, i] * (a[j, i] - sum)

    print("l = ", l)
    print("u = ", u)


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


a = np.array([[7, 3, -11], [-6, 7, 10], [-11, 2, -2]], float)
vector = np.array([3, 2, 1])
lu(a)
lu_decomposition(a) # also cheched, a != l*, so something is wrong
