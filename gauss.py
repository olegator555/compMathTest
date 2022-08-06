import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from decimal import Decimal


def gauss_solve(matrix, vector):
    n = len(matrix[0])
    if n != len(vector):
        raise ValueError("incorrect size")
    for k in range(n - 1):
        for i in range(k + 1, n):
            if matrix[i, k] == 0:
                continue
            factor = matrix[k, k] / matrix[i, k]
            for j in range(k, n):
                matrix[i][j] = matrix[k, j] - matrix[i, j] * factor
            vector[i] = vector[k] - vector[i] * factor
    x = np.zeros(n, float)
    x[n - 1] = vector[n - 1] / matrix[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        sum_ax = 0
        for j in range(i + 1, n):
            sum_ax += matrix[i, j] * x[j]
        x[i] = (vector[i] - sum_ax) / matrix[i, i]
    return x


a = np.array([[1, 2, 3], [2, 2, 3], [3, 2, 3]], float)
vector = np.array([3, 2, 1])
print(gauss_solve(a, vector))
