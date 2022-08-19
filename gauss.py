import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from decimal import Decimal


def gauss_solve(matrix, vector):
    n = len(matrix[0]) 
    if n != len(vector): # matrix[0] is the first matrix line. what if other lines has wrong size? the same is for columns.
        raise ValueError("incorrect size") # definitely not all exceptions are covered
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
    return x # where are exceptions for system, which might not have a unique solution or has no solution at all?


a = np.array([[1, 2, 3, -1], [4, 1, 0, 3], [-1, 1, 0, 2], [0, 0, 0, 7]], float)
vector = np.array([1, 3, 3, 0])
print(gauss_solve(a, vector)) # wrong, there is no solution for the above system, but executed your code gives it
