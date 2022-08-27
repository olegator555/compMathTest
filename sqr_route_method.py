import numpy
import numpy as np


def sign(x):
    if x == 0.0:
        return 0
    if x > 0.0:
        return 1
    else:
        return -1


def solve(matrix, vector):
    n = len(matrix)
    u = numpy.zeros((n + 1, n + 1))
    d = numpy.zeros((n + 1, n + 1))
    u[0, 0] = numpy.sqrt(abs(matrix[0, 0]))
    d[0, 0] = sign(matrix[0, 0])
    for j in range(1, n):
        u[0, j] = matrix[0, j] / (u[0, 0] * d[0, 0])
    for i in range(1, n):
        sum = 0
        for l in range(1, i - 1):
            sum += u[l, i] * u[l, i] * d[l, l]
        u[i, i] = numpy.sqrt(abs(matrix[i, i]) - sum)
        d[i][i] = numpy.sign(matrix[i][i] - sum)
        for j in range(i + 1, n):
            sum = 0
            for l in range(1, i - 1):
                sum += u[l, i] * u[l, j] * d[l, l]
            u[i, j] = (matrix[i, j] - sum) / (u[i, i] * d[i, i])
    y = numpy.array(vector)
    m = n + 1
    y[0] = vector[0] / u[0, 0] * d[0, 0]
    for i in range(1, n):
        sum = 0
        for l in range(1, i - 1):
            sum += u[l, i] * y[l] * d[l, l]
        y[i] = (vector[0] - sum) / (u[i, i] * d[i, i])
    x = numpy.zeros(len(vector))
    x[n - 1] = y[n - 1] / u[n - 1, n - 1]
    for i in range(n - 1, 1):
        sum = 0
        for l in range(i + 1, n):
            sum += u[i, l] * x[l]
        x[i] = (y[i] - sum) / u[i, i]
    return x


a = numpy.array([[3, 1, 1],
                 [1, 3, 1],
                 [1, 1, 3]])
vector = numpy.array([8, 10, 12])
x = solve(a, vector)
print(x)
