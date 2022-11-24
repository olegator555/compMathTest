import mpmath
import numpy
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt


def swap(a, s1, s2):
    n = len(a)
    tmp = np.array(a[s1])
    for i in range(0, n):
        a.itemset((s1, i), a[s2, i])
        a.itemset((s2, i), tmp[i])
    return a


def index_of_max_element(a):
    max = 0
    max_ind = 0
    for i in range(0, len(a)):
        if a[i] > max:
            max = a[i]
            max_ind = i
    return max_ind


def lu(a1, permute):
    a = a1
    n = len(a)

    p = np.identity(n)

    if not permute:

        n = len(a)

        u = a
        l = np.identity(n)
        for j in range(0, n):
            m_j = np.identity(n)
            for i in range(j + 1, n):
                m_j.itemset((i, j), -a[i, j] / a[j, j])
                l.itemset((i, j), a[i, j] / a[j, j])
            a = np.matmul(m_j, a)
            u = np.matmul(m_j, u)
        return [l, u, p]
    else:
        u = a
        l = np.identity(n)
        for j in range(0, n):
            m_j = np.identity(n)
            for i in range(j + 1, n):
                if abs(a[j, j]) > 10e-10:
                    m_j.itemset((i, j), -a[i, j] / a[j, j])
                    l.itemset((i, j), a[i, j] / a[j, j])

                else:
                    print('check')
                    max_ind = index_of_max_element(a[:][j])
                    print(f"max_ind = ", max_ind)
                    a = swap(a, j, max_ind)
                    p = swap(p, j, max_ind)
                    m_j.itemset((i, j), -a[i, j] / a[j, j])
                    l.itemset((i, j), a[i, j] / a[j, j])
            a = np.matmul(m_j, a)
            u = np.matmul(m_j, u)
        return [np.matmul(np.matmul(p, l), np.linalg.inv(p)), np.matmul(p, u), p]


a = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])
vec = [4, 1, -3, 4]
a1 = np.array([[3, 1, -3], [6, 2, 5], [1, 4, -3]])
vec1 = [-16, 12, -39]
tmp = lu(a1, True)
l1 = tmp[0]
u1 = tmp[1]

tmp = lu(a, False)
l = tmp[0]
u = tmp[1]

tmp1 = lu(a1, True)
l1 = tmp1[0]
u1 = tmp1[1]

print('1st matrix, LU-decomposition: ')
print('a = ', '\n', a, ' = ')
print('l = ', '\n', l, ' *')
print('*u = ', '\n', u)
print('l*u = ', '\n', np.matmul(l, u))


def convert_to_p(piv):
    P = np.eye(3)
    for i, p in enumerate(piv):
        Q = np.eye(3, 3)
        q = Q[i, :].copy()
        Q[i, :] = Q[p, :]
        Q[p, :] = q
        P = np.dot(P, Q)
        return P


def convert_to_lu(l, u):
    n = len(l)
    lu = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if i > j:
                lu.itemset((i, j), l[i, j])
            if i <= j:
                lu.itemset((i, j), u[i, j])
    return lu


def solve(l, u, vec1, p):
    vec = np.matmul(vec1, p)
    n = len(vec)
    y = []
    for k in range(0, n):
        sum = 0
        for i in range(0, k):
            sum += l[k, i] * y[i]
            print(sum)
        y.append(vec[k] - sum)
    x = [0.0] * n
    for k in range(0, n):
        sum = 0
        for i in range(k + 1, n):
            sum += u[k, i] * x[i]
            print(sum)
        x[k] = y[k] / u[k, k] - sum / u[k, k]
    return x


print('p = ', '\n', tmp1[2])
print('1st matrix, LU-decomposition: ')

print('l = ', '\n', l, ' *')
print('*u = ', '\n', u)
print('l*u = ', '\n', np.matmul(l, u))

print('2nd matrix, LUP-decomposition: ')
print('p*a = ', '\n', np.matmul(tmp1[2], a1), ' = ')
print('l = ', '\n', l1, ' *')
print('*u = ', '\n', u1)
print('l*u = ', '\n', np.matmul(l1, u1))

print('p = ', '\n', tmp1[2])

# result = solve(l, u, vec, p)

# print("solve result", result)



print("done")