import mpmath
import numpy
import numpy as np


def max_value_and_index(a):
    max = 0
    max_ind = 0
    for i in range(0, len(a)):
        if a[i] > max:
            max = a[i]
            max_ind = i
    return max_ind


def lu(a1, permute):
    a = a1
    # m_n = []
    if not permute:

        n = len(a)
        u = a
        # u = np.zeros((n, n))
        l = np.identity(n)
        # p = np.ones((n,n))
        # m = np.zeros((n, n))
        for j in range(0, n):
            m_j = np.identity(n)
            for i in range(j + 1, n):
                m_j.itemset((i, j), -a[i, j] / a[j, j])
                l.itemset((i, j), a[i, j] / a[j, j])  # plus, and these expressions are for the l matrix
            a = np.matmul(m_j, a)
            # m_n.append(m_i)
            u = np.matmul(m_j, u)
        # u = np.matmul(np.linalg.inv(l), a)
        # u = a
        # for i in range(len(m_n))
        # u = np.matmul(m_n[i], u)
        return [l, u]
    else:
        n = len(a)
        u = a
        # u = np.zeros((n, n))
        l = np.identity(n)
        p = np.identity(n)
        # m = np.zeros((n, n))
        for j in range(0, n):
            m_j = np.identity(n)
            for i in range(j + 1, n):
                if abs(a[j, j] - a[j - 1, j]) > 10e-10:
                    m_j.itemset((i, j), -a[j, i] / a[j, j])
                    l.itemset((i, j), a[j, i] / a[j, j])  # plus, and these expressions are for the l matrix

                else:
                    print('check')
                    max_ind = max_value_and_index(a[i][i:])
                    print(f"max_ind = ", max_ind)
                    tmp2 = a[i, j]
                    a[i, j] = a[max_ind, j]

                    a[max_ind, j] = tmp2

                    tmp2 = p[i, j]
                    p[i, j] = p[max_ind, j]
                    p[max_ind, j] = tmp2
                    m_j.itemset((i, j), -a[j, i] / a[j, j])
                    l.itemset((i, j), a[j, i] / a[j, j])
            # m_n.append(m_i)
            u = np.matmul(m_j, u)
        # u = np.matmul(np.linalg.inv(l), a)
        # u = a
        # for i in range(len(m_n))
        # u = np.matmul(m_n[i], u)
        return [l, u, p]


a = np.array([[1, 1, 0, -3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])
a1 = np.array([[1, 2, 0], [1, 2, 1], [0, 1, 2]])
# l = lu(a, False)[0]
# u = lu(a, False)[1]
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
print('l*u = ', '\n', np.matmul(l,u))

def solve(l, u, vec):
    n = len(l)
    b = np.matmul(l, vec)
    y = []
    x = []
    for k in range(0, n):
        sum = 0
        for i in range(0, k - 1):
            sum += l[k, i] * y[i]
        y.append(b[k] - sum)
    for k in range(0, n):
        sum = 0
        for i in range(k + 1, n):
            sum += u[k, i] * x[i]
        x.append(y[k] / u[k, k] - sum / u[k, k])
    return x


print('p = ', '\n', tmp1[2])
print('1st matrix, LU-decomposition: ')
print('a = ', '\n', a, ' = ')
print('l = ', '\n', l, ' *')
print('*u = ', '\n', u)
print('l*u = ', '\n', np.matmul(l,u))

print('2nd matrix, LUP-decomposition: ')
print('p*a = ', '\n', np.matmul(tmp1[2],a1), ' = ')
print('l = ', '\n', l1, ' *')
print('*u = ', '\n', u1)
print('l*u = ', '\n', np.matmul(l1,u1))

print('p = ', '\n', tmp1[2])