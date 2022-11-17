import numpy as np


def lu(a, permute=False):
    n = len(a)
    u = np.zeros((n, n))
    l = np.zeros((n, n))
    p = np.ones((n, n))
    m = np.zeros((n, n))
    u = a
    for j in range(0, n):
        m_i = np.zeros((n, n))
        for i in range(0, n):
            m_i.itemset((i,i),1)
        for i in range(j + 1, n):
            m_i.itemset((i, j), -a[j, i] / a[j, j])
        u = np.matmul(u, m_i)
    return u


a = np.array([[1, 1, 0, -3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]])
print(lu(a))
