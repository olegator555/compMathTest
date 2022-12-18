import numpy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def J(z):
    x = z[0]
    y = z[1]
    return np.array([[3 - 0.002 * y, -0.002 * x], [0.0006 * y, 0.0006 * x - 0.5]])


def f(z):
    x = z[0]
    y = z[1]
    return np.array([3 * x - 0.002 * x * y, 0.0006 * x * y - 0.5 * y])


def h(y, z, t, f):
    f_fix = f(y - t / np.linalg.norm(z) * z)
    g = np.dot(f_fix, f_fix)
    return g


def rk4(x_0, t_n, f, h):
    w = []  # np.zeros((t_n+1,2))
    w.append(x_0)
    t = [i * h for i in range(t_n + 1)]
    for i in range(1, t_n + 1):
        wi = w[i - 1]
        k1 = h * f(w[i - 1])
        k2 = h * f(w[i - 1] + k1 / 2)
        k3 = h * f(w[i - 1] + k2 / 2)
        k4 = h * f(w[i - 1] + k3)
        w.append(w[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
    return w


x_0array = [[200 * i, 200 * i] for i in range(1, 10)]

x_0t = x_0array[4]

rk1 = rk4([100, 100], 100, f, 0.1)
rk2 = rk4([830, 1499], 100, f, 0.1)
x1 = [rk1[i][0] for i in range(len(rk1))]
y1 = [rk1[i][1] for i in range(len(rk1))]
x2 = [rk2[i][0] for i in range(len(rk2))]
y2 = [rk2[i][1] for i in range(len(rk2))]

plt.plot(x1, y1, label='from (100,100)')
plt.plot(x2, y2, label='from (800,1000)')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()


# plt.show()

def newton(x_0, f, J):
    y1 = x_0
    slau_sol = np.linalg.solve(J(y1), f(y1))
    y2 = y1 - slau_sol
    n = 1
    while np.linalg.norm(y2 - y1, ord=np.inf) >= 10e-8:
        y1 = y2
        y2 = y1 - np.linalg.solve(J(y1), f(y1))
        n += 1
    return y2, n


print(newton(np.array([400, 300]), f, J))


def solve_t3(h, y, z, f):
    t = 1
    while h(y, z, t, f) >= h(y, z, 0, f):
        t /= 2
    return t


def gradient_descent(x_0, f, J):
    y1 = x_0
    z = np.matmul(np.matrix.transpose(J(y1)), f(y1))
    t3 = solve_t3(h, y1, z, f)
    t2 = t3 / 2
    a = (h(y1, z, 0, f) / (t2 * t3))
    b = (h(y1, z, t2, f) / (t2 * (t2 - t3)))
    c = (h(y1, z, t3, f) / (t3 * (t3 - t2)))
    t = (a * (t2 + t3) + b * t3 + c * t2) / (2 * (a + b + c))
    y2 = y1 - 2 * t * z / np.linalg.norm(z)
    n = 1
    while np.linalg.norm(y2 - y1, ord=np.inf) >= 10e-8:
        y1 = y2
        z = np.matmul(np.matrix.transpose(J(y1)), f(y1))
        t3 = solve_t3(h, y1, z, f)
        t2 = t3 / 2
        a = (h(y1, z, 0, f) / (t2 * t3))
        b = (h(y1, z, t2, f) / (t2 * (t2 - t3)))
        c = (h(y1, z, t3, f) / (t3 * (t3 - t2)))
        t = (a * (t2 + t3) + b * t3 + c * t2) / (2 * (a + b + c))
        y2 = y1 - 2 * t * z / np.linalg.norm(z)
        n += 1
    return y2, n


print(gradient_descent(np.array([1, 3]), f, J))
