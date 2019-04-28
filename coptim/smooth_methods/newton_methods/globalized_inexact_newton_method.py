import numpy as np
from numpy.linalg import norm
from numpy import power

"""
Cody Mazza-Anthony
260405012
ECSE 507
Hmwk 7
7.1 (Implementation of globalized inexact Newton's method)
"""


def func_sum(x):
    total = 0
    for i in range(1, len(x) + 1):
        total += i * (x[i - 1] - 1)

    return total


def func(x):
    total = 0
    for i in range(len(x)):
        Fi = x[i] - 1
        total += Fi ** 2

    return total + np.power(func_sum(x), 2) + np.power(func_sum(x), 4)


def func_gradient(x):
    fs = func_sum(x)
    grad = 2 * (x - 1)
    for i in range(1, len(x) + 1):
        grad[i - 1] = grad[i - 1] + 2 * i * fs + 4 * i * np.power(fs, 3)

    return grad


def func_hessian(x):
    fs = func_sum(x)
    hess_mat = np.zeros(shape=(len(x), len(x)))

    for i in range(1, len(x) + 1):
        for j in range(1, len(x) + 1):
            if i == j:
                hess_mat[i - 1, j - 1] = 2 + 2 * i * j + 12 * i * j * np.power(fs, 2)

            else:
                hess_mat[i - 1, j - 1] = 2 * i * j + 12 * i * j * np.power(fs, 2)

    return hess_mat


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def rosenbrock_gradient(x_p):
    new_x = np.array([
        2 * (-200 * x_p[0] * x_p[1] + 200 * np.power(x_p[0], 3) - 1 + x_p[0]),
        200 * (x_p[1] - x_p[0] ** 2)
    ])

    return new_x


def rosenbrock_hessian(x):
    dfdx1 = -400 * x[1] + 1200 * x[0] ** 2 + 2
    dfdx1dx2 = -400 * x[0]
    dfdx2dx1 = -400 * x[0]
    dfdx2 = 200

    return np.array([[dfdx1, dfdx1dx2], [dfdx2dx1, dfdx2]])


def choose_nk(x, gradient, k):
    c1 = 10e-2
    c2 = 1
    return min(c1 / (k + 1), c2 * norm(gradient(x)))


def armijo_rule(beta, sigma, x_p, d, gradient, f):
    i = 0
    inequality_satisfied = True
    while inequality_satisfied:
        if f(x_p + np.power(beta, i) * d) <= f(x_p) + np.power(beta, i) * sigma * gradient(x_p).dot(d):
            break

        i += 1

    return np.power(beta, i)


def conjugate_gradient_method(x0, eta, gradient, hessian, n):
    epsilon = eta * norm(gradient(x0))
    b = - gradient(x0)
    A = hessian(x0)
    g = A.dot(x0) - b
    d = - g
    k = 0
    x = x0
    while norm(g) > epsilon:
        t_k = np.power(norm(g), 2) / (np.dot(np.dot(d.T, A), d))
        x = x + t_k * d
        g_prev = g
        g = g_prev + t_k * A.dot(d)
        beta = np.power(norm(g), 2) / np.power(norm(g_prev), 2)
        d = -g + beta * d
        k += 1

    return False, x


def global_inexact_newton_method(x_0, rho, p, beta, sigma, epsilon, f, gradient, hessian, n):
    x = x_0
    k = 0
    while norm(gradient(x)) >= epsilon:
        n_k = choose_nk(x, gradient, k)

        condition, d = conjugate_gradient_method(x, n_k, gradient, hessian, n)

        if gradient(x).T.dot(d) > - rho * power(norm(d), p) or condition:
            d = -1 * gradient(x)

        # Determine t_k using Armijo Rule
        step_size = armijo_rule(beta, sigma, x, d, gradient, f)
        x = x + step_size * d

        k += 1

    # print("new x: {}".format(x))
    print("Number of iterations: {}".format(k))
    return k


if __name__ == '__main__':
    print('Rosenbrock')
    global_inexact_newton_method(x_0=np.array([-1.2, 1]),
                                 rho=1e-8,
                                 p=2.1,
                                 beta=0.5,
                                 sigma=1e-4,
                                 epsilon=1e-6,
                                 f=rosenbrock,
                                 gradient=rosenbrock_gradient,
                                 hessian=rosenbrock_hessian,
                                 n=2)

    print('Function with n = 10')
    n = 10
    x_0 = [1 - i/n for i in range(1, n + 1)]
    global_inexact_newton_method(x_0=np.array(x_0),
                                 rho=1e-8,
                                 p=2.1,
                                 beta=0.5,
                                 sigma=1e-4,
                                 epsilon=1e-6,
                                 f=func,
                                 gradient=func_gradient,
                                 hessian=func_hessian,
                                 n=n)

    print('Function with n = 100')
    n = 100
    x_0 = [1 - i / n for i in range(1, n + 1)]
    global_inexact_newton_method(x_0=np.array(x_0),
                                 rho=1e-8,
                                 p=2.1,
                                 beta=0.5,
                                 sigma=1e-4,
                                 epsilon=1e-6,
                                 f=func,
                                 gradient=func_gradient,
                                 hessian=func_hessian,
                                 n=n)
