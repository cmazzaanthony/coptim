import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

"""
Cody Mazza-Anthony
260405012
ECSE 507
Hmwk 6
6.4 (Implementing the globalized BFGS method)
"""

T = np.array([15, 25, 35, 45, 55, 65, 75, 85, 105, 185, 245, 305, 365])
Y = np.array([0.038, 0.085, 0.1, 0.103, 0.093, 0.095, 0.088, 0.08, 0.073, 0.05, 0.038, 0.028, 0.02])


def bateman(x, t_list=T):
    y_x_t = x[2] * (np.exp(-x[0] * t_list) - np.exp(-x[1] * t_list))
    f_raw = np.power(y_x_t - Y, 2)

    return 0.5 * np.sum(f_raw)


def bateman_gradient(x):
    y_x_t = x[2] * (np.exp(-x[0] * T) - np.exp(-x[1] * T))

    dx1 = 0
    dx2 = 0
    dx3 = 0
    for i in range(13):
        dx1 = dx1 + (x[2] * -T[i] * (np.exp(-x[0] * T[i]))) * (y_x_t[i] - Y[i])
        dx2 = dx2 + (x[2] * T[i] * np.exp(-x[1] * T[i])) * (y_x_t[i] - Y[i])
        dx3 = dx3 + (np.exp(-x[0] * T[i]) - np.exp(-x[1] * T[i])) * (y_x_t[i] - Y[i])

    return np.array([dx1, dx2, dx3])


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


def psi(t, sigma, x, d, func, gradient):
    return func(x + t * d) - \
           func(x) - \
           sigma * t * gradient(x).dot(d)


def wolfe_powell_rule_phase_B(a, b, rho, sigma, x, d, func, gradient):
    a_j = a
    b_j = b
    stop = False

    while not stop:
        t_j = a_j + (b_j - a_j) / 2

        if psi(t_j, sigma, x, d, func, gradient) >= 0:
            b_j = t_j

        if psi(t_j, sigma, x, d, func, gradient) < 0 and \
                gradient(x + t_j * d).dot(d) >= rho * gradient(x).dot(d):
            t = t_j
            stop = True

        if psi(t_j, sigma, x, d, func, gradient) < 0 and \
                gradient(x + t_j * d).dot(d) < rho * gradient(x).dot(d):
            a_j = t_j

    return t


def wolfe_powell_rule_phase_A(rho, sigma, x, d, func, gradient):
    gamma = 2
    t_i = 1
    stop = False
    i = 0

    while not stop:
        if psi(t_i, sigma, x, d, func, gradient) >= 0:
            a = 0
            b = t_i

            t_i = wolfe_powell_rule_phase_B(a, b, rho, sigma, x, d, func, gradient)

        if psi(t_i, sigma, x, d, func, gradient) < 0 and \
                gradient(x + t_i * d).dot(d) >= rho * gradient(x).dot(d):
            t = t_i
            stop = True

        if psi(t_i, sigma, x, d, func, gradient) < 0 and \
                gradient(x + t_i * d).dot(d) < rho * gradient(x).dot(d):
            i += 1
            t_i = gamma * t_i

    return t


def bfgs_method(x_0, H_0, rho, sigma, epsilon, func, gradient):
    x = x_0
    H = H_0
    k = 0
    while np.linalg.norm(gradient(x)) > epsilon:
        nabla_x = -gradient(x)

        inverse = inv(H)
        d = inverse.dot(nabla_x)

        step_size = wolfe_powell_rule_phase_A(rho=rho,
                                              sigma=sigma,
                                              x=x,
                                              d=d,
                                              func=func,
                                              gradient=gradient)

        # print("step size: {}".format(step_size))
        # print("gradient {}".format(nabla_x))
        x_new = x + step_size * d
        s = x_new - x
        y = gradient(x_new) - gradient(x)

        H = H + (np.outer(y, y.T) / np.dot(y.T, s)) - np.outer(H.dot(s), s.T).dot(H) / (np.dot(np.dot(s.T, H), s))

        # print("new H: {}".format(H))
        # print("new x: {}".format(x_new))
        x = x_new
        k += 1

    print("final x: {}".format(x_new))
    print("iterations needed {}".format(k))
    return k


if __name__ == '__main__':
    bfgs_method(x_0=np.array([-1.2, 1]),
                H_0=np.array([[1, 0],
                              [0, 1]]),
                rho=0.9,
                sigma=1e-4,
                epsilon=1e-6,
                func=rosenbrock,
                gradient=rosenbrock_gradient)

    bfgs_method(x_0=np.array([0.05, 0.1, 0.4]),
                H_0=np.array([[1, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]]),
                rho=0.9,
                sigma=1e-4,
                epsilon=1e-6,
                func=bateman,
                gradient=bateman_gradient)

    # Plot the bateman function
    x = np.array([0.00574278, 0.04440698, 0.14692522])
    t_plot = np.arange(365)
    y = [x[2] * (np.exp(-x[0] * t) - np.exp(-x[1] * t)) for t in t_plot]


    plt.plot(t_plot, y)
    plt.show()