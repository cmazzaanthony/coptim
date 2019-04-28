import numpy as np
from numpy.linalg import inv, LinAlgError

"""
Cody Mazza-Anthony
260405012
ECSE 507
Hmwk 5
5.4 (Implementing the globalized Newton Method)
"""


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def rosenbrock_gradient(x_p):
    new_x = np.array([
        2 * (-200 * x_p[0] * x_p[1] + 200 * np.power(x_p[0], 3) - 1 + x_p[0]),
        200 * (x_p[1] - x_p[0] ** 2)
    ])

    return new_x


def brown(x):
    return np.power(x[0] - 1e6, 2) + np.power((x[1] - (2 * 1e-6)), 2) + np.power((x[0] * x[1] - 2), 2)


def brown_gradient(x):
    return np.array([
        2 * x[0] * (x[1] ** 2 + 1) - 4 * (x[1] + 500000),
        2 * (x[0] * (x[0] * x[1] - 2) + x[1] - (1 / 500000))
    ])


def brown_hessian(x):
    dfdx1 = 2 + (2 * x[1] ** 2)
    dfdx1dx2 = 4 * x[0] * x[1] - 4
    dfdx2dx1 = 4 * x[0] * x[1] - 4
    dfdx2 = 2 + (2 * x[0] ** 2)

    return np.array([[dfdx1, dfdx1dx2], [dfdx2dx1, dfdx2]])


def armijo_rule(beta, sigma, x_p, d, gradient, f):
    i = 0
    inequality_satisfied = True
    while inequality_satisfied:
        # print("left side: {}".format(rosenbrock(x_p + np.power(beta, i) * d)))
        # print("right side: {}".format(rosenbrock(x_p) + np.power(beta, i) * sigma * rosenbrock_gradient(x_p).T.dot(d)))
        if f(x_p + np.power(beta, i) * d) <= f(x_p) + np.power(beta, i) * sigma * gradient(x_p).dot(d):
            break

        i += 1

    return np.power(beta, i)


def rosenbrock_hessian(x):
    dfdx1 = -400 * x[1] + 1200 * x[0] ** 2 + 2
    dfdx1dx2 = -400 * x[0]
    dfdx2dx1 = -400 * x[0]
    dfdx2 = 200

    return np.array([[dfdx1, dfdx1dx2], [dfdx2dx1, dfdx2]])


def newton_solvable(x, gradient, rho, p, hessian):
    inverse = inv(hessian(x))
    d = inverse.dot(-gradient)

    if np.linalg.cond(inverse) > 1e-12:
        if gradient.T.dot(d) > -rho * np.power(np.linalg.norm(d), p):
            return False
    else:
        return False

    return True


def global_newton_method(x_0, rho, p, k_max, beta, sigma, epsilon, f, gradient, hessian):
    x = x_0
    k = 0
    while np.linalg.norm(gradient(x)) >= epsilon or k > k_max:
        descent_direction = -1 * gradient(x)

        # update step
        if newton_solvable(x, descent_direction, rho, p, hessian):
            inverse = inv(hessian(x))
            d = inverse.dot(descent_direction)
        else:
            d = descent_direction

        step_size = armijo_rule(beta, sigma, x, d, gradient, f)
        x = x + step_size * d

        # print("new x: {}".format(x))
        k += 1

    print("Number of iterations: {}".format(k))
    return k


if __name__ == '__main__':
    global_newton_method(x_0=[1.2, 1],
                         rho=1e-8,
                         p=2.1,
                         k_max=200,
                         beta=0.5,
                         sigma=1e-4,
                         epsilon=1e-6,
                         f=rosenbrock,
                         gradient=rosenbrock_gradient,
                         hessian=rosenbrock_hessian)

    # Brown
    global_newton_method(x_0=[1, 1],
                         rho=1e-8,
                         p=2.1,
                         k_max=200,
                         beta=0.5,
                         sigma=1e-4,
                         epsilon=1e-6,
                         f=brown,
                         gradient=brown_gradient,
                         hessian=brown_hessian)
