import numpy as np

"""
Cody Mazza-Anthony
260405012 
"""


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def rosenbrock_gradient(x_p):
    new_x = np.array([
        2 * (-200 * x_p[0] * x_p[1] + 200 * np.power(x_p[0], 3) - 1 + x_p[0]),
        200 * (x_p[1] - x_p[0] ** 2)
    ])

    return new_x


def armijo_rule(beta, sigma, x_p, d, gradient, f):
    i = 0
    inequality_satisfied = True
    while inequality_satisfied:
        if f(x_p + np.power(beta, i) * d) <= f(x_p) + np.power(beta, i) * sigma * gradient(
                x_p).dot(d):
            break

        i += 1

    return np.power(beta, i)


def gradient_method_rosenbrock(x_0, beta, sigma, epsilon, f, gradient):
    x = x_0
    iterations = 0
    while np.linalg.norm(gradient(x)) >= epsilon:
        descent_direction = -1 * gradient(x)

        step_size = armijo_rule(beta, sigma, x, descent_direction, gradient, f)

        # update step
        x = x + step_size * descent_direction
        iterations += 1

    print("Number of iterations: {}".format(iterations))
    return iterations


if __name__ == '__main__':
    gradient_method_rosenbrock(x_0=[-1.2, 1],
                               beta=0.5,
                               sigma=0.0001,
                               epsilon=0.0001,
                               f=rosenbrock,
                               gradient=rosenbrock_gradient)
