import numpy as np

from src.optimizer import Optimizer


class InexactNewtonMethod(Optimizer):
    def __init__(self):
        # TODO: More metrics: vector of x's, objective values, etc.
        self.iterations = 0

    def choose_nk(self, x, func):
        c1 = 10e-2
        c2 = 1
        return min(c1 / (self.iterations + 1), c2 * np.linalg.norm(func.gradient(x)))

    def step_size(self, beta, sigma, x_p, d, func):
        i = 0
        inequality_satisfied = True
        while inequality_satisfied:
            if func.eval(x_p + np.power(beta, i) * d) <= func.eval(x_p) + np.power(beta, i) * sigma * func.gradient(
                    x_p).dot(d):
                break

            i += 1

        return np.power(beta, i)

    def conjugate_gradient_method(self, x0, eta, func):
        epsilon = eta * np.linalg.norm(func.gradient(x0))
        b = - func.gradient(x0)
        A = func.hessian(x0)
        g = A.dot(x0) - b
        d = - g
        k = 0
        x = x0
        while np.linalg.norm(g) > epsilon:
            t_k = np.power(np.linalg.norm(g), 2) / (np.dot(np.dot(d.T, A), d))
            x = x + t_k * d
            g_prev = g
            g = g_prev + t_k * A.dot(d)
            beta = np.power(np.linalg.norm(g), 2) / np.power(np.linalg.norm(g_prev), 2)
            d = -g + beta * d
            k += 1

        return False, x

    def optimize(self, x_0, func, beta, sigma, epsilon, n, rho, p):
        x = x_0
        while self.stopping_criteria(x, func, epsilon):
            n_k = self.choose_nk(x, func)

            condition, d = self.conjugate_gradient_method(x, n_k, func)

            if func.gradient(x).T.dot(d) > - rho * np.power(np.linalg.norm(d), p) or condition:
                d = -1 * func.gradient(x)

            # Determine t_k using Armijo Rule
            step_size = self.step_size(beta, sigma, x, d, func)
            x = x + step_size * d

            self.iterations += 1

        return x

    def stopping_criteria(self, x, func, epsilon):
        return np.linalg.norm(func.gradient(x)) >= epsilon
