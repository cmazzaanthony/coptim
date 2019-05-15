import numpy as np

from coptim.optimizer import Optimizer


class NewtonMethod(Optimizer):

    def __init__(self):
        # TODO: More metrics: vector of x's, objective values, etc.
        self.iterations = 0

    def newton_solvable(self, x, gradient, rho, p, hessian):
        inverse = np.linalg.inv(hessian(x))
        d = inverse.dot(-gradient)

        if np.linalg.cond(inverse) > 1e-12:
            if gradient.T.dot(d) > -rho * np.power(np.linalg.norm(d), p):
                return False
        else:
            return False

        return True

    def optimize(self, x_0, rho, p, k_max, beta, sigma, epsilon, func):
        x = x_0
        while self.stopping_criteria(x, func, epsilon, k_max):
            descent_direction = -1 * func.gradient(x)

            # update step
            if self.newton_solvable(x, descent_direction, rho, p, func.hessian):
                inverse = np.linalg.inv(func.hessian(x))
                d = inverse.dot(descent_direction)
            else:
                d = descent_direction

            step_size = self.step_size(beta, sigma, x, d, func)
            x = x + step_size * d

            self.iterations += 1

    def stopping_criteria(self, x, func, epsilon, k_max):
        return np.linalg.norm(func.gradient(x)) >= epsilon or self.iterations > k_max

    def step_size(self, beta, sigma, x, d, func):
        """
        Armijo's Rule
        """
        i = 0
        inequality_satisfied = True
        while inequality_satisfied:
            if func.eval(x + np.power(beta, i) * d) <= func.eval(x) + np.power(beta, i) * sigma * func.gradient(x).dot(
                    d):
                break

            i += 1

        return np.power(beta, i)
