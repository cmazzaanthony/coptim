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

    def optimize(self, x_0, func, beta, sigma, epsilon):
        x = x_0
        while np.linalg.norm(func.gradient(x)) >= epsilon or k > k_max:
            descent_direction = -1 * func.gradient(x)

            # update step
            if newton_solvable(x, descent_direction, rho, p, hessian):
                inverse = np.linalg.inv(func.hessian(x))
                d = inverse.dot(descent_direction)
            else:
                d = descent_direction

            step_size = step_size(beta, sigma, x, d, func.gradient, func)
            x = x + step_size * d

            self.iterations += 1

    def stopping_criteria(self, x, func, epsilon):


    def step_size(self, beta, sigma, x, d, func):
        """
        Armijo's Rule
        """
        i = 0
        inequality_satisfied = True
        while inequality_satisfied:
            if func.eval(x + np.power(beta, i) * d) <= func.eval(x) + np.power(beta, i) * sigma * func.gradient(x).dot(d):
                break

            i += 1

        return np.power(beta, i)
