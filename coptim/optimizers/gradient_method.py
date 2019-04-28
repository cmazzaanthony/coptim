import numpy as np

from coptim.optimizer import Optimizer


class GradientMethod(Optimizer):

    def optimize(self, x_0, func, beta, sigma, epsilon):
        x = x_0
        iterations = 0
        while self.stopping_criteria(x, func, epsilon):
            descent_direction = -1 * func.gradient(x)

            step_size = self.step_size(x,
                                       func,
                                       beta,
                                       descent_direction,
                                       sigma)

            # update step
            x = x + step_size * descent_direction
            iterations += 1

        return iterations

    def stopping_criteria(self, x, func, epsilon):
        return np.linalg.norm(func.gradient(x)) >= epsilon

    def step_size(self, x, func, beta, d, sigma):
        i = 0
        inequality_satisfied = True
        while inequality_satisfied:
            if func(x + np.power(beta, i) * d) <= func(x) + np.power(beta, i) * sigma * func.gradient(x).dot(d):
                break

            i += 1

        return np.power(beta, i)
