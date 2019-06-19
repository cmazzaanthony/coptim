import numpy as np

from src.optimizer import Optimizer


class GradientMethodExactMinimization(Optimizer):
    def __init__(self):
        # TODO: More metrics: vector of x's, objective values, etc.
        self.iterations = 0

    def step_size(self, Q, c, x, d, func):
        g = func.gradient(Q, c, x)
        return -1 * g.T.dot(d) / d.T.dot(Q).dot(d)

    def optimize(self, x_0, func, delta, epsilon):
        Q = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, delta]])
        c = np.array([1, 1, 1, 1])
        x = x_0
        while self.stopping_criteria(x, Q, c, func, epsilon):
            descent_direction = -1 * func.gradient(Q, c, x)

            step_size = self.step_size(Q, c, x, descent_direction, func)

            # update step
            x = x + step_size * descent_direction

            self.iterations += 1

        return x

    def stopping_criteria(self, x, Q, c, func, epsilon):
        return np.linalg.norm(func.gradient(Q, c, x)) >= epsilon
