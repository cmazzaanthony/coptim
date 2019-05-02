import numpy as np

from coptim.optimizer import Optimizer


class GradientMethodExactMinimization(Optimizer):
    def __init__(self):
        # TODO: More metrics: vector of x's, objective values, etc.
        self.iterations = 0

    def funct(self, Q, c, x, gamma):
        return 0.5 * x.T.dot(Q).dot(x) + c + gamma

    def dfunct(self, Q, c, x):
        return Q.dot(x) + c

    def step_size(self, Q, c, x, d):
        g = self.dfunct(Q, c, x)
        return -1 * g.T.dot(d) / d.T.dot(Q).dot(d)

    def optimize(self, x_0, delta, epsilon):
        Q = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, delta]])
        c = np.array([1, 1, 1, 1])
        x = x_0
        while np.linalg.norm(self.dfunct(Q, c, x)) >= epsilon:
            descent_direction = -1 * self.dfunct(Q, c, x)

            step_size = self.step_size(Q, c, x, descent_direction)

            # update step
            x = x + step_size * descent_direction

            self.iterations += 1

        print('Final parameters are \n'
              'x => {x}\n'
              'iterations => {k}\n'
              'delta => {delta}\n'
              'Q => \n {Q}'.format(x=x,
                                   k=k,
                                   delta=delta,
                                   Q=Q))
