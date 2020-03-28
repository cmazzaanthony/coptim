import numpy as np

from coptim.optimizer import Optimizer


def make_iterator(self, X, y, batch_size):
    for i in range(len(X)):
        u = min(i + batch_size, len(X))
        yield X[i:u], y[i:u]


class StochasticGradientMethod(Optimizer):

    def __init__(self):
        # TODO: More metrics: vector of x's, objective values, etc.
        self.iterations = 0

    def m_i(self, x, y, beta):
        return x @ beta - y

    def update_beta(self, beta, p, step_size, m, x_i, lam):
        return beta - step_size * (x_i.T @ m + lam * beta)

    def permute(self, X, y):
        idx = np.random.permutation([x for x in range(len(y))])
        return X[idx], y[idx]

    def optimize(self, beta_0, intercept, X, Y, step_size, num_epochs, lam, batch_size):
        beta = beta_0
        iterations = 0
        m = X.shape[0]
        p = X.shape[1] - 1
        error_per_epoch = 0

        losses = []
        for epoch in range(num_epochs):
            it = make_iterator(*self.permute(X, Y), batch_size)
            error_per_epoch = 0
            i = 0
            while True:
                try:
                    x, y = next(it)
                    f_value = self.m_i(x, y, beta)
                    error_per_epoch += 0.5 * (f_value.T @ f_value + lam * beta.T @ beta)
                    intercept = y.mean() - x.mean(0) @ beta  # intercept update
                    beta = self.update_beta(beta, p, step_size, f_value, x, lam)
                    i += len(y)
                except StopIteration as e:
                    error_per_epoch /= i
                    losses.append(error_per_epoch)
                    break
        return beta, np.stack(losses).reshape(len(losses))

    def stopping_criteria(self, **kwargs):
        pass
