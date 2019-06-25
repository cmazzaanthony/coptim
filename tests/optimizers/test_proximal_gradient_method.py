import unittest

import numpy as np

from src.functions.mse import MSE
from src.functions.l1 import L1
from src.optimizers.proximal_gradient_method import ProximalGradientMethod


class TestProximalGradientMethod(unittest.TestCase):

    def test_lasso(self):
        np.random.seed(42)

        n_samples, n_features = 50, 100
        X = np.random.randn(n_samples, n_features)

        idx = np.arange(n_features)
        coef = (-1) ** idx * np.exp(-idx / 10)
        coef[10:] = 0
        y = np.dot(X, coef)

        optimizer = ProximalGradientMethod()
        sfunc = MSE(X, y)
        nsfunc = L1()

        starting_point = np.zeros(n_features)
        step_size = 0.1

        x = optimizer.optimize(starting_point,
                               sfunc,
                               nsfunc,
                               step_size)
