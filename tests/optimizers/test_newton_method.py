import unittest

import numpy as np

from coptim.functions.brown import Brown
from coptim.functions.rosenbrock import Rosenbrock
from coptim.optimizers.newton_method import NewtonMethod


class TestNewtonMethod(unittest.TestCase):

    def test_newton_method_with_rosenbrock_objective(self):
        starting_point = np.array([1.2, 1])
        rho = 1e-8
        p = 2.1
        k_max = 200
        beta = 0.5
        sigma = 1e-4
        epsilon = 1e-6
        objective = Rosenbrock()

        optimizer = NewtonMethod()

        optimizer.optimize(x_0=starting_point,
                           rho=rho,
                           p=p,
                           k_max=k_max,
                           beta=beta,
                           sigma=sigma,
                           epsilon=epsilon,
                           func=objective)

        self.assertEqual(optimizer.iterations, 8)

    def test_newton_method_with_brown_objective(self):
        starting_point = np.array([1.2, 1])
        rho = 1e-8
        p = 2.1
        k_max = 200
        beta = 0.5
        sigma = 1e-4
        epsilon = 1e-6
        objective = Brown()

        optimizer = NewtonMethod()

        optimizer.optimize(x_0=starting_point,
                           rho=rho,
                           p=p,
                           k_max=k_max,
                           beta=beta,
                           sigma=sigma,
                           epsilon=epsilon,
                           func=objective)

        self.assertEqual(optimizer.iterations, 11)
