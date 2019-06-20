import unittest

import numpy as np

from src.functions.func1 import Func1
from src.functions.rosenbrock import Rosenbrock
from src.optimizers.newton_method_inexact_minimization import InexactNewtonMethod


class TestNewtonInexactMethod(unittest.TestCase):

    def test_newton_inexact_method_with_rosenbrock_objective(self):
        objective = Rosenbrock()
        starting_point = np.array([-1.2, 1])
        rho = 1e-8
        p = 2.1
        beta = 0.5
        sigma = 1e-4
        epsilon = 1e-6
        n = 2

        optimizer = InexactNewtonMethod()

        x = optimizer.optimize(starting_point,
                               objective,
                               beta,
                               sigma,
                               epsilon,
                               n,
                               rho,
                               p)

        self.assertEqual(optimizer.iterations, 21)
        self.assertListEqual(list(x), [0.9999999999402008, 0.9999999998788666])

    def test_newton_inexact_method_with_n_10_objective(self):
        objective = Func1()
        n = 10
        starting_point = np.array([1 - i / n for i in range(1, n + 1)])
        rho = 1e-8
        p = 2.1
        beta = 0.5
        sigma = 1e-4
        epsilon = 1e-6

        optimizer = InexactNewtonMethod()

        x = optimizer.optimize(starting_point,
                               objective,
                               beta,
                               sigma,
                               epsilon,
                               n,
                               rho,
                               p)

        self.assertEqual(optimizer.iterations, 14)

    def test_newton_inexact_method_with_n_100_objective(self):
        objective = Func1()
        n = 100
        starting_point = np.array([1 - i / n for i in range(1, n + 1)])
        rho = 1e-8
        p = 2.1
        beta = 0.5
        sigma = 1e-4
        epsilon = 1e-6

        optimizer = InexactNewtonMethod()

        x = optimizer.optimize(starting_point,
                               objective,
                               beta,
                               sigma,
                               epsilon,
                               n,
                               rho,
                               p)

        self.assertEqual(optimizer.iterations, 21)
