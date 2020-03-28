import unittest

import numpy as np

from coptim.functions.bateman import Bateman
from coptim.functions.rosenbrock import Rosenbrock
from coptim.optimizers.bfgs_method import BFGSMethod


class TestBFGSMethod(unittest.TestCase):

    def test_bfgs_method_with_rosenbrock_objective(self):
        objective = Rosenbrock()
        starting_point = np.array([-1.2, 1])
        H_0 = np.array([[1, 0],
                        [0, 1]])

        rho = 0.9
        sigma = 1e-4
        epsilon = 1e-6

        optimizer = BFGSMethod()
        x = optimizer.optimize(starting_point,
                               H_0,
                               rho,
                               sigma,
                               epsilon,
                               objective)

        self.assertEqual(optimizer.iterations, 34)

    def test_bfgs_method_with_bateman_objective(self):
        objective = Bateman()
        starting_point = np.array([0.05, 0.1, 0.4])
        H_0 = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
        rho = 0.9
        sigma = 1e-4
        epsilon = 1e-6

        optimizer = BFGSMethod()
        x = optimizer.optimize(starting_point,
                               H_0,
                               rho,
                               sigma,
                               epsilon,
                               objective)

        self.assertEqual(optimizer.iterations, 29)
