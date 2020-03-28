import unittest

import numpy as np

from coptim.functions.rosenbrock import Rosenbrock
from coptim.optimizers.gradient_method import GradientMethod


class TestGradientMethod(unittest.TestCase):

    def test_gradient_method_with_rosenbrock_objective(self):
        objective = Rosenbrock()
        starting_point = np.array([-1.2, 1])
        beta = 0.5
        sigma = 0.0001
        epsilon = 0.0001

        optimizer = GradientMethod()

        x = optimizer.optimize(starting_point,
                               objective,
                               beta,
                               sigma,
                               epsilon)

        self.assertEqual(optimizer.iterations, 8058)
        self.assertListEqual(list(x), [0.999920582198063, 0.999840696392382])
