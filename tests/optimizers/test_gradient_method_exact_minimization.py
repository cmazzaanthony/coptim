import unittest

import numpy as np

from coptim.functions.rosenbrock import Rosenbrock
from coptim.optimizers.gradient_method import GradientMethod


class TestGradientMethodExactMinimization(unittest.TestCase):

    def test_inexact_gradient_method(self):
        pass
