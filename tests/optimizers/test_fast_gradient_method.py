import unittest

import numpy as np

from coptim.functions.rosenbrock import Rosenbrock
from coptim.optimizers.gradient_method import FastGradientMethod


class TestFastGradientMethod(unittest.TestCase):

    def test_fast_gradient_method_with_rosenbrock_objective(self):
        pass