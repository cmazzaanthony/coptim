import unittest

import numpy as np

from src.functions.rosenbrock import Rosenbrock
from src.optimizers.gradient_method import GradientMethod


class TestProximalGradientMethod(unittest.TestCase):