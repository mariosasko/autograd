from .vector import Vector
from .vector import exp


def sigmoid(x):
    return 1 / (1 + exp(-x))


def abs_grad(x):
    return Vector([1 if v > 0 else 0 if v == 0 else -1 for v in x])