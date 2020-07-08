import math
from .node import Node
from .utils import abs_grad, sigmoid
from .vector import (abs, neg, log, log2, log10, log1p, exp, sin, cos, tan, sinh, cosh, tanh, 
                     pow, add, sub, mul, matmul, div, sum, 
                     lt, le, eq, ge, gt, all, any, fill)


EPSILON = 1e-8 


class UnaryOp(Node):
    fn = None
    fn_grad = None

    def __init__(self, parent):
        super().__init__([parent])
        self.parent = parent

    @property
    def value(self):
        return self.__class__.fn(self.parent.value)

    def partial_derivative(self, prev_grad, wrt):
        if wrt == self.parent:
            return prev_grad * self.__class__.fn_grad(self.parent.value)
        return 0


class Abs(UnaryOp):
    fn = abs
    fn_grad = abs_grad


class Neg(UnaryOp):
    fn = neg
    fn_grad = lambda x: fill(x, -1)


class Sin(UnaryOp):
    fn = sin
    fn_grad = cos


class Cos(UnaryOp):
    fn = cos
    fn_grad = lambda x: -sin(x)


class Tan(UnaryOp):
    fn = tan
    fn_grad = lambda x: cos(x) ** -2


class Log(UnaryOp):
    fn = log
    fn_grad = lambda x: 1 / (x + EPSILON)


class Log2(UnaryOp):
    fn = log2
    fn_grad = lambda x: 1 / ((x * math.log(2)) + EPSILON)


class Log10(UnaryOp):
    fn = log10
    fn_grad = lambda x: 1 / ((x * math.log(10)) + EPSILON)


class Log1p(UnaryOp):
    fn = log1p
    fn_grad = lambda x: 1 / ((1 + x) + EPSILON)


class Exp(UnaryOp):
    fn = fn_grad = exp


class ReLU(UnaryOp):
    fn = lambda x: x * (x > 0)
    fn_grad = lambda x: x > 0


class Sigmoid(UnaryOp):
    fn = sigmoid
    fn_grad = lambda x: sigmoid(x) * (1 - sigmoid(x)) 


class Sinh(UnaryOp):
    fn = sinh
    fn_grad = cosh


class Cosh(UnaryOp):
    fn = cosh
    fn_grad = lambda x: -sinh(x)


class Tanh(UnaryOp):
    fn = tanh
    fn_grad = lambda x: 1 - tanh(x) ** 2


class Sum(UnaryOp):
    fn = sum
    fn_grad = lambda x: fill(x, 1)


