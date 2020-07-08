from abc import ABC, abstractmethod
import numbers
from .ops import (Abs, Neg, Log, Log2, Log10, Log1p, Exp, Sin, Cos, Tan, Sinh, Cosh, Tanh,
                  pow, add, sub, mul, matmul, div, Sum,  
                  lt, le, eq, ge, gt, all, any, fill, Sigmoid, ReLU)
from .vector import Vector



class Node(ABC):
    id = 0

    def __init__(self, parents):
        self.parents = parents
        self.id = id
        Node.id += 1

    @property
    def is_leaf(self):
        # leaf node has no parents
        return bool(self.parents)

    @property
    @abstractmethod
    def value(self):
        raise NotImplementedError

    @abstractmethod
    def partial_derivative(self, prev_grad, wrt):
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}({self.parents})'

    __abs__ = abs = Abs
    __neg__ = neg = Neg
    log = Log
    log2 = Log2
    log10 = Log10
    log1p = Log1p
    exp = Exp
    sin = Sin
    cos = Cos
    tan = Tan 
    sinh = Sinh
    cosh = Cosh
    tanh = Tanh
    sum = Sum
    sigmoid = Sigmoid
    relu = ReLU
    # def abs(self):
    #     return Abs(self)


class Variable(Node):

    def __init__(self, value):
        super().__init__([])

        if isinstance(value, Vector):
            self._value = value
        elif isinstance(value, numbers.Number):
            self._value = Vector([value])
        else:
            raise ValueError(f'{self.__class__.__name__} '
                              'supports only Python scalars and vectors as values')
        
        self._grad = None

    @property
    def value(self):
        return self._value

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    # instances of this class are leafs in the computation graph 
    # and as such don't provide partial derivatives
    partial_derivative = None

    # TODO: add fill
    



