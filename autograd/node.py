from abc import ABC, abstractmethod
import numbers
from .ops import (Abs, Neg, Log, Log2, Log10, Log1p, Exp, Sin, Cos, Tan, Sinh, Cosh, Tanh,
                  pow, add, sub, mul, matmul, div, Sum, Sigmoid, ReLU)
from .vector import Vector
from .vector import (lt as Lt, le as Le, eq as Eq, ne as Ne, ge as Ge, gt as Gt, 
                     all as All, any as Any, fill as Fill)


class Node(ABC):
    id = 0

    def __init__(self, parents):
        self.parents = parents
        self.id = id
        Node.id += 1

    @property
    def ndim(self):
        return self.value.ndim
    
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
        return f'{self.__class__.__name__}({self.value})'

    def copy(self):
        return self.__class__(self.parents)

    def detach(self):
        return self.value.copy

    def ones(self):
        return self.value.ones()

    def zeros(self):
        return self.value.zeros()

    def _unary_logic_op(self, op):
        return op(self.value)

    def _binary_logic_op(self, other, op):
        return op(self.value, other.value)

    def lt(self, other):
        return self._binary_logic_op(other, Lt)
    
    def le(self, other):
        return self._binary_logic_op(other, Le)
    
    def eq(self, other):
        return self._binary_logic_op(other, Eq)
    
    def ne(self, other):
        return self._binary_logic_op(other, Ne)
    
    def ge(self, other):
        return self._binary_logic_op(other, Ge)
    
    def gt(self, other):
        return self._binary_logic_op(other, Gt)

    def all(self):
        return self._unary_logic_op(All)

    def any(self):
        return self._unary_logic_op(Any)
    
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
    __lt__ = lt
    __le__ = le
    __eq__ = eq
    __ne__ = ne
    __ge__ = ge
    __gt__ = gt
    __all__ = all
    __any__ = any

    # fill is not supported by standard nodes, only variables
    fill = None

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
    
    def copy(self):
        return self.__class__(self._value)

    def fill(self, val):
        return self.__class__(self._value.fill(val))
    
# TODO: add abbreviations


