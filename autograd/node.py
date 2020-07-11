from abc import ABC, abstractmethod
import numbers
from .vector import Vector
from .vector import (lt as Lt, le as Le, eq as Eq, ne as Ne, ge as Ge, gt as Gt, 
                     all as All, any as Any, fill as Fill)


class Node(ABC):
    id = 0

    def __init__(self, parents):
        self.parents = [parent if isinstance(parent, Node) else Variable(parent) 
                        for parent in parents]
        self.id = Node.id
        Node.id += 1

    @property
    def dim(self):
        return self.value.dim
    
    @property
    def is_leaf(self):
        # leaf node has no parents
        return not bool(self.parents)

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
        return self.value.copy()

    def __int__(self):
        return int(self.value)
    
    def __float__(self):
        return float(self.value)
    
    def __complex__(self):
        return complex(self.value)

    def abs(self):
        from .ops import Abs
        return Abs(self)

    def neg(self):
        from .ops import Neg
        return Neg(self)
    
    def log(self):
        from .ops import Log
        return Log(self)

    def log2(self):
        from .ops import Log2
        return Log2(self)
    
    def log10(self):
        from .ops import Log10
        return Log10(self)

    def log1p(self):
        from .ops import Log1p
        return Log1p(self)

    def exp(self):
        from .ops import Exp
        return Exp(self)

    def sin(self):
        from .ops import Sin
        return Sin(self)
    
    def cos(self):
        from .ops import Cos
        return Cos(self)
    
    def tan(self):
        from .ops import Tan
        return Tan(self)
    
    def sinh(self):
        from .ops import Sinh
        return Sinh(self)
    
    def cosh(self):
        from .ops import Cosh
        return Cosh(self)
    
    def tanh(self):
        from .ops import Tanh
        return Tanh(self)

    def sigmoid(self):
        from .ops import Sigmoid
        return Sigmoid(self)

    def relu(self):
        from .ops import ReLU
        return ReLU(self)
    
    def sum(self):
        from .ops import Sum
        return Sum(self)

    def add(self, other):
        from .ops import Add
        return Add(self, other)

    def sub(self, other):
        from .ops import Sub
        return Sub(self, other)

    def __rsub__(self, other):
        from .ops import Sub
        return Sub(other, self)

    def mul(self, other):
        from .ops import Mul
        return Mul(self, other)
    
    def div(self, other):
        from .ops import Div
        return Div(self, other)

    def __rtruediv__(self, other):
        from .ops import Div
        return Div(other, self)
    
    def pow(self, other):
        from .ops import Pow
        return Pow(self, other)

    def __rpow__(self, other):
        from .ops import Pow
        return Pow(other, self)

    def matmul(self, other):
        from .ops import Matmul
        return Matmul(self, other)

    # def lt(self, other):
    #     return Lt(self.value, other.value)
    
    # def le(self, other):
    #     return Le(self.value, other.value)
    
    # def eq(self, other):
    #     return Eq(self.value, other.value)
    
    # def ne(self, other):
    #     return Ne(self.value, other.value)
    
    # def ge(self, other):
    #     return Ge(self.value, other.value)
    
    # def gt(self, other):
    #     return Gt(self.value, other.value)

    # def all(self):
    #     return All(self.value)

    # def any(self):
    #     return Any(self.value)
    
    __abs__ = abs 
    __neg__ = neg
    
    __radd__ = __add__ = add
    __sub__ = sub
    __rmul__ = __mul__ = mul
    __truediv__ = div
    __pow__ = pow
    __rmatmul__ = __matmul__ = matmul

    # logic operations (not included in computation graph)
    # __lt__ = lt
    # __le__ = le
    # __eq__ = eq
    # __ne__ = ne
    # __ge__ = ge
    # __gt__ = gt

    # fill is not supported by standard nodes, only variables
    fill = None


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
    

abs = Node.abs
neg = Node.neg
log = Node.log
log2 = Node.log2
log10 = Node.log10
log1p = Node.log1p
exp = Node.exp
sin = Node.sin
cos = Node.cos
tan = Node.tan
sinh = Node.sinh
cosh = Node.cosh
tanh = Node.tanh
sigmoid = Node.sigmoid
relu = Node.relu
sum = Node.sum
add = Node.add
sub = Node.sub
mul = Node.mul
div = Node.div
pow = Node.pow
matmul = Node.matmul
# lt = Node.lt
# le = Node.le
# eq = Node.eq
# ge = Node.ge
# ne = Node.ne
# gt = Node.gt
# all = Node.all
# any = Node.any

