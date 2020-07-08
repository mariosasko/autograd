from collections.abc import Sequence
import builtins
import math
import numbers
import operator


class Vector:

    def __init__(self, data):
        if not isinstance(data, Sequence):
            raise TypeError('data argument is not a sequence')

        if not builtins.all(isinstance(v, numbers.Number) for v in data):
            raise ValueError('data argument contains an element that is not a number')

        self.data = list(data)

    @property
    def dim(self):
        return len(self.data)

    def copy(self):
        return Vector(self.data.copy())

    @classmethod
    def ones(cls, dim):
        return cls([1] * dim)

    @classmethod
    def zeros(cls, dim):
        return cls([0] * dim)

    def fill(self, val):
        data_new = [val] * self.dim
        return Vector(data_new)

    def item(self):
        if self.dim != 1:
            raise ValueError('only one element vectors can be coverted to Python scalars')
        return self.data[0]

    def tolist(self):
        return self.data.copy()

    def __getitem__(self, idx):
        if not isinstance(idx, (int, slice)):
            raise TypeError('only integers and slices are valid indices')
        elif isinstance(idx, int):
            return Vector([self.data[idx]])
        else:
            return Vector(self.data[idx])
    
    def __setitem__(self, idx, val):
        if not isinstance(idx, (int, slice)):
            raise TypeError('only integers and slices are valid indices')
        elif isinstance(idx, int):
            if not isinstance(val, numbers.Number):
                raise TypeError('value argument is not a number')
            self.data[idx] = val
        else:
            if isinstance(val, Sequence):            
                if not builtins.all(isinstance(v, numbers.Number) for v in val):
                    raise ValueError('value argument contains an element that is not a number')
                if len(self.data[idx]) != len(val):
                    raise ValueError("number of indices doesn't match the number of values")
                self.data[idx] = val
            elif isinstance(val, numbers.Number):
                self.data[idx] = [val] * len(range(*idx.indices(self.dim)))
            else:
                raise TypeError('value argument is not a sequence or a number') 

    def __delitem__(self, idx):
        del self.data[idx]

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'({repr(self.data) if self.dim != 1 else repr(self.data[0])})')

    def __len__(self):
        return self.dim

    def _unary_op(self, op):
        data_new = [op(v) for v in self.data]
        return Vector(data_new)

    def log(self, base=math.e):
        def _log(v):
            return math.log(v, base)
        return self._unary_op(_log)
        
    def exp(self):
        return self._unary_op(math.exp)

    def abs(self):
        return self._unary_op(operator.abs)
            
    def __pos__(self):
        return self

    def neg(self):
        return self * -1

    def sin(self):
        return self._unary_op(math.sin)

    def cos(self):
        return self._unary_op(math.cos)
    
    def tan(self):
        return self._unary_op(math.tan)

    def sinh(self):
        return self._unary_op(math.sinh)

    def cosh(self):
        return self._unary_op(math.cosh)
    
    def tanh(self):
        return self._unary_op(math.tanh)

    def _binary_op(self, other, op):
        if isinstance(other, numbers.Number):
            data = [op(v, other) for v in self.data]
        elif isinstance(other, Vector):
            if self.dim != other.dim:
                raise ValueError("operand dimensions doesn't match")
            data = [op(u, v) for u, v in zip(self.data, other.data)]
        else:
            return NotImplemented

        return Vector(data)
    
    def pow(self, other):
        return self._binary_op(other, operator.pow)

    def __rpow__(self, other):
        if not isinstance(other, numbers.Number):
            return NotImplemented
        return Vector.zeros(self.dim).fill(other).pow(self)

    def add(self, other):
        return self._binary_op(other, operator.add)

    def sub(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def mul(self, other):
        return self._binary_op(other, operator.mul)

    def matmul(self, other):
        if not isinstance(other, Vector):
            raise TypeError('one of the operands is not a vector')
        if self.dim != other.dim:
            raise ValueError("operand dimensions doesn't match")
        return (self * other).sum()

    def div(self, other):
        return self._binary_op(other, operator.truediv)

    def __rtruediv__(self, other):
        return (self ** -1) * other

    def sum(self):
        return Vector([builtins.sum(self.data)])

    def lt(self, other):
        return self._binary_op(other, operator.lt)
    
    def le(self, other):
        return self._binary_op(other, operator.le)

    def eq(self, other):
        return self._binary_op(other, operator.eq)

    def __ne__(self, other):
        return self._binary_op(other, operator.ne)
    
    def ge(self, other):
        return self._binary_op(other, operator.ge)
   
    def gt(self, other):
        return self._binary_op(other, operator.gt)
    
    def all(self):
        return builtins.all(self)

    def any(self):
        return builtins.any(self)

    __int__ = __float__ = __complex__ = item
    __pow__ = pow
    __neg__ = neg
    __radd__ = __add__ = add
    __sub__ = sub
    __rmul__ = __mul__ = mul
    __rmatmul__ = __matmul__ = matmul
    __truediv__ = div
    __lt__ = lt
    __le__ = le
    __eq__ = eq
    __ge__ = ge
    __gt__ = gt


abs = Vector.abs
neg = Vector.neg
log = Vector.log
exp = Vector.exp
sin = Vector.sin
cos = Vector.cos
tan = Vector.tan
sinh = Vector.sinh
cosh = Vector.cosh
tanh = Vector.tanh
pow = Vector.pow
add = Vector.add
sub = Vector.sub
mul = Vector.mul
matmul = Vector.matmul
div = Vector.div
sum = Vector.sum
lt = Vector.lt
le = Vector.le
eq = Vector.eq
ge = Vector.ge
gt = Vector.gt
all = Vector.all
any = Vector.any
    
