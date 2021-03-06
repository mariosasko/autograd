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

    def __int__(self):
        return int(self.item())

    def __float__(self):
        return float(self.item())

    def __complex__(self):
        return complex(self.item())

    def _unary_op(self, op):
        data_new = [op(v) for v in self.data]
        return Vector(data_new)

    def log(self):
        return self._unary_op(math.log)

    def log2(self):
        return self._unary_op(math.log2)
    
    def log10(self):
        return self._unary_op(math.log10)
    
    def log1p(self):
        return self._unary_op(math.log1p)

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
            if self.dim == other.dim:
                data = [op(u, v) for u, v in zip(self.data, other.data)]
            elif self.dim == 1:
                scalar = self[0].item()
                data = [op(v, scalar) for v in other.data]
            elif other.dim == 1:
                scalar = other[0].item()
                data = [op(v, scalar) for v in self.data]
            else:
                raise ValueError("operand dimensions don't match and aren't broadcastable")
        else:
            return NotImplemented

        return Vector(data)
    
    def sum(self):
        return Vector([builtins.sum(self.data)])

    def add(self, other):
        return self._binary_op(other, operator.add)

    def sub(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def mul(self, other):
        return self._binary_op(other, operator.mul)

    def div(self, other):
        return self._binary_op(other, operator.truediv)

    def __rtruediv__(self, other):
        return (self ** -1) * other
    
    def pow(self, other):
        return self._binary_op(other, operator.pow)

    def __rpow__(self, other):
        if not isinstance(other, numbers.Number):
            return NotImplemented
        return Vector.zeros(self.dim).fill(other).pow(self)
    
    def matmul(self, other):
        if not isinstance(other, Vector):
            raise TypeError('one of the operands is not a vector')
        if self.dim != other.dim:
            raise ValueError("operand dimensions don't match")
        return (self * other).sum()

    def lt(self, other):
        return self._binary_op(other, operator.lt)
    
    def le(self, other):
        return self._binary_op(other, operator.le)

    def eq(self, other):
        return self._binary_op(other, operator.eq)

    def ne(self, other):
        return self._binary_op(other, operator.ne)
    
    def ge(self, other):
        return self._binary_op(other, operator.ge)
   
    def gt(self, other):
        return self._binary_op(other, operator.gt)
    
    def all(self):
        return builtins.all(self)

    def any(self):
        return builtins.any(self)

    __abs__ = abs
    __neg__ = neg
  
    __radd__ = __add__ = add
    __sub__ = sub
    __rmul__ = __mul__ = mul
    __truediv__ = div
    __pow__ = pow
    __rmatmul__ = __matmul__ = matmul

    __lt__ = lt
    __le__ = le
    __eq__ = eq
    __ne__ = ne
    __ge__ = ge
    __gt__ = gt