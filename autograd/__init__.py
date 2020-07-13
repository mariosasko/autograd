from .node import Node, Variable
# from .node import (abs, neg, log, log2, log10, log1p, 
#                    exp, sin, cos, tan, sinh, cosh, tanh, sum, sigmoid, relu,
#                    add, sub, mul, div, pow, matmul) 
# from .node import lt, le, eq, ne, ge, gt, all, any
from .vector import Vector
from .grad import grad


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
