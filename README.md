# Autograd

autograd is a automatic differentiation package that relies on operator overloading to dynamically build a computation graph and compute gradients during the backward pass.

## Motivation

The goal of this project is to better understand the idea behind automatic differentiation that is present in basically every modern Deep Learning framework.

## Current state

Currently, autograd supports working with Python scalars or 1-D arrays of type `Vector`, which is part of the project. In the future, I plan to add support for multidimensional arrays. Another limitation is the lack of support for higher order gradients. Right now, it's my primary goal to add support for that. 

# Example

Next example shows how to compute the gradients of a binary logistic regression model. Since autograd builds the graph dynamically, it supports conditionals and looping in Python.

```
import autograd as ag

# model parameters
w = ag.Variable(ag.Vector([-0.5, 0.3, 1]))
b = ag.Variable(2)

# instance
x = ag.Vector([10, 0.4, 3.5])
y = 1

model = ag.sigmoid(w @ x + b)

if y == 1:
    loss = -ag.log(model)
else:
    loss = -ag.log(1 - model)

# access values (forward pass)
print(f'loss: {loss.value}')

# compute gradients
ag.grad(loss)

# access gradients 
print(f'b_grad: {b.grad}')
print(f'w_grad: {w.grad}')

# output
loss: Vector(0.430446744029496)
b_grad: Vector(-0.349781451425635)
w_grad: Vector([-3.49781451425635, -0.139912580570254, -1.2242350799897226])
```

and this is the computation graph it builds.

![](assets/log_reg_comp_graph.png)

# References 
* [Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, Jeffrey Mark Siskind: Automatic differentiation in machine learning: a survey](https://arxiv.org/pdf/1502.05767.pdf)
* [Charles C. Margossian: A Review of Automatic Differentiation and its Efficient Implementation](https://arxiv.org/pdf/1811.05031.pdf)
