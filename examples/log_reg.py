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

# compute grad
ag.grad(loss)

print(model.value)
print(loss.value)
print(b.grad)
print(w.grad)