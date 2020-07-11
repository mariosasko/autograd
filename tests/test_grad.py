import math
import unittest
import autograd as ag


class TestGrad(unittest.TestCase):

    def test_sin(self):
        v = ag.Variable(ag.Vector([2, 1, -3]))
        t = v.sin()
        
        ag.grad(t)
        v_grad_true = v.value.cos()
    
        self.assertAlmostEqual(v.grad[0], v_grad_true[0], places=8)
        self.assertAlmostEqual(v.grad[1], v_grad_true[1], places=8)
        self.assertAlmostEqual(v.grad[2], v_grad_true[2], places=8)

    def test_log_reg(self):
        w = ag.Variable(ag.Vector([-0.5, 2, 3]))
        b = ag.Variable(2)
        x = ag.Vector([12, 3, 2])
        
        h = w @ x + b
        ag.grad(h)

        self.assertAlmostEqual(w.grad[0], x[0], places=8)
        self.assertAlmostEqual(w.grad[1], x[1], places=8)
        self.assertAlmostEqual(w.grad[2], x[2], places=8)

        self.assertAlmostEqual(b.grad.item(), 1, places=8)

    def test_log_normal(self):
        y = 10
        v2 = mu = ag.Variable(5)
        v3 = sigma = ag.Variable(2)
        v4 = y - v2
        v5 = v4 / v3
        v6 = v5 ** 2
        v7 = -0.5 * v6
        v8 = v3.log()
        v9 = v7 - v8
        v10 = v9 - 0.5 * math.log(2 * math.pi)

        ag.grad(v10)

        self.assertAlmostEqual(mu.grad, 1.25, places=8)
        self.assertAlmostEqual(sigma.grad, 2.625, places=8)


if __name__ == '__main__':
    unittest.main()