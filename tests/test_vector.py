import autograd as ag
import unittest


class TestVector(unittest.TestCase):

    def test_getitem(self):
        v = ag.Vector([2, 18, 3, 10, 31])
        self.assertEqual(v[0].item(), 2)
        self.assertTrue(v[1:5:2].tolist(), [18, 10])

    def test_setitem(self):
        v = ag.Vector([2, 18, 3, 10, 400, 21])
        v[0] = 100
        self.assertEqual(v[0], 100) 

        data_new = [20, 4, 10]
        v[1:6:2] = data_new
        self.assertEqual(v[1:6:2].tolist(), data_new)

    def test_add(self):
        v1 = ag.Vector([12, 3, 9])
        v2 = ag.Vector([4, -140, 2])
        v3 = v1 + v2
        self.assertEqual(v3[0], v1[0] + v2[0])
        self.assertEqual(v3[1], v1[2] + v2[1])
        self.assertEqual(v3[2], v1[2] + v2[2])

    def test_pow(self):
        v = ag.Vector([2, 18, 3])
        v1 = v ** 3
        self.assertEqual(v1[0], v[0] ** 3)
        self.assertEqual(v1[1], v[1] ** 3)
        self.assertEqual(v1[2], v[2] ** 3)

        exp = ag.Vector([2, 10, 4])
        v2 = v ** exp
        self.assertEqual(v2[0], v[0] ** exp[0])
        self.assertEqual(v2[1], v[1] ** exp[1])
        self.assertEqual(v2[2], v[2] ** exp[2])

        base = 5
        v3 = base ** v
        self.assertEqual(v3[0], base ** v[0])
        self.assertEqual(v3[1], base ** v[1])
        self.assertEqual(v3[2], base ** v[2])

    def test_div(self):
        v = ag.Vector([2, 18, 3])

        div_scalar = 21.5
        v1 = v / div_scalar
        self.assertAlmostEqual(v1[0], v[0] / div_scalar, places=8)
        self.assertAlmostEqual(v1[1], v[1] / div_scalar, places=8)
        self.assertAlmostEqual(v1[2], v[2] / div_scalar, places=8)
        
        div_v = ag.Vector([31, 3.1, 2])
        v2 = v / div_v
        self.assertAlmostEqual(v2[0], v[0] / div_v[0], places=8)
        self.assertAlmostEqual(v2[1], v[1] / div_v[1], places=8)
        self.assertAlmostEqual(v2[2], v[2] / div_v[2], places=8)

        divid_scalar = 132
        v3 = div_scalar / v
        self.assertAlmostEqual(v3[0], divid_scalar / v[0], places=8)
        self.assertAlmostEqual(v3[1], divid_scalar / v[1], places=8)
        self.assertAlmostEqual(v3[2], divid_scalar / v[2], places=8)

    def test_sum(self):
        data = [2, 18, 3]
        v = ag.Vector(data)
        self.assertEqual(v.sum(), sum(data))

    def test_equal(self):
        v1 = ag.Vector([2, 18, 3])
        v2 = ag.Vector([10, 1, 2])
        self.assertTrue((v1 == v1).all())
        self.assertTrue((v1 != v2).any())


if __name__ == '__main__':
    unittest.main()