import unittest

import numpy as np

from run_baseline import Parameters, assemble_beam, element_matrices, hermite


class BaselineTests(unittest.TestCase):
    def test_element_matrices_are_symmetric(self):
        k, m = element_matrices(1.0, Parameters())
        np.testing.assert_allclose(k, k.T)
        np.testing.assert_allclose(m, m.T)
        self.assertTrue(np.linalg.eigvalsh(m).min() > 0)

    def test_partition_of_unity(self):
        n, _ = hermite(0.37, 1.0)
        self.assertAlmostEqual(n[0] + n[2], 1.0)

    def test_beam_dimensions(self):
        k, m, free = assemble_beam(Parameters())
        self.assertEqual(k.shape, (120, 120))
        self.assertEqual(m.shape, (120, 120))
        self.assertEqual(len(free), 120)


if __name__ == "__main__":
    unittest.main()

