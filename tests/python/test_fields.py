from base import TestCase, TestSimulation, cup2d

import numpy as np

from typing import Tuple, Union

class TestFields(TestCase):
    def test_load_uniform_no_amr(self):
        sim = TestSimulation(cells=(128, 64), nlevels=1)
        sim.init()

        # Test scalar.
        tmp = np.random.uniform(0.0, 1.0, (64, 128))  # (y, x)
        sim.fields.chi.load_uniform(tmp)
        self.assertArrayEqual(tmp, sim.fields.chi.to_uniform())
        with self.assertRaises(TypeError):
            sim.fields.chi.load_uniform(tmp.T)  # Wrong shape.

        # Test vector.
        tmp = np.random.uniform(0.0, 1.0, (64, 128, 2))  # (y, x, channel)
        sim.fields.vel.load_uniform(tmp)
        self.assertArrayEqual(tmp, sim.fields.vel.to_uniform())
        with self.assertRaises(TypeError):
            sim.fields.vel.load_uniform(tmp.T)  # Wrong shape.

    def _test_load_uniform_amr(
            self,
            field_name: str,
            shape: Tuple[int],
            reshuffle_axes: bool):
        max_level = 2
        sim = TestSimulation(cells=(128, 64), nlevels=max_level + 1, start_level=0)
        sim.add_shape(cup2d.Disk(sim, r=0.1, center=(0.3, 0.25),
                                 vel=(1.0, 0.0), fixed=True, forced=True))
        sim.init()
        sim.simulate(nsteps=1)  # Initialize the AMR grid and make fields non-zero.

        def _average_chunks(mat, s):
            """E.g. for s=4, average a 32x32 matrix into an 8x8 matrix."""
            mat = mat.reshape(mat.shape[0] // s, s, mat.shape[1] // s, s, *mat.shape[2:])
            mat = mat.mean((1, 3))
            return mat

        # Fill with a random matrix on the highest level.
        expected = np.random.uniform(
                0.0, 1.0, (64 << max_level, 128 << max_level, *shape))  # (y, x)
        if reshuffle_axes:
            # Test that the matrix is reordered properly automatically.
            old_shape = expected.shape
            old_strides = expected.strides
            expected = np.ascontiguousarray(expected.T).T
            # assert expected.shape == old_shape
            # assert expected.strides != old_strides

        field = getattr(sim.fields, field_name)
        field.load_uniform(expected)
        computed = field.to_uniform()

        # Method #1: compare block by block.
        for block in field.blocks:
            # Get the block, group it into chunks of size `s`, average
            # over chunks and compare to the block content.
            part = expected[block.cell_range(level=max_level)]
            part = _average_chunks(part, (1 << (max_level - block.level)))
            self.assertArrayAlmostEqual(part, block.data)

        # Method #2: compare only level max_level, then max_level-1 etc.
        for level in range(max_level, -1, -1):
            mask = np.zeros(expected.shape, dtype=bool)
            for block in field.blocks:
                if block.level == level:
                    mask[block.cell_range(level=level)] = True
            self.assertArrayAlmostEqual(mask * expected, mask * computed)

            expected = _average_chunks(expected, 2)
            computed = _average_chunks(computed, 2)

    def test_load_uniform_amr_scalar(self):
        self._test_load_uniform_amr('chi', (), False)

    def test_load_uniform_amr_vector(self):
        self._test_load_uniform_amr('vel', (2,), False)

    def test_load_uniform_amr_vector_reshuffled(self):
        self._test_load_uniform_amr('vel', (2,), True)



