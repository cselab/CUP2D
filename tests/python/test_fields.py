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
            expected = np.ascontiguousarray(expected)
            # assert expected.shape == old_shape
            # assert expected.strides != old_strides

        field = getattr(sim.fields, field_name)
        field.load_uniform(expected)
        computed = field.to_uniform(interpolate=False)

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


class TestExportUniform(TestCase):
    def test_semi_quadratic_interpolation(self):
        """Test that interpolation is O(h) accurate everywhere and O(h^2)
        within blocks. Current interpolation fails to be O(h^2) at a boundary
        between coarse and refined blocks.

        The accuracy for O(h) and O(h^2) is tested by applying a linear and a
        quadratic function, respectively, to cells (functions are evaluated at
        cell centers).

        A non-uniform grid is constructed by resetting whole grid to 0, setting
        a single cell's value to 1 and running a single step of mesh adaption.
        """

        nlevels = 4
        cells0 = (32, 16)  # Initial cells at start_level of 0.
        extent = 1.0
        sim = TestSimulation(cells=cells0, nlevels=nlevels, start_level=0, extent=extent)
        sim.init()
        cells = (cells0[0] << (nlevels - 1), cells0[1] << (nlevels - 1))

        def linear(x, y):
            return 3 * x + 7 * y

        def quadratic(x, y):
            return (41.0*x + 52.0*y + 1.0) * (19.0*x + 151.0*y + 1.0)

        def calc(func, ix, iy, shape, level):
            """Evaluate function at cell centers of a given "block" of shape
            `shape`, offset `(ix, iy)` and level `level`."""
            ny, nx, *per_cell = shape
            factor = 1 << (nlevels - level - 1)
            h = extent / cells[0] * factor
            x = (ix * nx + 0.5 + np.arange(nx)) * h
            y = (iy * ny + 0.5 + np.arange(ny)) * h
            x, y = np.meshgrid(x, y)
            return func(x, y)

        # 1 == scalar, 2 == vector
        expected1h = calc(linear, 0, 0, cells[::-1], nlevels - 1)
        expected1hh = calc(quadratic, 0, 0, cells[::-1], nlevels - 1)
        expected2h = np.stack([expected1h, expected1h], axis=-1)
        expected2hh = np.stack([expected1hh, expected1hh], axis=-1)

        # Start with completely coarse grid, add one level at a time.
        for level in range(nlevels):
            # Test O(h) accuracy with a linear function. Test both scalar and vector grids.
            for block in sim.fields.chi.blocks:
                block.data[:, :] = calc(linear, *block.ij, block.shape, block.level)
            for block in sim.fields.vel.blocks:
                block.data[:, :, :] = \
                        calc(linear, *block.ij, block.shape, block.level)[:, :, np.newaxis]

            computed1h = sim.fields.chi.to_uniform(interpolate=True)
            computed2h = sim.fields.vel.to_uniform(interpolate=True)
            skip = 1 << (nlevels - 1)
            self.assertArrayAlmostEqual(
                    expected1h[skip:-skip, skip:-skip],
                    computed1h[skip:-skip, skip:-skip])
            self.assertArrayAlmostEqual(
                    expected2h[skip:-skip, skip:-skip, :],
                    computed2h[skip:-skip, skip:-skip, :])

            # Test O(h^2) accuracy within blocks with a quadratic function.
            for block in sim.fields.chi.blocks:
                block.data[:, :] = calc(quadratic, *block.ij, block.shape, block.level)
            for block in sim.fields.vel.blocks:
                block.data[:, :, :] = \
                        calc(quadratic, *block.ij, block.shape, block.level)[:, :, np.newaxis]

            computed1hh = sim.fields.chi.to_uniform(interpolate=True)
            computed2hh = sim.fields.vel.to_uniform(interpolate=True)
            for block in sim.fields.chi.blocks:
                skip = 1 << (nlevels - block.level - 1)
                cell_range = block.cell_range(level=nlevels - 1)
                self.assertArrayAlmostEqual(
                        expected1hh[cell_range][skip:-skip, skip:-skip],
                        computed1hh[cell_range][skip:-skip, skip:-skip])
                self.assertArrayAlmostEqual(
                        expected2hh[cell_range][skip:-skip, skip:-skip, :],
                        computed2hh[cell_range][skip:-skip, skip:-skip, :])

            # Reset block values and refine the top-left block.
            for block in sim.fields.chi.blocks:
                block.data[:, :] = 0.0
            for block in sim.fields.vel.blocks:
                block.data[:, :] = 0.0
                if block.ij == (0, 0):
                    block.data[-1, -1, :] = 1.0
            sim.adapt_mesh()
