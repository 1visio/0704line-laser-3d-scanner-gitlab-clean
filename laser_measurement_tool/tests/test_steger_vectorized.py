from __future__ import annotations

import unittest

import numpy as np

from laser.steger_laser_center import _steger_points_for_peaks, steger_point


class VectorizedStegerTests(unittest.TestCase):
    def test_batch_projector_matches_scalar_eigh(self) -> None:
        rng = np.random.default_rng(42)
        shape = (7, 19)
        derivatives = tuple(
            rng.normal(0.0, 2.0, shape).astype(np.float32) for _ in range(5)
        )
        gx, gy, gxx, gxy, gyy = derivatives
        gxx[:] = -np.abs(gxx) - 2.0
        gyy[:] = -np.abs(gyy) - 0.2
        columns = np.arange(shape[1], dtype=np.intp)
        rows = rng.integers(0, shape[0], size=shape[1], dtype=np.intp)
        dx, dy, response, offset, normal_y_abs = _steger_points_for_peaks(
            columns, rows, derivatives
        )
        for index, (column, row) in enumerate(zip(columns, rows, strict=True)):
            scalar = steger_point(int(column), int(row), derivatives)
            assert scalar is not None
            x, y, scalar_response, scalar_offset, _nx, ny = scalar
            self.assertAlmostEqual(dx[index], x - column, places=10)
            self.assertAlmostEqual(dy[index], y - row, places=10)
            self.assertAlmostEqual(response[index], scalar_response, places=10)
            self.assertAlmostEqual(offset[index], abs(scalar_offset), places=10)
            self.assertAlmostEqual(normal_y_abs[index], abs(ny), places=10)


if __name__ == "__main__":
    unittest.main()
