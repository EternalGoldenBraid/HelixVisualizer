import unittest

import numpy as np

from helix_viz.helix_math import D_BOTTOM_OFFSET_RADIANS, frequency_to_xyz, helix_turns


class TestHelixMath(unittest.TestCase):
    def test_helix_turns_matches_octave_distance(self) -> None:
        turns = helix_turns(100.0, 400.0)
        self.assertAlmostEqual(turns, 2.0, places=8)

    def test_octave_has_same_xy(self) -> None:
        base = frequency_to_xyz(freq=110.0, min_freq=55.0, radius=10.0, pitch=3.0)
        octave = frequency_to_xyz(freq=220.0, min_freq=55.0, radius=10.0, pitch=3.0)
        self.assertAlmostEqual(base[0], octave[0], places=8)
        self.assertAlmostEqual(base[1], octave[1], places=8)

    def test_octave_z_spacing_is_constant(self) -> None:
        f1 = frequency_to_xyz(freq=110.0, min_freq=55.0, radius=10.0, pitch=3.0)
        f2 = frequency_to_xyz(freq=220.0, min_freq=55.0, radius=10.0, pitch=3.0)
        self.assertAlmostEqual(float(f2[2] - f1[2]), 6.0 * np.pi, places=8)

    def test_reject_non_positive_frequency(self) -> None:
        with self.assertRaises(ValueError):
            frequency_to_xyz(freq=0.0, min_freq=55.0, radius=10.0, pitch=3.0)

    def test_c_note_aligns_to_c_label_axis(self) -> None:
        c4 = frequency_to_xyz(freq=261.6255653, min_freq=82.41, radius=10.0, pitch=3.0)
        self.assertAlmostEqual(c4[0], 10.0, places=4)
        self.assertAlmostEqual(c4[1], 0.0, places=4)

    def test_d_note_aligns_to_bottom_when_offset_enabled(self) -> None:
        d4 = frequency_to_xyz(
            freq=293.6647679,
            min_freq=82.41,
            radius=10.0,
            pitch=3.0,
            angular_offset_radians=D_BOTTOM_OFFSET_RADIANS,
        )
        self.assertAlmostEqual(d4[0], 0.0, places=4)
        self.assertAlmostEqual(d4[1], -10.0, places=4)


if __name__ == "__main__":
    unittest.main()
