import unittest

import numpy as np

from helix_viz.audio_processor import AudioProcessor


class TestAudioProcessorFilters(unittest.TestCase):
    def test_compute_rms(self) -> None:
        samples = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
        self.assertAlmostEqual(AudioProcessor.compute_rms(samples), 1.0, places=8)

    def test_log_smoothing_in_frequency_domain(self) -> None:
        smoothed = AudioProcessor._smooth_frequency(previous=220.0, current=440.0, alpha=0.5)
        self.assertAlmostEqual(smoothed, np.sqrt(220.0 * 440.0), places=8)

    def test_parabolic_interpolation_returns_near_peak(self) -> None:
        freqs = np.array([100.0, 101.0, 102.0], dtype=np.float64)
        mags = np.array([0.95, 1.0, 0.94], dtype=np.float64)
        estimated = AudioProcessor._interpolate_peak_frequency(freqs, mags, 1)
        self.assertGreater(estimated, 100.8)
        self.assertLess(estimated, 101.2)


if __name__ == "__main__":
    unittest.main()
