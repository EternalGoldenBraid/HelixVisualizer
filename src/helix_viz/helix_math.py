from __future__ import annotations

import numpy as np

C0_FREQUENCY_HZ = 16.351597831287414
D_BOTTOM_OFFSET_RADIANS = (-np.pi / 2.0) - (2.0 * np.pi * (2.0 / 12.0))


def helix_turns(min_freq: float, max_freq: float) -> float:
    if min_freq <= 0 or max_freq <= 0:
        raise ValueError("Frequencies must be positive.")
    if max_freq < min_freq:
        raise ValueError("max_freq must be >= min_freq.")
    return float(np.log2(max_freq / min_freq))


def frequency_to_xyz(
    freq: float,
    min_freq: float,
    radius: float,
    pitch: float,
    reference_frequency: float = C0_FREQUENCY_HZ,
    angular_offset_radians: float = 0.0,
) -> np.ndarray:
    if freq <= 0 or min_freq <= 0:
        raise ValueError("Frequencies must be positive.")
    if reference_frequency <= 0:
        raise ValueError("reference_frequency must be positive.")
    theta = 2.0 * np.pi * np.log2(freq / reference_frequency) + angular_offset_radians
    theta_min = 2.0 * np.pi * np.log2(min_freq / reference_frequency)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = pitch * (theta - theta_min)
    return np.array([x, y, z], dtype=np.float64)
