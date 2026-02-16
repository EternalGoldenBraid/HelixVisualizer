from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from helix_viz.pitch_helix_visualizer import PitchHelixVisualizer


class _DummyView:
    def __init__(self) -> None:
        self.added = []
        self.removed = []

    def addItem(self, item) -> None:  # noqa: N802
        self.added.append(item)

    def removeItem(self, item) -> None:  # noqa: N802
        self.removed.append(item)


class _FakeMesh:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.gl_options = None

    def setGLOptions(self, options: str) -> None:  # noqa: N802
        self.gl_options = options


class TestPitchHelixVisualizerMemoryFill(unittest.TestCase):
    def _build_viz(self) -> PitchHelixVisualizer:
        viz = PitchHelixVisualizer.__new__(PitchHelixVisualizer)
        viz._memory_fill_meshes = {}
        viz.view = _DummyView()
        viz.memory_face_opacity = 1.0
        viz._source_color_map = {0: (1.0, 0.2, 0.1), 1: (0.1, 0.8, 1.0)}
        viz._note_position = lambda midi_note: np.array([float(midi_note), float(midi_note % 5), 0.0])  # type: ignore[assignment]
        return viz

    def test_memory_fill_builds_meshes_per_source_with_distinct_colors(self) -> None:
        viz = self._build_viz()
        source_states = {
            0: ({60: 1.0, 62: 0.7, 64: 0.4}, {}),
            1: ({67: 0.9, 69: 0.6, 71: 0.3}, {}),
        }
        with patch("helix_viz.pitch_helix_visualizer.gl.GLMeshItem", side_effect=lambda **kwargs: _FakeMesh(**kwargs)):
            viz._update_memory_fill(source_states)

        self.assertEqual(set(viz._memory_fill_meshes.keys()), {0, 1})
        mesh0 = viz._memory_fill_meshes[0]
        mesh1 = viz._memory_fill_meshes[1]
        color0 = mesh0.kwargs["faceColors"][0]
        color1 = mesh1.kwargs["faceColors"][0]
        self.assertNotEqual(tuple(color0[:3]), tuple(color1[:3]))
        self.assertGreater(float(color0[3]), 0.0)
        self.assertGreater(float(color1[3]), 0.0)

    def test_memory_fill_alpha_decays_with_lower_node_strength(self) -> None:
        viz = self._build_viz()
        strong_state = {0: ({60: 1.0, 62: 0.8, 64: 0.6}, {})}
        weak_state = {0: ({60: 0.3, 62: 0.25, 64: 0.2}, {})}

        with patch("helix_viz.pitch_helix_visualizer.gl.GLMeshItem", side_effect=lambda **kwargs: _FakeMesh(**kwargs)):
            viz._update_memory_fill(strong_state)
            strong_alpha = float(viz._memory_fill_meshes[0].kwargs["faceColors"][0, 3])
            viz._update_memory_fill(weak_state)
            weak_alpha = float(viz._memory_fill_meshes[0].kwargs["faceColors"][0, 3])

        self.assertLess(weak_alpha, strong_alpha)


if __name__ == "__main__":
    unittest.main()
