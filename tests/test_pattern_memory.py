import unittest

from helix_viz.pattern_memory import (
    NoteEvent,
    build_fading_graph_state,
    frequency_to_midi_note,
    frequency_to_pitch_class,
    linear_decay_weight,
    midi_note_to_frequency,
)


class TestPatternMemory(unittest.TestCase):
    def test_frequency_to_pitch_class(self) -> None:
        self.assertEqual(frequency_to_pitch_class(261.6255653), 0)  # C4
        self.assertEqual(frequency_to_pitch_class(293.6647679), 2)  # D4
        self.assertEqual(frequency_to_pitch_class(440.0), 9)  # A4

    def test_frequency_to_midi_note_roundtrip(self) -> None:
        midi = frequency_to_midi_note(440.0)
        self.assertEqual(midi, 69)
        self.assertAlmostEqual(midi_note_to_frequency(midi), 440.0, places=8)

    def test_linear_decay_weight(self) -> None:
        self.assertAlmostEqual(linear_decay_weight(age_s=0.0, fade_time_s=2.0), 1.0, places=8)
        self.assertAlmostEqual(linear_decay_weight(age_s=1.0, fade_time_s=2.0), 0.5, places=8)
        self.assertAlmostEqual(linear_decay_weight(age_s=3.0, fade_time_s=2.0), 0.0, places=8)

    def test_fading_graph_builds_nodes_and_edges(self) -> None:
        events = [
            NoteEvent(timestamp_s=10.0, midi_note=62),  # D4
            NoteEvent(timestamp_s=10.2, midi_note=66),  # F#4
            NoteEvent(timestamp_s=10.4, midi_note=69),  # A4
        ]
        nodes, edges = build_fading_graph_state(events=events, now_s=11.0, fade_time_s=2.0)
        self.assertIn(62, nodes)
        self.assertIn(66, nodes)
        self.assertIn(69, nodes)
        self.assertIn((62, 66), edges)
        self.assertIn((66, 69), edges)
        self.assertGreater(edges[(62, 66)], 0.0)
        self.assertGreater(edges[(66, 69)], 0.0)

    def test_expired_events_are_removed_from_state(self) -> None:
        events = [NoteEvent(timestamp_s=1.0, midi_note=60), NoteEvent(timestamp_s=3.4, midi_note=67)]
        nodes, edges = build_fading_graph_state(events=events, now_s=4.0, fade_time_s=1.0)
        self.assertNotIn(60, nodes)
        self.assertIn(67, nodes)
        self.assertEqual(edges, {})

    def test_edge_window_blocks_far_apart_events(self) -> None:
        events = [
            NoteEvent(timestamp_s=10.00, midi_note=62),
            NoteEvent(timestamp_s=10.08, midi_note=66),
            NoteEvent(timestamp_s=10.60, midi_note=69),
        ]
        _nodes, edges = build_fading_graph_state(
            events=events,
            now_s=10.8,
            fade_time_s=2.0,
            edge_window_s=0.15,
        )
        self.assertIn((62, 66), edges)
        self.assertNotIn((66, 69), edges)

    def test_edge_window_builds_clique_inside_window(self) -> None:
        events = [
            NoteEvent(timestamp_s=5.00, midi_note=62),
            NoteEvent(timestamp_s=5.05, midi_note=66),
            NoteEvent(timestamp_s=5.10, midi_note=69),
        ]
        _nodes, edges = build_fading_graph_state(
            events=events,
            now_s=5.2,
            fade_time_s=2.0,
            edge_window_s=0.2,
        )
        self.assertIn((62, 66), edges)
        self.assertIn((66, 69), edges)
        self.assertIn((62, 69), edges)


if __name__ == "__main__":
    unittest.main()
