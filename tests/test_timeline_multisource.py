import unittest

from helix_viz.main import _channel_color_map, _timeline_note_events_for_time
from helix_viz.note_timeline import NoteSpan


class TestTimelineMultiSource(unittest.TestCase):
    def test_timeline_note_events_include_multiple_channels(self) -> None:
        spans = [
            NoteSpan(start_s=0.0, end_s=1.0, midi_note=60, velocity=100, channel=0),
            NoteSpan(start_s=0.2, end_s=1.2, midi_note=64, velocity=100, channel=1),
        ]
        events = _timeline_note_events_for_time(
            spans=spans,
            t_s=0.9,
            memory_fade_time_s=2.6,
            event_interval_s=0.08,
        )
        self.assertGreater(len(events), 0)
        sources = {ev.source_id for ev in events}
        self.assertEqual(sources, {0, 1})

    def test_channel_color_map_returns_distinct_colors(self) -> None:
        spans = [
            NoteSpan(start_s=0.0, end_s=0.5, midi_note=60, channel=0),
            NoteSpan(start_s=0.0, end_s=0.5, midi_note=64, channel=1),
        ]
        cmap = _channel_color_map(spans)
        self.assertIn(0, cmap)
        self.assertIn(1, cmap)
        self.assertNotEqual(cmap[0], cmap[1])


if __name__ == "__main__":
    unittest.main()
