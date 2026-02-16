import tempfile
import unittest
from pathlib import Path

from helix_viz.note_timeline import NoteSpan
from helix_viz.offline_renderer import render_note_video_frames


class TestOfflineRenderer(unittest.TestCase):
    def test_render_note_video_frames_generates_ppm_sequence(self) -> None:
        spans = [
            NoteSpan(start_s=0.0, end_s=0.4, midi_note=60, velocity=100),
            NoteSpan(start_s=0.2, end_s=0.6, midi_note=64, velocity=100),
        ]
        with tempfile.TemporaryDirectory() as td:
            frames_dir, frame_count = render_note_video_frames(
                spans=spans,
                output_frames_dir=td,
                fps=10,
                width=160,
                height=120,
                total_duration_s=0.7,
            )
            files = sorted(Path(frames_dir).glob("frame_*.ppm"))

            self.assertEqual(frame_count, 7)
            self.assertEqual(len(files), 7)
            self.assertTrue(files[0].read_bytes().startswith(b"P6\n160 120\n255\n"))
            self.assertNotEqual(files[0].read_bytes(), files[-1].read_bytes())


if __name__ == "__main__":
    unittest.main()
