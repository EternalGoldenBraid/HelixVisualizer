import tempfile
import unittest
from pathlib import Path

from helix_viz.note_io import load_note_spans_json


class TestNoteIo(unittest.TestCase):
    def test_load_note_spans_json(self) -> None:
        payload = '[{"start_s": 0.2, "end_s": 0.5, "midi_note": 64, "channel": 2}, {"start_s": 0.0, "end_s": 0.4, "midi_note": 60}]'
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "notes.json"
            p.write_text(payload, encoding="utf-8")
            spans = load_note_spans_json(p)

        self.assertEqual(len(spans), 2)
        self.assertEqual(spans[0].midi_note, 60)
        self.assertEqual(spans[1].midi_note, 64)
        self.assertEqual(spans[0].velocity, 100)
        self.assertEqual(spans[0].channel, 0)
        self.assertEqual(spans[1].channel, 2)


if __name__ == "__main__":
    unittest.main()
