import tempfile
import unittest
from pathlib import Path

from helix_viz.midi_file import load_midi_timeline


def _varlen(n: int) -> bytes:
    chunks = [n & 0x7F]
    n >>= 7
    while n:
        chunks.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(chunks))


def _simple_midi() -> bytes:
    # Format 0, 1 track, 480 ticks per quarter.
    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + (480).to_bytes(2, "big")

    events = bytearray()
    # Tempo 500000 us/qn at tick 0.
    events += _varlen(0) + bytes([0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20])
    # Note on C4 velocity 100 at tick 0.
    events += _varlen(0) + bytes([0x90, 60, 100])
    # Note off C4 after one quarter (480 ticks).
    events += _varlen(480) + bytes([0x80, 60, 0])
    # End of track.
    events += _varlen(0) + bytes([0xFF, 0x2F, 0x00])
    track = b"MTrk" + len(events).to_bytes(4, "big") + bytes(events)
    return header + track


def _two_channel_midi() -> bytes:
    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + (480).to_bytes(2, "big")
    events = bytearray()
    events += _varlen(0) + bytes([0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20])
    events += _varlen(0) + bytes([0x90, 60, 100])   # ch0 on
    events += _varlen(0) + bytes([0x91, 64, 100])   # ch1 on
    events += _varlen(480) + bytes([0x80, 60, 0])   # ch0 off
    events += _varlen(0) + bytes([0x81, 64, 0])     # ch1 off
    events += _varlen(0) + bytes([0xFF, 0x2F, 0x00])
    track = b"MTrk" + len(events).to_bytes(4, "big") + bytes(events)
    return header + track


class TestMidiFile(unittest.TestCase):
    def test_load_midi_timeline_extracts_note_spans(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            midi_path = Path(td) / "simple.mid"
            midi_path.write_bytes(_simple_midi())
            timeline = load_midi_timeline(midi_path)

        self.assertEqual(timeline.ticks_per_quarter, 480)
        self.assertEqual(len(timeline.note_spans), 1)
        span = timeline.note_spans[0]
        self.assertEqual(span.midi_note, 60)
        self.assertEqual(span.velocity, 100)
        self.assertEqual(span.channel, 0)
        self.assertAlmostEqual(span.start_s, 0.0, places=6)
        self.assertAlmostEqual(span.end_s, 0.5, places=6)
        self.assertAlmostEqual(timeline.duration_s, 0.5, places=6)

    def test_load_midi_timeline_preserves_channel(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            midi_path = Path(td) / "two_channel.mid"
            midi_path.write_bytes(_two_channel_midi())
            timeline = load_midi_timeline(midi_path)

        self.assertEqual(len(timeline.note_spans), 2)
        channels = sorted(span.channel for span in timeline.note_spans)
        self.assertEqual(channels, [0, 1])


if __name__ == "__main__":
    unittest.main()
