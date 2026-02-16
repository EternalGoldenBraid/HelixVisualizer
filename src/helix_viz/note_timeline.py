from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoteSpan:
    """Time-bounded musical note used by offline rendering and tests.

    A ``NoteSpan`` describes one note as a start/end interval in seconds, plus
    a MIDI note number, velocity, and source channel.

    - ``midi_note`` is a pitch index from 0 to 127 (for example, 60 = middle C).
    - ``velocity`` is the note intensity from 0 to 127.
    - ``channel`` is the MIDI channel index from 0 to 15.
    - ``start_s`` / ``end_s`` are absolute timeline times in seconds.
    """

    start_s: float
    end_s: float
    midi_note: int
    velocity: int = 100
    channel: int = 0

    def __post_init__(self) -> None:
        if self.end_s < self.start_s:
            raise ValueError("NoteSpan end_s must be >= start_s.")
        if not (0 <= self.midi_note <= 127):
            raise ValueError("NoteSpan midi_note must be in [0,127].")
        if not (0 <= self.velocity <= 127):
            raise ValueError("NoteSpan velocity must be in [0,127].")
        if not (0 <= self.channel <= 15):
            raise ValueError("NoteSpan channel must be in [0,15].")
