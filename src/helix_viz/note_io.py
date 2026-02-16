from __future__ import annotations

import json
from pathlib import Path

from helix_viz.note_timeline import NoteSpan


def load_note_spans_json(path: str | Path) -> list[NoteSpan]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Notes JSON must be a list.")
    spans: list[NoteSpan] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Notes JSON item {idx} must be an object.")
        spans.append(
            NoteSpan(
                start_s=float(item["start_s"]),
                end_s=float(item["end_s"]),
                midi_note=int(item["midi_note"]),
                velocity=int(item.get("velocity", 100)),
            )
        )
    spans.sort(key=lambda s: (s.start_s, s.midi_note, s.end_s))
    return spans
