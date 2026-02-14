from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoteEvent:
    timestamp_s: float
    midi_note: int


def frequency_to_midi_note(freq_hz: float, a4_hz: float = 440.0) -> int:
    if freq_hz <= 0:
        raise ValueError("Frequency must be positive.")
    return int(np.rint(69.0 + 12.0 * np.log2(freq_hz / a4_hz)))


def midi_note_to_frequency(midi_note: int, a4_hz: float = 440.0) -> float:
    return float(a4_hz * (2.0 ** ((midi_note - 69) / 12.0)))


def frequency_to_pitch_class(freq_hz: float, a4_hz: float = 440.0) -> int:
    return frequency_to_midi_note(freq_hz=freq_hz, a4_hz=a4_hz) % 12


def linear_decay_weight(age_s: float, fade_time_s: float) -> float:
    if fade_time_s <= 0:
        return 0.0
    return float(np.clip(1.0 - (age_s / fade_time_s), 0.0, 1.0))


def build_fading_graph_state(
    events: list[NoteEvent],
    now_s: float,
    fade_time_s: float,
    edge_window_s: float | None = None,
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    node_strength: dict[int, float] = {}
    edge_strength: dict[tuple[int, int], float] = {}

    weighted_events: list[tuple[NoteEvent, float]] = []
    for ev in events:
        w = linear_decay_weight(now_s - ev.timestamp_s, fade_time_s)
        if w <= 0:
            continue
        weighted_events.append((ev, w))
        prev = node_strength.get(ev.midi_note, 0.0)
        node_strength[ev.midi_note] = max(prev, w)

    if edge_window_s is None:
        for i in range(1, len(weighted_events)):
            prev_ev, prev_w = weighted_events[i - 1]
            curr_ev, curr_w = weighted_events[i]
            if prev_ev.midi_note == curr_ev.midi_note:
                continue
            edge = tuple(sorted((prev_ev.midi_note, curr_ev.midi_note)))
            ew = min(prev_w, curr_w)
            edge_strength[edge] = max(edge_strength.get(edge, 0.0), ew)
    else:
        for i, (ev_i, w_i) in enumerate(weighted_events):
            for j in range(i + 1, len(weighted_events)):
                ev_j, w_j = weighted_events[j]
                dt = ev_j.timestamp_s - ev_i.timestamp_s
                if dt > edge_window_s:
                    break
                if ev_i.midi_note == ev_j.midi_note:
                    continue
                edge = tuple(sorted((ev_i.midi_note, ev_j.midi_note)))
                ew = min(w_i, w_j)
                edge_strength[edge] = max(edge_strength.get(edge, 0.0), ew)

    return node_strength, edge_strength
