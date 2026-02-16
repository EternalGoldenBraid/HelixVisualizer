from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

from helix_viz.note_timeline import NoteSpan

DEFAULT_TEMPO_US_PER_QUARTER = 500_000


@dataclass(frozen=True)
class MidiTimeline:
    """Parsed MIDI content normalized into renderer-friendly timeline data.

    ``note_spans`` is the list of decoded note intervals.
    ``duration_s`` is the total timeline length in seconds.
    ``ticks_per_quarter`` is the source MIDI timing resolution.
    """

    note_spans: list[NoteSpan]
    duration_s: float
    ticks_per_quarter: int


def _read_u16_be(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 2 > len(data):
        raise ValueError("Unexpected EOF while reading u16.")
    return int.from_bytes(data[offset : offset + 2], "big"), offset + 2


def _read_u32_be(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(data):
        raise ValueError("Unexpected EOF while reading u32.")
    return int.from_bytes(data[offset : offset + 4], "big"), offset + 4


def _read_var_len(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        if offset >= len(data):
            raise ValueError("Unexpected EOF while reading var-len integer.")
        b = data[offset]
        offset += 1
        value = (value << 7) | (b & 0x7F)
        if (b & 0x80) == 0:
            return value, offset
    raise ValueError("Invalid var-len integer (too long).")


def _parse_track_events(track: bytes) -> tuple[list[tuple[int, int, int, int, int]], list[tuple[int, int]], int]:
    """
    Returns:
    - note_events: (tick, kind, channel, note, velocity), kind: 1=on, 0=off
    - tempo_events: (tick, microseconds_per_quarter)
    - max_tick encountered in track
    """
    offset = 0
    abs_tick = 0
    running_status: int | None = None
    note_events: list[tuple[int, int, int, int, int]] = []
    tempo_events: list[tuple[int, int]] = []

    while offset < len(track):
        delta, offset = _read_var_len(track, offset)
        abs_tick += delta
        if offset >= len(track):
            break

        first_data_byte: int | None = None
        status = track[offset]
        if status < 0x80:
            if running_status is None:
                raise ValueError("Running-status data byte encountered without status.")
            first_data_byte = status
            status = running_status
        else:
            offset += 1
            if status < 0xF0:
                running_status = status
            else:
                running_status = None

        if status == 0xFF:
            if offset >= len(track):
                raise ValueError("Unexpected EOF in meta event.")
            meta_type = track[offset]
            offset += 1
            size, offset = _read_var_len(track, offset)
            payload = track[offset : offset + size]
            if len(payload) != size:
                raise ValueError("Unexpected EOF in meta payload.")
            offset += size
            if meta_type == 0x51 and size == 3:
                tempo = int.from_bytes(payload, "big")
                tempo_events.append((abs_tick, tempo))
            if meta_type == 0x2F:
                break
            continue

        if status in (0xF0, 0xF7):
            size, offset = _read_var_len(track, offset)
            payload = track[offset : offset + size]
            if len(payload) != size:
                raise ValueError("Unexpected EOF in sysex event.")
            offset += size
            continue

        message_type = status & 0xF0
        channel = status & 0x0F

        if message_type in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
            if first_data_byte is None:
                if offset >= len(track):
                    raise ValueError("Unexpected EOF in MIDI event data.")
                data1 = track[offset]
                offset += 1
            else:
                data1 = first_data_byte
            if offset >= len(track):
                raise ValueError("Unexpected EOF in MIDI event data.")
            data2 = track[offset]
            offset += 1

            if message_type == 0x80:
                note_events.append((abs_tick, 0, channel, data1, data2))
            elif message_type == 0x90:
                kind = 0 if data2 == 0 else 1
                note_events.append((abs_tick, kind, channel, data1, data2))
            continue

        if message_type in (0xC0, 0xD0):
            if first_data_byte is None:
                if offset >= len(track):
                    raise ValueError("Unexpected EOF in MIDI event data.")
                offset += 1
            continue

        raise ValueError(f"Unsupported MIDI status byte: 0x{status:02X}")

    return note_events, tempo_events, abs_tick


def _normalize_tempo_map(tempo_events: list[tuple[int, int]]) -> tuple[list[int], list[int], list[float]]:
    merged = sorted(tempo_events, key=lambda x: x[0])
    if not merged or merged[0][0] != 0:
        merged.insert(0, (0, DEFAULT_TEMPO_US_PER_QUARTER))
    else:
        merged[0] = (0, merged[0][1])

    ticks: list[int] = []
    tempos: list[int] = []
    for tick, tempo in merged:
        if ticks and tick == ticks[-1]:
            tempos[-1] = tempo
        else:
            ticks.append(tick)
            tempos.append(tempo)

    cumulative_s = [0.0] * len(ticks)
    for i in range(1, len(ticks)):
        dt = ticks[i] - ticks[i - 1]
        cumulative_s[i] = cumulative_s[i - 1] + (dt * tempos[i - 1]) / 1_000_000.0
    return ticks, tempos, cumulative_s


def _ticks_to_seconds(
    tick: int,
    ticks_per_quarter: int,
    tempo_ticks: list[int],
    tempo_values: list[int],
    cumulative_s: list[float],
) -> float:
    idx = bisect_right(tempo_ticks, tick) - 1
    idx = max(0, idx)
    base_tick = tempo_ticks[idx]
    base_s = cumulative_s[idx]
    return base_s + ((tick - base_tick) * tempo_values[idx]) / (1_000_000.0 * ticks_per_quarter)


def load_midi_timeline(path: str | Path) -> MidiTimeline:
    data = Path(path).read_bytes()
    offset = 0

    if data[offset : offset + 4] != b"MThd":
        raise ValueError("Invalid MIDI header chunk.")
    offset += 4
    header_len, offset = _read_u32_be(data, offset)
    if header_len < 6:
        raise ValueError("Invalid MIDI header length.")
    fmt, offset = _read_u16_be(data, offset)
    n_tracks, offset = _read_u16_be(data, offset)
    division, offset = _read_u16_be(data, offset)
    offset += header_len - 6

    if fmt not in (0, 1):
        raise ValueError(f"Unsupported MIDI format: {fmt}")
    if division & 0x8000:
        raise ValueError("SMPTE time division is not supported.")
    ticks_per_quarter = int(division)
    if ticks_per_quarter <= 0:
        raise ValueError("Invalid ticks-per-quarter value.")

    all_note_events: list[tuple[int, int, int, int, int]] = []
    all_tempo_events: list[tuple[int, int]] = []
    max_tick = 0

    for _ in range(n_tracks):
        if data[offset : offset + 4] != b"MTrk":
            raise ValueError("Invalid MIDI track chunk header.")
        offset += 4
        track_len, offset = _read_u32_be(data, offset)
        track = data[offset : offset + track_len]
        if len(track) != track_len:
            raise ValueError("Unexpected EOF while reading MIDI track.")
        offset += track_len

        note_events, tempo_events, track_max_tick = _parse_track_events(track)
        all_note_events.extend(note_events)
        all_tempo_events.extend(tempo_events)
        max_tick = max(max_tick, track_max_tick)

    tempo_ticks, tempo_values, cumulative_s = _normalize_tempo_map(all_tempo_events)

    all_note_events.sort(key=lambda ev: (ev[0], 0 if ev[1] == 0 else 1))
    active: dict[tuple[int, int], list[tuple[int, int]]] = {}
    spans: list[NoteSpan] = []

    for tick, kind, channel, note, velocity in all_note_events:
        key = (channel, note)
        if kind == 1:
            active.setdefault(key, []).append((tick, velocity))
            continue
        queue = active.get(key)
        if not queue:
            continue
        start_tick, start_velocity = queue.pop(0)
        if not queue:
            del active[key]
        start_s = _ticks_to_seconds(start_tick, ticks_per_quarter, tempo_ticks, tempo_values, cumulative_s)
        end_s = _ticks_to_seconds(tick, ticks_per_quarter, tempo_ticks, tempo_values, cumulative_s)
        spans.append(NoteSpan(start_s=start_s, end_s=end_s, midi_note=note, velocity=start_velocity))

    duration_s = _ticks_to_seconds(max_tick, ticks_per_quarter, tempo_ticks, tempo_values, cumulative_s)
    for (_channel, note), queue in active.items():
        for start_tick, start_velocity in queue:
            start_s = _ticks_to_seconds(
                start_tick,
                ticks_per_quarter,
                tempo_ticks,
                tempo_values,
                cumulative_s,
            )
            spans.append(NoteSpan(start_s=start_s, end_s=duration_s, midi_note=note, velocity=start_velocity))

    spans.sort(key=lambda s: (s.start_s, s.midi_note, s.end_s))
    if spans:
        duration_s = max(duration_s, max(span.end_s for span in spans))

    return MidiTimeline(note_spans=spans, duration_s=duration_s, ticks_per_quarter=ticks_per_quarter)
