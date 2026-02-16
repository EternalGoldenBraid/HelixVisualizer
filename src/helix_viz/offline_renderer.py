from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np

from helix_viz.helix_math import D_BOTTOM_OFFSET_RADIANS, frequency_to_xyz, helix_turns
from helix_viz.note_timeline import NoteSpan
from helix_viz.pattern_memory import NoteEvent, build_fading_graph_state, midi_note_to_frequency


def _blend_pixel(img: np.ndarray, x: int, y: int, color: tuple[int, int, int], alpha: float) -> None:
    if x < 0 or y < 0 or y >= img.shape[0] or x >= img.shape[1]:
        return
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0:
        return
    base = img[y, x, :].astype(np.float32)
    over = np.array(color, dtype=np.float32)
    img[y, x, :] = np.clip(base * (1.0 - alpha) + over * alpha, 0, 255).astype(np.uint8)


def _draw_line(
    img: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[int, int, int],
    alpha: float = 1.0,
) -> None:
    dx = x1 - x0
    dy = y1 - y0
    steps = int(max(abs(dx), abs(dy)))
    if steps <= 0:
        _blend_pixel(img, int(round(x0)), int(round(y0)), color, alpha)
        return
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + dx * t))
        y = int(round(y0 + dy * t))
        _blend_pixel(img, x, y, color, alpha)


def _draw_circle(
    img: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    color: tuple[int, int, int],
    alpha: float = 1.0,
) -> None:
    if radius <= 0:
        return
    x0 = int(math.floor(cx - radius))
    x1 = int(math.ceil(cx + radius))
    y0 = int(math.floor(cy - radius))
    y1 = int(math.ceil(cy + radius))
    rr = radius * radius
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            dx = x - cx
            dy = y - cy
            if (dx * dx + dy * dy) <= rr:
                _blend_pixel(img, x, y, color, alpha)


def _write_ppm(path: Path, img: np.ndarray) -> None:
    h, w, _ = img.shape
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    path.write_bytes(header + img.tobytes())


class OfflineHelixRenderer:
    def __init__(
        self,
        min_freq: float = 82.41,
        max_freq: float = 1318.51,
        radius: float = 10.0,
        pitch: float = 3.0,
        memory_fade_time_s: float = 2.6,
        edge_window_s: float = 0.22,
        event_interval_s: float = 0.08,
        supersample_scale: int = 2,
    ):
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.radius = float(radius)
        self.pitch = float(pitch)
        self.turns = helix_turns(self.min_freq, self.max_freq)
        self.angular_offset_radians = float(D_BOTTOM_OFFSET_RADIANS)
        self.memory_fade_time_s = float(memory_fade_time_s)
        self.edge_window_s = float(edge_window_s)
        self.event_interval_s = float(event_interval_s)
        self.supersample_scale = max(1, int(supersample_scale))

    def frequency_to_xyz(self, freq: float) -> np.ndarray:
        return frequency_to_xyz(
            freq=freq,
            min_freq=self.min_freq,
            radius=self.radius,
            pitch=self.pitch,
            angular_offset_radians=self.angular_offset_radians,
        )

    def _project(self, points: np.ndarray, width: int, height: int) -> np.ndarray:
        yaw = math.radians(42.0)
        pitch = math.radians(24.0)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cp = math.cos(pitch)
        sp = math.sin(pitch)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        x1 = (cy * x) - (sy * y)
        y1 = (sy * x) + (cy * y)
        z1 = z

        y2 = (cp * y1) - (sp * z1)
        z2 = (sp * y1) + (cp * z1)

        cam_dist = 120.0
        focal = 280.0
        scale = focal / np.maximum(20.0, cam_dist - z2)
        sx = (width * 0.5) + x1 * scale
        sy2 = (height * 0.62) - y2 * scale
        return np.column_stack([sx, sy2, z2])

    def _note_events_for_time(self, t_s: float, spans: list[NoteSpan]) -> list[NoteEvent]:
        events: list[NoteEvent] = []
        if self.event_interval_s <= 0:
            return events
        window_start = max(0.0, t_s - self.memory_fade_time_s)
        step = self.event_interval_s
        for span in spans:
            if span.start_s > t_s:
                continue
            seg_start = max(span.start_s, window_start)
            seg_end = min(span.end_s, t_s)
            if seg_end < seg_start:
                continue
            k0 = int(math.floor((seg_start - span.start_s) / step))
            k1 = int(math.floor((seg_end - span.start_s) / step))
            for k in range(k0, k1 + 1):
                ts = span.start_s + (k * step)
                if seg_start <= ts <= seg_end:
                    events.append(NoteEvent(timestamp_s=ts, midi_note=span.midi_note))
        events.sort(key=lambda e: e.timestamp_s)
        return events

    def _render_frame_native(self, t_s: float, spans: list[NoteSpan], width: int, height: int) -> np.ndarray:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :, :] = np.array([12, 16, 24], dtype=np.uint8)

        helix_freqs = np.geomspace(self.min_freq, self.max_freq, 500)
        helix_pts = np.vstack([self.frequency_to_xyz(freq) for freq in helix_freqs])
        proj = self._project(helix_pts, width, height)
        for i in range(len(proj) - 1):
            _draw_line(
                img,
                proj[i, 0],
                proj[i, 1],
                proj[i + 1, 0],
                proj[i + 1, 1],
                color=(90, 95, 110),
                alpha=0.45,
            )

        node_strength, edge_strength = build_fading_graph_state(
            events=self._note_events_for_time(t_s=t_s, spans=spans),
            now_s=t_s,
            fade_time_s=self.memory_fade_time_s,
            edge_window_s=self.edge_window_s,
        )

        for (a, b), strength in edge_strength.items():
            pa = self.frequency_to_xyz(midi_note_to_frequency(a))
            pb = self.frequency_to_xyz(midi_note_to_frequency(b))
            p2 = self._project(np.vstack([pa, pb]), width, height)
            _draw_line(
                img,
                p2[0, 0],
                p2[0, 1],
                p2[1, 0],
                p2[1, 1],
                color=(255, 120, 36),
                alpha=0.1 + 0.55 * strength,
            )

        active_notes = {
            span.midi_note
            for span in spans
            if (span.start_s <= t_s <= span.end_s) and (self.min_freq <= midi_note_to_frequency(span.midi_note) <= self.max_freq)
        }

        for midi_note, strength in node_strength.items():
            freq = midi_note_to_frequency(midi_note)
            if not (self.min_freq <= freq <= self.max_freq):
                continue
            pos = self.frequency_to_xyz(freq)
            p = self._project(pos.reshape(1, 3), width, height)[0]
            is_active = midi_note in active_notes
            color = (255, 220, 80) if is_active else (255, 140, 48)
            radius = 1.2 + (4.8 * strength) + (0.9 if is_active else 0.0)
            alpha = 0.18 + (0.72 * strength)
            _draw_circle(img, p[0], p[1], radius=radius, color=color, alpha=alpha)

        return img

    def render_frame(self, t_s: float, spans: list[NoteSpan], width: int, height: int) -> np.ndarray:
        if self.supersample_scale <= 1:
            return self._render_frame_native(t_s=t_s, spans=spans, width=width, height=height)

        s = self.supersample_scale
        hi = self._render_frame_native(t_s=t_s, spans=spans, width=width * s, height=height * s)
        # Box-filter downsample for anti-aliased output.
        lo = hi.reshape(height, s, width, s, 3).mean(axis=(1, 3))
        return np.clip(lo, 0, 255).astype(np.uint8)


def render_note_video_frames(
    spans: list[NoteSpan],
    output_frames_dir: str | Path,
    fps: int = 30,
    width: int = 960,
    height: int = 540,
    total_duration_s: float | None = None,
    tail_seconds: float = 1.0,
    renderer: OfflineHelixRenderer | None = None,
) -> tuple[Path, int]:
    if fps <= 0:
        raise ValueError("fps must be > 0.")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0.")
    if tail_seconds < 0:
        raise ValueError("tail_seconds must be >= 0.")
    if renderer is None:
        renderer = OfflineHelixRenderer()

    frames_dir = Path(output_frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    inferred_duration = 0.0
    if spans:
        inferred_duration = max(span.end_s for span in spans)
    duration_s = total_duration_s if total_duration_s is not None else (inferred_duration + tail_seconds)
    duration_s = max(0.0, float(duration_s))
    num_frames = max(1, int(math.ceil(duration_s * fps)))

    for i in range(num_frames):
        t_s = i / float(fps)
        img = renderer.render_frame(t_s=t_s, spans=spans, width=width, height=height)
        _write_ppm(frames_dir / f"frame_{i:06d}.ppm", img)
    return frames_dir, num_frames


def encode_frames_to_mp4(frames_dir: str | Path, output_mp4: str | Path, fps: int = 30) -> Path:
    frames_path = Path(frames_dir)
    output_path = Path(output_mp4)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        str(frames_path / "frame_%06d.ppm"),
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path
