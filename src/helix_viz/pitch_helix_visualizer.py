import time
from collections import defaultdict
from collections import deque
from typing import List

import numpy as np
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets

from helix_viz.audio_processor import AudioProcessor
from helix_viz.guitar_profiles import GuitarProfile
from helix_viz.helix_math import D_BOTTOM_OFFSET_RADIANS, frequency_to_xyz, helix_turns
from helix_viz.pattern_memory import (
    NoteEvent,
    build_fading_graph_state,
    frequency_to_midi_note,
    midi_note_to_frequency,
)
from helix_viz.visualizer_base import VisualizerBase

NOTE_NAMES: List[str] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class PitchHelixVisualizer(VisualizerBase):
    def __init__(
        self,
        processor: AudioProcessor,
        guitar_profile: GuitarProfile,
        memory_fade_time_s: float = 2.6,
        min_event_interval_s: float = 0.08,
        edge_window_s: float = 0.22,
        face_opacity: float = 1.0,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(processor, parent=parent)

        self.min_freq = guitar_profile.lowest_frequency()
        self.max_freq = guitar_profile.highest_frequency()

        self.radius = 10.0
        self.pitch = 3.0
        self.turns = helix_turns(self.min_freq, self.max_freq)
        self.angular_offset_radians = float(D_BOTTOM_OFFSET_RADIANS)
        self.memory_fade_time_s = float(memory_fade_time_s)
        self.min_event_interval_s = float(min_event_interval_s)
        self.edge_window_s = float(edge_window_s)
        self.memory_face_opacity = float(face_opacity)
        self.memory_edge_opacity = 0.0
        self.camera_follow_note = False
        self._camera_follow_spring = 0.045
        self._camera_follow_damping = 0.93
        self._camera_follow_momentum_gain = 0.85
        self._camera_follow_azim = 45.0
        self._camera_follow_elev = 60.0
        self._camera_follow_angular_velocity = 0.0
        self._camera_follow_last_midi_note: int | None = None
        self.note_events: deque[NoteEvent] = deque(maxlen=256)
        self._last_event_time_by_note: dict[int, float] = {}
        self._memory_fill_meshes: dict[int, gl.GLMeshItem] = {}
        self._external_note_events: list[NoteEvent] | None = None
        self._source_color_map: dict[int, tuple[float, float, float]] = {}
        self._external_now_s: float | None = None

        self.view = gl.GLViewWidget()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.view)

        self.view.opts["distance"] = 40
        self.view.orbit(45, 60)

        self.create_helix()
        self.scatter = gl.GLScatterPlotItem()
        self.memory_nodes_scatter = gl.GLScatterPlotItem()
        self.memory_edges_line = gl.GLLinePlotItem(mode="lines", antialias=True, width=2.0)
        self.add_note_labels()
        self.create_semitone_nodes()
        self.view.addItem(self.memory_edges_line)
        self.view.addItem(self.memory_nodes_scatter)
        self.view.addItem(self.scatter)

    def set_external_timeline_events(
        self,
        note_events: list[NoteEvent] | None,
        now_s: float | None = None,
        source_color_map: dict[int, tuple[float, float, float]] | None = None,
    ) -> None:
        """Set externally generated memory events (e.g. MIDI channels in preview/render-gl)."""
        self._external_note_events = note_events
        self._external_now_s = now_s
        if source_color_map is not None:
            self._source_color_map = dict(source_color_map)

    def create_semitone_nodes(self) -> None:
        self.semitone_scatter = gl.GLScatterPlotItem()
        self.view.addItem(self.semitone_scatter)

        lowest = self.min_freq
        highest = self.max_freq
        num_semitones = int(np.round(12 * np.log2(highest / lowest)))
        freqs = [lowest * (2 ** (i / 12)) for i in range(num_semitones + 1)]

        positions = []
        colors = []
        for freq in freqs:
            positions.append(self.frequency_to_xyz(freq))
            colors.append((255, 255, 0, 0.2))

        self.semitone_scatter.setData(pos=np.array(positions), color=np.array(colors), size=18.0)

    def add_note_labels(self) -> None:
        from pyqtgraph.opengl.items.GLTextItem import GLTextItem

        for i, name in enumerate(NOTE_NAMES):
            theta = (2 * np.pi * (i / 12)) + self.angular_offset_radians
            label_radius = self.radius * 1.1
            x = label_radius * np.cos(theta)
            y = label_radius * np.sin(theta)
            z = 0
            text_item = GLTextItem(pos=(x, y, z), text=name, color=(255, 255, 255, 255))
            self.view.addItem(text_item)

    def create_helix(self) -> None:
        freqs = np.geomspace(self.min_freq, self.max_freq, 1000)
        pts = np.vstack([self.frequency_to_xyz(freq) for freq in freqs])
        line = gl.GLLinePlotItem(pos=pts, color=(0.5, 0.5, 0.5, 1.0), width=1.0, antialias=True)
        self.view.addItem(line)

    def frequency_to_xyz(self, freq: float) -> np.ndarray:
        return frequency_to_xyz(
            freq=freq,
            min_freq=self.min_freq,
            radius=self.radius,
            pitch=self.pitch,
            angular_offset_radians=self.angular_offset_radians,
        )

    def update_visualization(self) -> None:
        now_s = self._external_now_s if self._external_note_events is not None else time.monotonic()
        if now_s is None:
            now_s = time.monotonic()

        dominant_freq = self.processor.current_top_k_frequencies[0]
        if dominant_freq is None:
            self.scatter.setData(pos=np.zeros((0, 3)))
        else:
            points = []
            colors = []
            if self.min_freq <= dominant_freq <= self.max_freq:
                points.append(self.frequency_to_xyz(dominant_freq))
                colors.append((1.0, 0.2, 0.2, 1.0))
                self._update_camera_follow(dominant_freq)

            if points:
                self.scatter.setData(pos=np.array(points), color=np.array(colors), size=12.0)
            else:
                self.scatter.setData(pos=np.zeros((0, 3)))

        if self._external_note_events is None:
            for freq in self.processor.current_top_k_frequencies:
                if freq is None:
                    continue
                if not (self.min_freq <= freq <= self.max_freq):
                    continue
                midi_note = frequency_to_midi_note(freq)
                prev_t = self._last_event_time_by_note.get(midi_note)
                if prev_t is not None and (now_s - prev_t) < self.min_event_interval_s:
                    continue
                self.note_events.append(NoteEvent(timestamp_s=now_s, midi_note=midi_note, source_id=0))
                self._last_event_time_by_note[midi_note] = now_s

            while self.note_events and (now_s - self.note_events[0].timestamp_s) > self.memory_fade_time_s:
                self.note_events.popleft()
            events = list(self.note_events)
        else:
            events = list(self._external_note_events)

        events_by_source: dict[int, list[NoteEvent]] = defaultdict(list)
        for ev in events:
            events_by_source[ev.source_id].append(ev)

        source_states: dict[int, tuple[dict[int, float], dict[tuple[int, int], float]]] = {}
        for source_id, source_events in events_by_source.items():
            node_strength, edge_strength = build_fading_graph_state(
                events=source_events,
                now_s=now_s,
                fade_time_s=self.memory_fade_time_s,
                edge_window_s=self.edge_window_s,
            )
            if node_strength or edge_strength:
                source_states[source_id] = (node_strength, edge_strength)

        self._update_memory_nodes(source_states)
        self._update_memory_edges(source_states)
        self._update_memory_fill(source_states)

    def _note_position(self, midi_note: int) -> np.ndarray:
        freq = midi_note_to_frequency(midi_note)
        return self.frequency_to_xyz(freq)

    def _color_for_source(self, source_id: int) -> tuple[float, float, float]:
        default = (1.0, 0.62, 0.1)
        return self._source_color_map.get(source_id, default)

    def _update_memory_nodes(
        self,
        source_states: dict[int, tuple[dict[int, float], dict[tuple[int, int], float]]],
    ) -> None:
        positions = []
        colors = []
        sizes = []
        for source_id, (node_strength, _edge_strength) in source_states.items():
            r, g, b = self._color_for_source(source_id)
            for midi_note, strength in node_strength.items():
                pos = self._note_position(midi_note)
                positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
                node_alpha = float(np.clip(0.95 * (strength**1.35), 0.0, 1.0))
                colors.append((r, g, b, node_alpha))
                sizes.append(5.0 + 14.0 * strength)
        if not positions:
            self.memory_nodes_scatter.setData(pos=np.zeros((0, 3)))
            return
        self.memory_nodes_scatter.setData(
            pos=np.array(positions, dtype=np.float64),
            color=np.array(colors, dtype=np.float64),
            size=np.array(sizes, dtype=np.float64),
        )

    def _update_memory_edges(
        self,
        source_states: dict[int, tuple[dict[int, float], dict[tuple[int, int], float]]],
    ) -> None:
        line_points = []
        line_colors = []
        for source_id, (_node_strength, edge_strength) in source_states.items():
            r, g, b = self._color_for_source(source_id)
            for (a, bnote), strength in edge_strength.items():
                pa = self._note_position(a)
                pb = self._note_position(bnote)
                edge_alpha = float(np.clip(0.70 * (strength**1.45) * self.memory_edge_opacity, 0.0, 1.0))
                color = (r, g, b, edge_alpha)
                line_points.extend([pa.tolist(), pb.tolist()])
                line_colors.extend([color, color])
        if not line_points:
            self.memory_edges_line.setData(pos=np.zeros((0, 3)))
            return
        self.memory_edges_line.setData(
            pos=np.array(line_points, dtype=np.float64),
            color=np.array(line_colors, dtype=np.float64),
            width=2.0,
            mode="lines",
        )

    def _update_memory_fill(
        self,
        source_states: dict[int, tuple[dict[int, float], dict[tuple[int, int], float]]],
    ) -> None:
        active_sources = set(source_states.keys())
        stale_sources = set(self._memory_fill_meshes.keys()) - active_sources
        for source_id in stale_sources:
            self.view.removeItem(self._memory_fill_meshes[source_id])
            del self._memory_fill_meshes[source_id]

        for source_id, (node_strength, _edge_strength) in source_states.items():
            active = [(note, w) for note, w in node_strength.items() if w > 0.015]
            if len(active) < 3:
                if source_id in self._memory_fill_meshes:
                    self.view.removeItem(self._memory_fill_meshes[source_id])
                    del self._memory_fill_meshes[source_id]
                continue

            active.sort(key=lambda x: x[0])
            boundary = []
            for midi_note, _strength in active:
                pos = self._note_position(midi_note)
                boundary.append([float(pos[0]), float(pos[1]), float(pos[2])])

            center = np.mean(np.array(boundary, dtype=np.float64), axis=0)
            vertices = np.vstack([np.array([center]), np.array(boundary, dtype=np.float64)])
            faces = []
            for i in range(1, len(boundary) + 1):
                j = 1 if i == len(boundary) else i + 1
                faces.append([0, i, j])
            faces = np.array(faces, dtype=np.int32)

            avg_strength = float(np.mean([w for _, w in active]))
            r, g, b = self._color_for_source(source_id)
            alpha = float(np.clip(0.42 * (avg_strength**1.6) * self.memory_face_opacity, 0.0, 1.0))
            face_color = np.array([r, g, b, alpha], dtype=np.float32)
            face_colors = np.repeat(face_color[np.newaxis, :], len(faces), axis=0)

            if source_id in self._memory_fill_meshes:
                self.view.removeItem(self._memory_fill_meshes[source_id])
                del self._memory_fill_meshes[source_id]
            mesh = gl.GLMeshItem(
                vertexes=vertices,
                faces=faces,
                faceColors=face_colors,
                smooth=False,
                # Keep mesh border hidden so only decaying face alpha is visible.
                drawEdges=False,
                drawFaces=True,
            )
            mesh.setGLOptions("translucent")
            self._memory_fill_meshes[source_id] = mesh
            self.view.addItem(mesh)

    def _update_camera_follow(self, dominant_freq: float) -> None:
        if not self.camera_follow_note:
            self._camera_follow_angular_velocity = 0.0
            self._camera_follow_last_midi_note = None
            return
        midi_note = frequency_to_midi_note(dominant_freq)
        prev_midi = self._camera_follow_last_midi_note
        if prev_midi is not None and prev_midi != midi_note:
            interval = midi_note - prev_midi
            interval = int(np.clip(interval, -12, 12))
            self._camera_follow_angular_velocity += interval * self._camera_follow_momentum_gain
        self._camera_follow_last_midi_note = midi_note

        target_azim = (midi_note % 12) * 30.0
        delta = (target_azim - self._camera_follow_azim + 540.0) % 360.0 - 180.0
        spring_accel = delta * self._camera_follow_spring
        self._camera_follow_angular_velocity += spring_accel
        self._camera_follow_angular_velocity *= self._camera_follow_damping
        self._camera_follow_azim = (self._camera_follow_azim + self._camera_follow_angular_velocity) % 360.0
        self.view.setCameraPosition(azimuth=self._camera_follow_azim, elevation=self._camera_follow_elev)
