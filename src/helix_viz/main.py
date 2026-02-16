import argparse
from datetime import datetime
import signal
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from helix_viz.configuration import get_default_config, load_config_file, save_config_file
from helix_viz.helix_math import D_BOTTOM_OFFSET_RADIANS, frequency_to_xyz

if TYPE_CHECKING:
    from helix_viz.guitar_profiles import GuitarProfile
    from helix_viz.note_timeline import NoteSpan
    from helix_viz.pattern_memory import NoteEvent


def _cli_option_set(argv: list[str], option: str) -> bool:
    for token in argv:
        if token == option or token.startswith(option + "="):
            return True
    return False


def _resolve_value(
    argv: list[str],
    option: str,
    arg_value,
    cfg_section: dict,
    cfg_key: str,
):
    if _cli_option_set(argv, option):
        return arg_value
    if cfg_key in cfg_section:
        return cfg_section[cfg_key]
    return arg_value


def _load_effective_config(config_path: str | None) -> dict:
    if config_path is None:
        return get_default_config()
    return load_config_file(config_path)


def _create_qapplication(msaa_samples: int = 4):
    from PyQt5 import QtGui, QtWidgets

    fmt = QtGui.QSurfaceFormat()
    fmt.setSamples(max(0, int(msaa_samples)))
    fmt.setSwapInterval(1)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    app = QtWidgets.QApplication([])
    return app


def _standard_guitar() -> "GuitarProfile":
    from helix_viz.guitar_profiles import GuitarProfile

    return GuitarProfile(
        open_strings=[82.41, 110.00, 146.83, 196.00, 246.94, 329.63],
        num_frets=22,
    )


def launch_ui(
    min_rms_threshold: float = 0.008,
    min_peak_prominence_ratio: float = 10.0,
    frequency_smoothing_alpha: float = 0.28,
    memory_fade_time_s: float = 2.6,
    min_event_interval_s: float = 0.08,
    edge_window_s: float = 0.22,
    face_opacity: float = 1.0,
    camera_follow_note: bool = False,
    msaa_samples: int = 4,
    save_config_path: str | None = None,
) -> None:
    from PyQt5 import QtCore, QtWidgets

    from helix_viz.audio_devices import select_input_device
    from helix_viz.audio_processor import AudioProcessor
    from helix_viz.pitch_helix_visualizer import PitchHelixVisualizer

    config = select_input_device(config_file=Path("outputs/audio_devices.json"))
    sr = int(config["samplerate"])

    app = _create_qapplication(msaa_samples=msaa_samples)

    processor = AudioProcessor(
        sr=sr,
        input_device_index=int(config["input_device_index"]),
        input_channels=max(1, int(config["input_channels"])),
        io_blocksize=2048,
        fft_size=4096,
        number_top_k_frequencies=3,
        min_rms_threshold=min_rms_threshold,
        min_peak_prominence_ratio=min_peak_prominence_ratio,
        frequency_smoothing_alpha=frequency_smoothing_alpha,
    )

    block_duration_ms = (processor.io_blocksize / sr) * 1000
    processing_timer = QtCore.QTimer()
    processing_timer.setInterval(max(10, int(block_duration_ms)))
    processing_timer.timeout.connect(processor.process_pending_audio)
    processing_timer.start()

    standard_guitar = _standard_guitar()

    helix_window = PitchHelixVisualizer(
        processor=processor,
        guitar_profile=standard_guitar,
        memory_fade_time_s=memory_fade_time_s,
        min_event_interval_s=min_event_interval_s,
        edge_window_s=edge_window_s,
        face_opacity=face_opacity,
    )
    helix_window.camera_follow_note = bool(camera_follow_note)
    helix_window.setWindowTitle("Pitch Helix Visualizer")
    helix_window.resize(900, 700)
    helix_window.show()

    controls = QtWidgets.QWidget()
    controls.setWindowTitle("Helix Controls")
    controls.setWindowFlag(QtCore.Qt.Tool, True)
    controls.setAttribute(QtCore.Qt.WA_AlwaysShowToolTips, True)
    controls.setMouseTracking(True)
    controls_layout = QtWidgets.QVBoxLayout(controls)
    controls.setStyleSheet(
        """
        QWidget {
            background-color: #101722;
            color: #e6edf3;
        }
        QDoubleSpinBox {
            background-color: #1a2332;
            border: 1px solid #2b3a4f;
            border-radius: 4px;
            padding: 4px 6px;
            color: #e6edf3;
        }
        QLabel {
            color: #d0d7de;
        }
        QPushButton {
            background-color: #243247;
            border: 1px solid #3b4d66;
            border-radius: 6px;
            color: #e6edf3;
            padding: 6px 10px;
        }
        QPushButton:hover {
            background-color: #2b3d56;
        }
        QPushButton:pressed {
            background-color: #1f2d40;
        }
        QToolTip {
            background-color: #182436;
            color: #ffd866;
            border: 1px solid #4d6482;
            padding: 6px;
        }
        """
    )

    def _make_spin(min_v: float, max_v: float, step: float, value: float, decimals: int = 3) -> QtWidgets.QDoubleSpinBox:
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(min_v, max_v)
        w.setSingleStep(step)
        w.setDecimals(decimals)
        w.setValue(float(value))
        return w

    def _add_group(title: str) -> QtWidgets.QFormLayout:
        group = QtWidgets.QGroupBox(title)
        group_form = QtWidgets.QFormLayout(group)
        controls_layout.addWidget(group)
        return group_form

    audio_form = _add_group("Audio")
    memory_form = _add_group("Memory")
    camera_form = _add_group("Camera")
    actions_form = _add_group("Actions")

    def _add_slider_control(
        target_form: QtWidgets.QFormLayout,
        label: str,
        tooltip: str,
        min_v: float,
        max_v: float,
        step: float,
        decimals: int,
        value: float,
        on_change,
    ) -> QtWidgets.QDoubleSpinBox:
        scale = max(1, int(round(1.0 / step)))
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(int(round(min_v * scale)), int(round(max_v * scale)))
        slider.setSingleStep(1)
        slider.setPageStep(max(1, int(scale * step * 10)))

        spin = _make_spin(min_v=min_v, max_v=max_v, step=step, value=value, decimals=decimals)
        container = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(slider, 2)
        row.addWidget(spin, 1)

        slider.setToolTip(tooltip)
        spin.setToolTip(tooltip)
        container.setToolTip(tooltip)

        def _from_slider(ivalue: int) -> None:
            fvalue = ivalue / float(scale)
            if abs(spin.value() - fvalue) > (0.5 / scale):
                spin.blockSignals(True)
                spin.setValue(fvalue)
                spin.blockSignals(False)
            on_change(float(spin.value()))

        def _from_spin(fvalue: float) -> None:
            ivalue = int(round(fvalue * scale))
            if slider.value() != ivalue:
                slider.blockSignals(True)
                slider.setValue(ivalue)
                slider.blockSignals(False)
            on_change(float(fvalue))

        slider.valueChanged.connect(_from_slider)
        spin.valueChanged.connect(_from_spin)
        slider.setValue(int(round(value * scale)))

        label_widget = QtWidgets.QLabel(label)
        label_widget.setToolTip(tooltip)
        target_form.addRow(label_widget, container)
        return spin

    def _apply_change(fn, value: float) -> None:
        fn(value)
        helix_window.update_visualization()
        app.processEvents()

    _add_slider_control(
        audio_form,
        label="Min RMS",
        tooltip="Effect: reject weak input.\nHigher values hide low-volume noise but can miss soft notes.",
        min_v=0.0,
        max_v=0.08,
        step=0.0005,
        decimals=4,
        value=processor.min_rms_threshold,
        on_change=lambda v: _apply_change(lambda x: setattr(processor, "min_rms_threshold", float(x)), v),
    )
    _add_slider_control(
        audio_form,
        label="Peak/Noise Ratio",
        tooltip="Effect: require clearer spectral peak.\nHigher values reduce false positives in noisy rooms.",
        min_v=1.0,
        max_v=40.0,
        step=0.5,
        decimals=2,
        value=processor.min_peak_prominence_ratio,
        on_change=lambda v: _apply_change(lambda x: setattr(processor, "min_peak_prominence_ratio", float(x)), v),
    )
    _add_slider_control(
        audio_form,
        label="Freq Smooth Alpha",
        tooltip="Effect: controls pitch responsiveness.\nHigher is faster; lower is smoother but laggier.",
        min_v=0.0,
        max_v=1.0,
        step=0.01,
        decimals=3,
        value=processor.frequency_smoothing_alpha,
        on_change=lambda v: _apply_change(lambda x: setattr(processor, "frequency_smoothing_alpha", float(x)), v),
    )
    _add_slider_control(
        memory_form,
        label="Memory Fade (s)",
        tooltip="Effect: lifetime of memory graph.\nHigher values keep historical structure longer.",
        min_v=0.1,
        max_v=20.0,
        step=0.1,
        decimals=2,
        value=helix_window.memory_fade_time_s,
        on_change=lambda v: _apply_change(lambda x: setattr(helix_window, "memory_fade_time_s", float(x)), v),
    )
    _add_slider_control(
        memory_form,
        label="Min Event Interval (ms)",
        tooltip="Effect: event debounce.\nHigher values reduce duplicate events; lower captures fast articulation.",
        min_v=1.0,
        max_v=500.0,
        step=1.0,
        decimals=1,
        value=helix_window.min_event_interval_s * 1000.0,
        on_change=lambda v: _apply_change(lambda x: setattr(helix_window, "min_event_interval_s", float(x) / 1000.0), v),
    )
    _add_slider_control(
        memory_form,
        label="Edge Window (ms)",
        tooltip="Effect: edge linking window.\nHigher values connect more notes; lower emphasizes immediate transitions.",
        min_v=0.0,
        max_v=1000.0,
        step=1.0,
        decimals=1,
        value=helix_window.edge_window_s * 1000.0,
        on_change=lambda v: _apply_change(lambda x: setattr(helix_window, "edge_window_s", float(x) / 1000.0), v),
    )
    _add_slider_control(
        memory_form,
        label="Face Opacity",
        tooltip="Effect: transparency of filled memory surfaces.\nLower values are subtle; higher values emphasize harmonic planes.",
        min_v=0.0,
        max_v=1.0,
        step=0.01,
        decimals=2,
        value=helix_window.memory_face_opacity,
        on_change=lambda v: _apply_change(lambda x: setattr(helix_window, "memory_face_opacity", float(x)), v),
    )
    camera_follow_chk = QtWidgets.QCheckBox("Camera Follow Note")
    camera_follow_chk.setChecked(bool(helix_window.camera_follow_note))
    camera_follow_chk.setToolTip("When enabled, camera azimuth follows the current dominant note.")
    camera_follow_chk.toggled.connect(lambda on: setattr(helix_window, "camera_follow_note", bool(on)))
    camera_form.addRow(camera_follow_chk)

    camera_advanced_chk = QtWidgets.QCheckBox("Show Camera Dynamics")
    camera_advanced_chk.setChecked(False)
    camera_advanced_chk.setToolTip("Reveal spring, damping, and momentum tuning sliders for camera follow.")
    camera_form.addRow(camera_advanced_chk)

    camera_advanced = QtWidgets.QWidget()
    camera_advanced_layout = QtWidgets.QFormLayout(camera_advanced)
    camera_advanced_layout.setContentsMargins(0, 0, 0, 0)
    camera_form.addRow(camera_advanced)

    _add_slider_control(
        camera_advanced_layout,
        label="Follow Spring",
        tooltip="How strongly camera is pulled toward the current note angle.\nHigher values snap faster.",
        min_v=0.001,
        max_v=0.2,
        step=0.001,
        decimals=3,
        value=float(helix_window._camera_follow_spring),
        on_change=lambda v: _apply_change(lambda x: setattr(helix_window, "_camera_follow_spring", float(x)), v),
    )
    _add_slider_control(
        camera_advanced_layout,
        label="Follow Damping",
        tooltip="Velocity damping per frame.\nLower values settle faster; higher values keep motion longer.",
        min_v=0.5,
        max_v=0.999,
        step=0.001,
        decimals=3,
        value=float(helix_window._camera_follow_damping),
        on_change=lambda v: _apply_change(lambda x: setattr(helix_window, "_camera_follow_damping", float(x)), v),
    )
    _add_slider_control(
        camera_advanced_layout,
        label="Momentum Gain",
        tooltip="Spin impulse added when pitch steps up/down.\nHigher values create stronger directional spin.",
        min_v=0.0,
        max_v=4.0,
        step=0.01,
        decimals=2,
        value=float(helix_window._camera_follow_momentum_gain),
        on_change=lambda v: _apply_change(lambda x: setattr(helix_window, "_camera_follow_momentum_gain", float(x)), v),
    )
    camera_advanced.setVisible(False)
    camera_advanced_chk.toggled.connect(camera_advanced.setVisible)

    status_label = QtWidgets.QLabel("")
    save_btn = QtWidgets.QPushButton("Save Config...")
    save_btn.setToolTip("Save current control values to a JSON profile.")

    def _current_config_payload() -> dict:
        return {
            "audio_processing": {
                "min_rms_threshold": float(processor.min_rms_threshold),
                "min_peak_prominence_ratio": float(processor.min_peak_prominence_ratio),
                "frequency_smoothing_alpha": float(processor.frequency_smoothing_alpha),
            },
            "ui": {
                "memory_fade_seconds": float(helix_window.memory_fade_time_s),
                "min_event_interval_ms": float(helix_window.min_event_interval_s * 1000.0),
                "edge_window_ms": float(helix_window.edge_window_s * 1000.0),
                "face_opacity": float(helix_window.memory_face_opacity),
                "camera_follow_note": bool(helix_window.camera_follow_note),
                "msaa_samples": int(msaa_samples),
            },
        }

    def _save_config() -> None:
        path = save_config_path
        if path is None:
            selected, _ = QtWidgets.QFileDialog.getSaveFileName(
                controls,
                "Save Helix Config",
                str(Path("configs") / "ui_profile.json"),
                "JSON Files (*.json)",
            )
            if not selected:
                return
            path = selected
        saved = save_config_file(path, _current_config_payload())
        status_label.setText(f"Saved: {saved}")

    save_btn.clicked.connect(_save_config)
    actions_form.addRow(save_btn)
    actions_form.addRow(status_label)

    recording_state: dict[str, object] = {
        "active": False,
        "frames_dir": None,
        "frame_index": 0,
    }
    record_btn = QtWidgets.QPushButton("Start Recording")
    record_btn.setToolTip("Record live UI and microphone input. Click again to stop and save MP4.")
    actions_form.addRow(record_btn)

    frame_timer = QtCore.QTimer(controls)
    frame_timer.setInterval(int(round(1000 / 30.0)))

    def _capture_frame() -> None:
        if not recording_state["active"]:
            return
        frames_dir = recording_state["frames_dir"]
        if not isinstance(frames_dir, Path):
            return
        img = helix_window.view.grabFramebuffer()
        if img.isNull():
            return
        idx = int(recording_state["frame_index"])
        out_file = frames_dir / f"frame_{idx:06d}.png"
        img.save(str(out_file), "PNG")
        recording_state["frame_index"] = idx + 1

    frame_timer.timeout.connect(_capture_frame)

    def _start_recording() -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path("outputs") / "live_recordings" / f"session_{ts}"
        frames_dir = session_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        recording_state["active"] = True
        recording_state["frames_dir"] = frames_dir
        recording_state["frame_index"] = 0
        processor.start_recording_capture()
        frame_timer.start()
        record_btn.setText("Stop Recording")
        status_label.setText(f"Recording... {session_dir}")

    def _stop_recording() -> None:
        frame_timer.stop()
        recording_state["active"] = False

        frames_dir = recording_state["frames_dir"]
        if not isinstance(frames_dir, Path):
            record_btn.setText("Start Recording")
            return
        session_dir = frames_dir.parent
        wav_path = session_dir / "audio.wav"
        mp4_path = session_dir.with_suffix(".mp4")
        audio = processor.stop_recording_capture()
        _write_wav_mono(path=wav_path, samples=audio, sample_rate=sr)
        frame_count = int(recording_state["frame_index"])
        if frame_count <= 0:
            status_label.setText("No frames captured. Recording discarded.")
            record_btn.setText("Start Recording")
            return

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            "30",
            "-i",
            str(frames_dir / "frame_%06d.png"),
            "-i",
            str(wav_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(mp4_path),
        ]
        status_label.setText("Encoding recording with ffmpeg...")
        app.processEvents()
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            status_label.setText(f"Saved recording: {mp4_path}")
        except FileNotFoundError:
            status_label.setText("ffmpeg not found. Saved raw frames and WAV only.")
        except subprocess.CalledProcessError:
            status_label.setText("ffmpeg failed. Saved raw frames and WAV only.")
        record_btn.setText("Start Recording")

    def _toggle_recording() -> None:
        if bool(recording_state["active"]):
            _stop_recording()
        else:
            _start_recording()

    record_btn.clicked.connect(_toggle_recording)
    controls.show()
    controls.raise_()
    controls.activateWindow()
    helix_window._controls_window = controls  # keep reference + ensure visibility

    processor.start()

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app.aboutToQuit.connect(processor.stop)
    app.aboutToQuit.connect(frame_timer.stop)
    app.aboutToQuit.connect(controls.close)

    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        processor.stop()
        app.quit()


def probe_frequencies(freqs: list[float], radius: float, pitch: float) -> int:
    guitar = _standard_guitar()
    min_freq = guitar.lowest_frequency()
    max_freq = guitar.highest_frequency()

    print("freq_hz,x,y,z,status")
    code = 0
    for freq in freqs:
        xyz = frequency_to_xyz(
            freq=freq,
            min_freq=min_freq,
            radius=radius,
            pitch=pitch,
            angular_offset_radians=D_BOTTOM_OFFSET_RADIANS,
        )
        in_range = min_freq <= freq <= max_freq
        status = "ok" if in_range else "out_of_range"
        if not in_range:
            code = 2
        print(f"{freq:.6f},{xyz[0]:.6f},{xyz[1]:.6f},{xyz[2]:.6f},{status}")
    return code


def _load_note_spans_from_source(
    midi_path: str | None,
    notes_json_path: str | None,
) -> tuple[list["NoteSpan"], float]:
    from helix_viz.midi_file import load_midi_timeline
    from helix_viz.note_io import load_note_spans_json

    if bool(midi_path) == bool(notes_json_path):
        raise ValueError("Provide exactly one of midi_path or notes_json_path.")

    if midi_path:
        timeline = load_midi_timeline(midi_path)
        return timeline.note_spans, timeline.duration_s

    spans = load_note_spans_json(notes_json_path)
    return spans, max((s.end_s for s in spans), default=0.0)


class _TimelineProcessor:
    """Minimal processor interface consumed by PitchHelixVisualizer."""

    def __init__(self) -> None:
        self.current_top_k_frequencies: list[float | None] = [None, None, None]


def _timeline_top_frequencies(
    spans: list["NoteSpan"],
    t_s: float,
    top_k: int = 3,
) -> list[float | None]:
    from helix_viz.pattern_memory import midi_note_to_frequency

    active = [span for span in spans if span.start_s <= t_s <= span.end_s]
    active.sort(key=lambda s: s.velocity, reverse=True)
    freqs: list[float | None] = [midi_note_to_frequency(s.midi_note) for s in active[:top_k]]
    while len(freqs) < top_k:
        freqs.append(None)
    return freqs


def _timeline_note_events_for_time(
    spans: list["NoteSpan"],
    t_s: float,
    memory_fade_time_s: float,
    event_interval_s: float,
) -> list["NoteEvent"]:
    from helix_viz.pattern_memory import NoteEvent

    if event_interval_s <= 0:
        return []

    events: list[NoteEvent] = []
    window_start = max(0.0, t_s - memory_fade_time_s)
    for span in spans:
        if span.start_s > t_s:
            continue
        seg_start = max(span.start_s, window_start)
        seg_end = min(span.end_s, t_s)
        if seg_end < seg_start:
            continue
        k0 = int(np.floor((seg_start - span.start_s) / event_interval_s))
        k1 = int(np.floor((seg_end - span.start_s) / event_interval_s))
        for k in range(k0, k1 + 1):
            ts = span.start_s + (k * event_interval_s)
            if seg_start <= ts <= seg_end:
                events.append(NoteEvent(timestamp_s=ts, midi_note=span.midi_note, source_id=span.channel))
    events.sort(key=lambda e: e.timestamp_s)
    return events


def _channel_color_map(spans: list["NoteSpan"]) -> dict[int, tuple[float, float, float]]:
    channels = sorted({span.channel for span in spans})
    if not channels:
        return {0: (1.0, 0.62, 0.1)}
    palette = [
        (1.0, 0.38, 0.18),  # warm orange-red
        (0.22, 0.68, 1.0),  # sky blue
        (0.25, 0.86, 0.49),  # green
        (0.92, 0.42, 1.0),  # magenta
        (1.0, 0.82, 0.24),  # yellow
        (0.50, 0.70, 1.0),  # light blue
    ]
    out: dict[int, tuple[float, float, float]] = {}
    for idx, channel in enumerate(channels):
        out[channel] = palette[idx % len(palette)]
    return out


def render_offline_video(
    midi_path: str | None,
    notes_json_path: str | None,
    output_mp4: str,
    fps: int,
    width: int,
    height: int,
    tail_seconds: float,
    duration_seconds: float | None,
    keep_frames: bool,
    frames_dir: str | None,
    memory_fade_time_s: float,
    edge_window_s: float,
    event_interval_s: float,
    supersample_scale: int,
) -> int:
    import time

    from helix_viz.offline_renderer import OfflineHelixRenderer, encode_frames_to_mp4, render_note_video_frames

    spans, base_duration_s = _load_note_spans_from_source(midi_path=midi_path, notes_json_path=notes_json_path)
    default_duration_s = base_duration_s + tail_seconds

    target_duration = duration_seconds if duration_seconds is not None else default_duration_s
    working_frames_dir: str
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if frames_dir is not None:
        working_frames_dir = frames_dir
    elif keep_frames:
        source_name = Path(midi_path or notes_json_path).stem
        working_frames_dir = str(Path("outputs") / f"{source_name}_frames")
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="helix_viz_frames_")
        working_frames_dir = temp_dir.name

    renderer = OfflineHelixRenderer(
        memory_fade_time_s=memory_fade_time_s,
        edge_window_s=edge_window_s,
        event_interval_s=event_interval_s,
        supersample_scale=supersample_scale,
    )

    try:
        frames_out, frame_count = render_note_video_frames(
            spans=spans,
            output_frames_dir=working_frames_dir,
            fps=fps,
            width=width,
            height=height,
            total_duration_s=target_duration,
            tail_seconds=tail_seconds,
            renderer=renderer,
        )
        print("Encoding MP4 with ffmpeg... this may take a while.")
        t0 = time.monotonic()
        try:
            encode_frames_to_mp4(frames_dir=frames_out, output_mp4=output_mp4, fps=fps)
        except FileNotFoundError:
            print("ffmpeg was not found on PATH. Install ffmpeg to encode MP4 output.")
            return 3
        except subprocess.CalledProcessError:
            print("ffmpeg failed while encoding MP4 output.")
            return 3
        dt = time.monotonic() - t0
        print(f"ffmpeg encoding finished in {dt:.1f}s.")
        print(f"Rendered {frame_count} frames to {frames_out}")
        print(f"Wrote video to {output_mp4}")
    finally:
        if temp_dir is not None and (not keep_frames):
            temp_dir.cleanup()

    return 0


def render_gl_video(
    midi_path: str | None,
    notes_json_path: str | None,
    output_mp4: str,
    fps: int,
    width: int,
    height: int,
    tail_seconds: float,
    duration_seconds: float | None,
    keep_frames: bool,
    frames_dir: str | None,
    memory_fade_time_s: float,
    min_event_interval_s: float,
    edge_window_s: float,
    face_opacity: float,
    msaa_samples: int,
    output_width: int | None,
    output_height: int | None,
) -> int:
    import math
    import time

    from PyQt5 import QtWidgets

    from helix_viz.pitch_helix_visualizer import PitchHelixVisualizer

    spans, base_duration_s = _load_note_spans_from_source(midi_path=midi_path, notes_json_path=notes_json_path)
    source_colors = _channel_color_map(spans)
    default_duration_s = base_duration_s + tail_seconds
    target_duration = duration_seconds if duration_seconds is not None else default_duration_s
    duration_s = max(0.0, float(target_duration))
    num_frames = max(1, int(math.ceil(duration_s * fps)))

    working_frames_dir: str
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if frames_dir is not None:
        working_frames_dir = frames_dir
    elif keep_frames:
        source_name = Path(midi_path or notes_json_path).stem
        working_frames_dir = str(Path("outputs") / f"{source_name}_gl_frames")
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="helix_viz_gl_frames_")
        working_frames_dir = temp_dir.name

    frame_path = Path(working_frames_dir)
    frame_path.mkdir(parents=True, exist_ok=True)

    app = _create_qapplication(msaa_samples=msaa_samples)
    processor = _TimelineProcessor()
    helix_window = PitchHelixVisualizer(
        processor=processor,  # type: ignore[arg-type]
        guitar_profile=_standard_guitar(),
        memory_fade_time_s=memory_fade_time_s,
        min_event_interval_s=min_event_interval_s,
        edge_window_s=edge_window_s,
        face_opacity=face_opacity,
    )
    # Pull camera back so full helix scale is visible in rendered video.
    helix_window.view.opts["distance"] = 68
    helix_window.set_external_timeline_events(note_events=[], now_s=0.0, source_color_map=source_colors)
    helix_window.setWindowTitle("Pitch Helix Visualizer - OpenGL Render")
    helix_window.resize(width, height)
    helix_window.show()
    app.processEvents()

    playback_started = False
    render_start_mono = time.monotonic()
    try:
        import sounddevice as sd

        live_audio = _synthesize_timeline_waveform(spans=spans, sample_rate=44100, duration_s=duration_s)
        if live_audio.size > 0:
            sd.play(live_audio, samplerate=44100, blocking=False)
            playback_started = True
            render_start_mono = time.monotonic()
    except Exception as exc:
        print(f"Render-gl audio playback unavailable: {exc}")

    try:
        for i in range(num_frames):
            # Keep encoded output deterministic and AV-synced.
            # Wall-clock based stepping can freeze visuals early when rendering is slower than realtime.
            t_s = i / float(fps)
            processor.current_top_k_frequencies = _timeline_top_frequencies(spans=spans, t_s=t_s, top_k=3)
            helix_window.set_external_timeline_events(
                note_events=_timeline_note_events_for_time(
                    spans=spans,
                    t_s=t_s,
                    memory_fade_time_s=memory_fade_time_s,
                    event_interval_s=min_event_interval_s,
                ),
                now_s=t_s,
            )
            helix_window.update_visualization()
            app.processEvents()
            image = helix_window.view.grabFramebuffer()
            if image.isNull():
                print("OpenGL framebuffer capture failed.")
                return 4
            out_file = frame_path / f"frame_{i:06d}.png"
            if not image.save(str(out_file), "PNG"):
                print(f"Failed to save frame: {out_file}")
                return 4

        output_path = Path(output_mp4)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path = frame_path / "timeline_audio.wav"
        audio = _synthesize_timeline_waveform(spans=spans, sample_rate=44100, duration_s=duration_s)
        _write_wav_mono(path=audio_path, samples=audio, sample_rate=44100)
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(int(fps)),
            "-i",
            str(frame_path / "frame_%06d.png"),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vcodec",
            "libx264",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
        ]
        if (output_width is not None) and (output_height is not None):
            cmd.extend(["-vf", f"scale={int(output_width)}:{int(output_height)}:flags=lanczos"])
        cmd.extend(
            [
                "-pix_fmt",
                "yuv420p",
            "-shortest",
            str(output_path),
            ]
        )
        print("Encoding MP4 with ffmpeg... this may take a while.")
        t0 = time.monotonic()
        try:
            ff = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            print("ffmpeg was not found on PATH. Install ffmpeg to encode MP4 output.")
            return 3
        except subprocess.CalledProcessError as exc:
            print("ffmpeg failed while encoding MP4 output.")
            err = (exc.stderr or "").strip()
            if err:
                tail = "\n".join(err.splitlines()[-20:])
                print(tail)
            return 3
        dt = time.monotonic() - t0
        print(f"ffmpeg encoding finished in {dt:.1f}s.")
        try:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=index",
                    "-of",
                    "csv=p=0",
                    str(output_path),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if not probe.stdout.strip():
                print("Rendered video has no audio stream.")
                return 5
        except FileNotFoundError:
            pass
        print(f"Rendered {num_frames} OpenGL frames to {frame_path}")
        print(f"Wrote video to {output_mp4}")
    finally:
        if playback_started:
            try:
                import sounddevice as sd

                sd.stop()
            except Exception:
                pass
        helix_window.close()
        app.quit()
        if temp_dir is not None and (not keep_frames):
            temp_dir.cleanup()

    return 0


def _synthesize_timeline_waveform(
    spans: list["NoteSpan"],
    sample_rate: int,
    duration_s: float,
) -> np.ndarray:
    from helix_viz.pattern_memory import midi_note_to_frequency

    if duration_s <= 0:
        return np.zeros((1,), dtype=np.float32)

    total_samples = max(1, int(round(duration_s * sample_rate)))
    audio = np.zeros((total_samples,), dtype=np.float64)
    attack_s = 0.005
    release_s = 0.02
    attack_n = max(1, int(round(attack_s * sample_rate)))
    release_n = max(1, int(round(release_s * sample_rate)))

    for span in spans:
        start_i = int(max(0, round(span.start_s * sample_rate)))
        end_i = int(min(total_samples, round(span.end_s * sample_rate)))
        if end_i <= start_i:
            continue

        n = end_i - start_i
        t = np.arange(n, dtype=np.float64) / float(sample_rate)
        freq = midi_note_to_frequency(span.midi_note)
        vel_gain = float(np.clip(span.velocity / 127.0, 0.0, 1.0))
        signal_buf = np.sin(2.0 * np.pi * freq * t) * vel_gain

        env = np.ones((n,), dtype=np.float64)
        a = min(attack_n, n)
        r = min(release_n, n)
        if a > 1:
            env[:a] = np.linspace(0.0, 1.0, a, endpoint=True)
        if r > 1:
            env[-r:] *= np.linspace(1.0, 0.0, r, endpoint=True)
        signal_buf *= env
        audio[start_i:end_i] += signal_buf

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-8:
        audio = audio / max(1.0, peak * 1.1)
    return audio.astype(np.float32)


def _write_wav_mono(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(output), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return output


def play_timeline_audio(
    midi_path: str | None,
    notes_json_path: str | None,
    sample_rate: int,
    tail_seconds: float,
    duration_seconds: float | None,
    output_wav: str | None,
    no_playback: bool,
) -> int:
    spans, base_duration_s = _load_note_spans_from_source(midi_path=midi_path, notes_json_path=notes_json_path)
    default_duration_s = base_duration_s + tail_seconds
    duration_s = max(0.0, float(duration_seconds if duration_seconds is not None else default_duration_s))
    audio = _synthesize_timeline_waveform(spans=spans, sample_rate=sample_rate, duration_s=duration_s)

    if output_wav is not None:
        wav_path = _write_wav_mono(path=output_wav, samples=audio, sample_rate=sample_rate)
        print(f"Wrote synthesized audio to {wav_path}")

    if not no_playback:
        import sounddevice as sd

        sd.play(audio, samplerate=sample_rate, blocking=True)
        print("Playback complete.")

    return 0


def launch_timeline_preview(
    midi_path: str | None,
    notes_json_path: str | None,
    memory_fade_time_s: float = 2.6,
    min_event_interval_s: float = 0.08,
    edge_window_s: float = 0.22,
    speed: float = 1.0,
    no_audio: bool = False,
    audio_sample_rate: int = 44100,
    face_opacity: float = 1.0,
    msaa_samples: int = 4,
    save_config_path: str | None = None,
) -> None:
    import time

    from PyQt5 import QtCore, QtWidgets

    from helix_viz.pitch_helix_visualizer import PitchHelixVisualizer

    spans, duration_s = _load_note_spans_from_source(midi_path=midi_path, notes_json_path=notes_json_path)
    source_colors = _channel_color_map(spans)

    app = _create_qapplication(msaa_samples=msaa_samples)
    processor = _TimelineProcessor()
    standard_guitar = _standard_guitar()

    helix_window = PitchHelixVisualizer(
        processor=processor,  # type: ignore[arg-type]
        guitar_profile=standard_guitar,
        memory_fade_time_s=memory_fade_time_s,
        min_event_interval_s=min_event_interval_s,
        edge_window_s=edge_window_s,
        face_opacity=face_opacity,
    )
    # Match render-gl framing for easier comparison between preview and output.
    helix_window.view.opts["distance"] = 68
    helix_window.set_external_timeline_events(note_events=[], now_s=0.0, source_color_map=source_colors)
    helix_window.setWindowTitle("Pitch Helix Visualizer - Timeline Preview")
    helix_window.resize(900, 700)
    helix_window.show()

    controls = QtWidgets.QWidget()
    controls.setWindowTitle("Timeline Controls")
    controls.setWindowFlag(QtCore.Qt.Tool, True)
    controls.setAttribute(QtCore.Qt.WA_AlwaysShowToolTips, True)
    controls.setMouseTracking(True)
    controls_layout = QtWidgets.QVBoxLayout(controls)
    controls.setStyleSheet(
        """
        QWidget {
            background-color: #101722;
            color: #e6edf3;
        }
        QDoubleSpinBox {
            background-color: #1a2332;
            border: 1px solid #2b3a4f;
            border-radius: 4px;
            padding: 4px 6px;
            color: #e6edf3;
        }
        QLabel {
            color: #d0d7de;
        }
        QPushButton {
            background-color: #243247;
            border: 1px solid #3b4d66;
            border-radius: 6px;
            color: #e6edf3;
            padding: 6px 10px;
        }
        QPushButton:hover {
            background-color: #2b3d56;
        }
        QPushButton:pressed {
            background-color: #1f2d40;
        }
        QToolTip {
            background-color: #182436;
            color: #ffd866;
            border: 1px solid #4d6482;
            padding: 6px;
        }
        """
    )

    def _add_group(title: str) -> QtWidgets.QFormLayout:
        group = QtWidgets.QGroupBox(title)
        group_form = QtWidgets.QFormLayout(group)
        controls_layout.addWidget(group)
        return group_form

    memory_form = _add_group("Memory")
    playback_form = _add_group("Playback")
    actions_form = _add_group("Actions")

    state = {
        "memory_fade_time_s": float(memory_fade_time_s),
        "min_event_interval_s": float(min_event_interval_s),
        "edge_window_s": float(edge_window_s),
        "face_opacity": float(face_opacity),
        "speed": float(speed),
    }

    def _make_spin(min_v: float, max_v: float, step: float, value: float, decimals: int = 3) -> QtWidgets.QDoubleSpinBox:
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(min_v, max_v)
        w.setSingleStep(step)
        w.setDecimals(decimals)
        w.setValue(float(value))
        return w

    def _add_slider_control(
        target_form: QtWidgets.QFormLayout,
        label: str,
        tooltip: str,
        min_v: float,
        max_v: float,
        step: float,
        decimals: int,
        value: float,
        on_change,
    ) -> None:
        scale = max(1, int(round(1.0 / step)))
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(int(round(min_v * scale)), int(round(max_v * scale)))
        slider.setSingleStep(1)
        slider.setPageStep(max(1, int(scale * step * 10)))
        spin = _make_spin(min_v=min_v, max_v=max_v, step=step, value=value, decimals=decimals)
        container = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(slider, 2)
        row.addWidget(spin, 1)
        slider.setToolTip(tooltip)
        spin.setToolTip(tooltip)
        container.setToolTip(tooltip)

        def _from_slider(ivalue: int) -> None:
            fvalue = ivalue / float(scale)
            if abs(spin.value() - fvalue) > (0.5 / scale):
                spin.blockSignals(True)
                spin.setValue(fvalue)
                spin.blockSignals(False)
            on_change(float(spin.value()))

        def _from_spin(fvalue: float) -> None:
            ivalue = int(round(fvalue * scale))
            if slider.value() != ivalue:
                slider.blockSignals(True)
                slider.setValue(ivalue)
                slider.blockSignals(False)
            on_change(float(fvalue))

        slider.valueChanged.connect(_from_slider)
        spin.valueChanged.connect(_from_spin)
        slider.setValue(int(round(value * scale)))
        label_widget = QtWidgets.QLabel(label)
        label_widget.setToolTip(tooltip)
        target_form.addRow(label_widget, container)

    _add_slider_control(
        memory_form,
        label="Memory Fade (s)",
        tooltip="Effect: lifetime of memory graph.\nHigher values keep historical structure longer.",
        min_v=0.1,
        max_v=20.0,
        step=0.1,
        decimals=2,
        value=state["memory_fade_time_s"],
        on_change=lambda v: (state.__setitem__("memory_fade_time_s", float(v)), setattr(helix_window, "memory_fade_time_s", float(v))),
    )
    _add_slider_control(
        memory_form,
        label="Min Event Interval (ms)",
        tooltip="Effect: event debounce.\nHigher values reduce duplicate events; lower captures fast articulation.",
        min_v=1.0,
        max_v=500.0,
        step=1.0,
        decimals=1,
        value=state["min_event_interval_s"] * 1000.0,
        on_change=lambda v: (
            state.__setitem__("min_event_interval_s", float(v) / 1000.0),
            setattr(helix_window, "min_event_interval_s", float(v) / 1000.0),
            timer.setInterval(max(10, int(float(v)))),
        ),
    )
    _add_slider_control(
        memory_form,
        label="Edge Window (ms)",
        tooltip="Effect: edge linking window.\nHigher values connect more notes; lower emphasizes immediate transitions.",
        min_v=0.0,
        max_v=1000.0,
        step=1.0,
        decimals=1,
        value=state["edge_window_s"] * 1000.0,
        on_change=lambda v: (state.__setitem__("edge_window_s", float(v) / 1000.0), setattr(helix_window, "edge_window_s", float(v) / 1000.0)),
    )
    _add_slider_control(
        memory_form,
        label="Face Opacity",
        tooltip="Effect: transparency of filled memory surfaces.\nLower values are subtle; higher values emphasize harmonic planes.",
        min_v=0.0,
        max_v=1.0,
        step=0.01,
        decimals=2,
        value=state["face_opacity"],
        on_change=lambda v: (state.__setitem__("face_opacity", float(v)), setattr(helix_window, "memory_face_opacity", float(v))),
    )
    _add_slider_control(
        playback_form,
        label="Speed",
        tooltip="Effect: timeline playback speed.\nHigher values play faster.",
        min_v=0.1,
        max_v=3.0,
        step=0.05,
        decimals=2,
        value=state["speed"],
        on_change=lambda v: state.__setitem__("speed", float(v)),
    )

    status_label = QtWidgets.QLabel("")
    save_btn = QtWidgets.QPushButton("Save Config...")
    save_btn.setToolTip("Save current preview/render-gl tuning to a JSON profile.")

    def _save_config() -> None:
        path = save_config_path
        if path is None:
            selected, _ = QtWidgets.QFileDialog.getSaveFileName(
                controls,
                "Save Timeline Config",
                str(Path("configs") / "timeline_profile.json"),
                "JSON Files (*.json)",
            )
            if not selected:
                return
            path = selected
        payload = {
            "preview": {
                "memory_fade_seconds": float(state["memory_fade_time_s"]),
                "min_event_interval_ms": float(state["min_event_interval_s"] * 1000.0),
                "edge_window_ms": float(state["edge_window_s"] * 1000.0),
                "face_opacity": float(state["face_opacity"]),
                "speed": float(state["speed"]),
                "audio_sample_rate": int(audio_sample_rate),
                "no_audio": bool(no_audio),
                "msaa_samples": int(msaa_samples),
            },
            "render_gl": {
                "memory_fade_seconds": float(state["memory_fade_time_s"]),
                "min_event_interval_ms": float(state["min_event_interval_s"] * 1000.0),
                "edge_window_ms": float(state["edge_window_s"] * 1000.0),
                "face_opacity": float(state["face_opacity"]),
                "msaa_samples": int(msaa_samples),
            },
        }
        saved = save_config_file(path, payload)
        status_label.setText(f"Saved: {saved}")

    save_btn.clicked.connect(_save_config)
    actions_form.addRow(save_btn)
    actions_form.addRow(status_label)
    controls.show()
    controls.raise_()
    controls.activateWindow()
    helix_window._controls_window = controls  # keep reference + ensure visibility

    start_mono = time.monotonic()

    def tick() -> None:
        elapsed = (time.monotonic() - start_mono) * float(state["speed"])
        processor.current_top_k_frequencies = _timeline_top_frequencies(spans=spans, t_s=elapsed, top_k=3)
        helix_window.set_external_timeline_events(
            note_events=_timeline_note_events_for_time(
                spans=spans,
                t_s=elapsed,
                memory_fade_time_s=float(state["memory_fade_time_s"]),
                event_interval_s=float(state["min_event_interval_s"]),
            ),
            now_s=elapsed,
        )
        if elapsed > (duration_s + float(state["memory_fade_time_s"])):
            app.quit()

    timer = QtCore.QTimer()
    timer.setInterval(max(10, int(float(state["min_event_interval_s"]) * 1000)))
    timer.timeout.connect(tick)
    timer.start()
    tick()

    playback_started = False
    if (midi_path is not None) and (not no_audio):
        try:
            import sounddevice as sd

            audio = _synthesize_timeline_waveform(
                spans=spans,
                sample_rate=audio_sample_rate,
                duration_s=duration_s + float(state["memory_fade_time_s"]),
            )
            if audio.size > 0:
                # Playback speed also affects pitch in this minimal implementation.
                sd.play(
                    audio,
                    samplerate=max(1, int(round(audio_sample_rate * float(state["speed"])))),
                    blocking=False,
                )
                playback_started = True
        except Exception as exc:
            print(f"Preview audio playback unavailable: {exc}")

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        app.exec()
    finally:
        controls.close()
        if playback_started:
            try:
                import sounddevice as sd

                sd.stop()
            except Exception:
                pass


def _add_config_args(cmd_parser: argparse.ArgumentParser) -> None:
    cmd_parser.add_argument("--config", type=str, default=None, help="Load parameters from a JSON config file.")
    cmd_parser.add_argument("--save-config", type=str, default=None, help="Save resolved parameters to a JSON config file.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pitch helix visualizer and probe utilities")
    subparsers = parser.add_subparsers(dest="command")

    ui = subparsers.add_parser("ui", help="Run the realtime Qt visualizer")
    ui.add_argument(
        "--min-rms-threshold",
        type=float,
        default=0.008,
        help="Hide pitch when RMS falls below this level (default: 0.008).",
    )
    ui.add_argument(
        "--min-peak-prominence-ratio",
        type=float,
        default=10.0,
        help="Required dominant FFT peak/noise-floor ratio (default: 10.0).",
    )
    ui.add_argument(
        "--frequency-smoothing-alpha",
        type=float,
        default=0.28,
        help="Smoothing factor in [0,1], higher responds faster (default: 0.28).",
    )
    ui.add_argument(
        "--memory-fade-seconds",
        type=float,
        default=2.6,
        help="Fading memory duration for note-graph persistence (default: 2.6).",
    )
    ui.add_argument(
        "--min-event-interval-ms",
        type=float,
        default=80.0,
        help="Debounce per pitch-class to avoid duplicate spam (default: 80ms).",
    )
    ui.add_argument(
        "--edge-window-ms",
        type=float,
        default=220.0,
        help="Connect notes only when event distance is within this window (default: 220ms).",
    )
    ui.add_argument(
        "--face-opacity",
        type=float,
        default=1.0,
        help="Opacity multiplier for memory fill faces in [0,1] (default: 1.0).",
    )
    ui.add_argument(
        "--camera-follow-note",
        action="store_true",
        help="Enable camera azimuth follow mode based on dominant note.",
    )
    ui.add_argument(
        "--msaa-samples",
        type=int,
        default=4,
        help="OpenGL MSAA samples for smoother rendering (default: 4).",
    )
    _add_config_args(ui)

    probe = subparsers.add_parser("probe", help="Print helix coordinates for frequencies")
    probe.add_argument(
        "--freq",
        dest="freqs",
        type=float,
        action="append",
        required=True,
        help="Frequency in Hz. Pass multiple --freq values to probe multiple points.",
    )
    probe.add_argument("--radius", type=float, default=10.0, help="Helix radius (default: 10.0)")
    probe.add_argument("--pitch", type=float, default=3.0, help="Helix pitch constant (default: 3.0)")

    render = subparsers.add_parser("render", help="Render offline video from MIDI or note spans")
    source_group = render.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--midi", type=str, help="Input MIDI file path (.mid/.midi).")
    source_group.add_argument(
        "--notes-json",
        type=str,
        help="Input JSON file with note spans: [{start_s,end_s,midi_note,velocity?}, ...].",
    )
    render.add_argument("--output", type=str, required=True, help="Output MP4 path.")
    render.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30).")
    render.add_argument("--width", type=int, default=960, help="Frame width in pixels (default: 960).")
    render.add_argument("--height", type=int, default=540, help="Frame height in pixels (default: 540).")
    render.add_argument("--tail-seconds", type=float, default=1.0, help="Extra tail after content (default: 1.0).")
    render.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Override total duration. If omitted, derived from note timeline.",
    )
    render.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="Directory for intermediate PPM frames (default: temporary dir).",
    )
    render.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep intermediate frames when temporary frame directory is used.",
    )
    render.add_argument(
        "--memory-fade-seconds",
        type=float,
        default=2.6,
        help="Fading memory duration used for node/edge persistence (default: 2.6).",
    )
    render.add_argument(
        "--edge-window-ms",
        type=float,
        default=220.0,
        help="Edge connect window in milliseconds (default: 220).",
    )
    render.add_argument(
        "--event-interval-ms",
        type=float,
        default=80.0,
        help="Synthetic event spacing from note spans in milliseconds (default: 80).",
    )
    render.add_argument(
        "--supersample-scale",
        type=int,
        default=2,
        help="Software antialiasing scale factor (1 disables, default: 2).",
    )
    _add_config_args(render)

    render_gl = subparsers.add_parser("render-gl", help="Render MP4 via the OpenGL Qt visualizer")
    render_gl_source = render_gl.add_mutually_exclusive_group(required=True)
    render_gl_source.add_argument("--midi", type=str, help="Input MIDI file path (.mid/.midi).")
    render_gl_source.add_argument(
        "--notes-json",
        type=str,
        help="Input JSON file with note spans: [{start_s,end_s,midi_note,velocity?}, ...].",
    )
    render_gl.add_argument("--output", type=str, required=True, help="Output MP4 path.")
    render_gl.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30).")
    render_gl.add_argument("--width", type=int, default=960, help="Window/frame width in pixels (default: 960).")
    render_gl.add_argument("--height", type=int, default=540, help="Window/frame height in pixels (default: 540).")
    render_gl.add_argument(
        "--tail-seconds",
        type=float,
        default=1.0,
        help="Extra tail after content (default: 1.0).",
    )
    render_gl.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Override total duration. If omitted, derived from note timeline.",
    )
    render_gl.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="Directory for intermediate PNG frames (default: temporary dir).",
    )
    render_gl.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep intermediate frames when temporary frame directory is used.",
    )
    render_gl.add_argument(
        "--memory-fade-seconds",
        type=float,
        default=2.6,
        help="Fading memory duration used for node/edge persistence (default: 2.6).",
    )
    render_gl.add_argument(
        "--min-event-interval-ms",
        type=float,
        default=80.0,
        help="Debounce interval in milliseconds for memory events (default: 80).",
    )
    render_gl.add_argument(
        "--edge-window-ms",
        type=float,
        default=220.0,
        help="Edge connect window in milliseconds (default: 220).",
    )
    render_gl.add_argument(
        "--face-opacity",
        type=float,
        default=1.0,
        help="Opacity multiplier for memory fill faces in [0,1] (default: 1.0).",
    )
    render_gl.add_argument(
        "--msaa-samples",
        type=int,
        default=4,
        help="OpenGL MSAA samples for smoother rendering (default: 4).",
    )
    render_gl.add_argument(
        "--output-width",
        type=int,
        default=None,
        help="Optional encoded output width (downscale from render size).",
    )
    render_gl.add_argument(
        "--output-height",
        type=int,
        default=None,
        help="Optional encoded output height (downscale from render size).",
    )
    _add_config_args(render_gl)

    play = subparsers.add_parser("play", help="Play MIDI/JSON timeline as synthesized audio")
    play_source = play.add_mutually_exclusive_group(required=True)
    play_source.add_argument("--midi", type=str, help="Input MIDI file path (.mid/.midi).")
    play_source.add_argument(
        "--notes-json",
        type=str,
        help="Input JSON file with note spans: [{start_s,end_s,midi_note,velocity?}, ...].",
    )
    play.add_argument("--sample-rate", type=int, default=44100, help="Audio sample rate (default: 44100).")
    play.add_argument("--tail-seconds", type=float, default=0.5, help="Extra tail after content (default: 0.5).")
    play.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Override total duration. If omitted, derived from note timeline.",
    )
    play.add_argument("--output-wav", type=str, default=None, help="Optional output WAV file path.")
    play.add_argument(
        "--no-playback",
        action="store_true",
        help="Generate audio without playing it through the output device.",
    )
    _add_config_args(play)

    preview = subparsers.add_parser("preview", help="Play MIDI/JSON timeline in realtime Qt visualizer")
    preview_source = preview.add_mutually_exclusive_group(required=True)
    preview_source.add_argument("--midi", type=str, help="Input MIDI file path (.mid/.midi).")
    preview_source.add_argument(
        "--notes-json",
        type=str,
        help="Input JSON file with note spans: [{start_s,end_s,midi_note,velocity?}, ...].",
    )
    preview.add_argument(
        "--memory-fade-seconds",
        type=float,
        default=2.6,
        help="Fading memory duration for note-graph persistence (default: 2.6).",
    )
    preview.add_argument(
        "--min-event-interval-ms",
        type=float,
        default=80.0,
        help="Debounce interval in milliseconds (default: 80).",
    )
    preview.add_argument(
        "--edge-window-ms",
        type=float,
        default=220.0,
        help="Connect notes only when event distance is within this window (default: 220).",
    )
    preview.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0).",
    )
    preview.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable synthesized timeline audio during preview.",
    )
    preview.add_argument(
        "--audio-sample-rate",
        type=int,
        default=44100,
        help="Audio sample rate for preview playback (default: 44100).",
    )
    preview.add_argument(
        "--face-opacity",
        type=float,
        default=1.0,
        help="Opacity multiplier for memory fill faces in [0,1] (default: 1.0).",
    )
    preview.add_argument(
        "--msaa-samples",
        type=int,
        default=4,
        help="OpenGL MSAA samples for smoother rendering (default: 4).",
    )
    _add_config_args(preview)
    return parser


def _validate_ui_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.min_rms_threshold < 0:
        parser.error("--min-rms-threshold must be >= 0.")
    if args.min_peak_prominence_ratio <= 0:
        parser.error("--min-peak-prominence-ratio must be > 0.")
    if not (0.0 <= args.frequency_smoothing_alpha <= 1.0):
        parser.error("--frequency-smoothing-alpha must be between 0 and 1.")
    if args.memory_fade_seconds <= 0:
        parser.error("--memory-fade-seconds must be > 0.")
    if args.min_event_interval_ms < 0:
        parser.error("--min-event-interval-ms must be >= 0.")
    if args.edge_window_ms < 0:
        parser.error("--edge-window-ms must be >= 0.")
    if not (0.0 <= args.face_opacity <= 1.0):
        parser.error("--face-opacity must be between 0 and 1.")
    if args.msaa_samples < 0:
        parser.error("--msaa-samples must be >= 0.")


def _validate_render_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.fps <= 0:
        parser.error("--fps must be > 0.")
    if args.width <= 0:
        parser.error("--width must be > 0.")
    if args.height <= 0:
        parser.error("--height must be > 0.")
    if args.tail_seconds < 0:
        parser.error("--tail-seconds must be >= 0.")
    if args.duration_seconds is not None and args.duration_seconds < 0:
        parser.error("--duration-seconds must be >= 0 when provided.")
    if args.memory_fade_seconds <= 0:
        parser.error("--memory-fade-seconds must be > 0.")
    if args.edge_window_ms < 0:
        parser.error("--edge-window-ms must be >= 0.")
    if args.event_interval_ms <= 0:
        parser.error("--event-interval-ms must be > 0.")
    if args.supersample_scale <= 0:
        parser.error("--supersample-scale must be > 0.")


def _validate_render_gl_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.fps <= 0:
        parser.error("--fps must be > 0.")
    if args.width <= 0:
        parser.error("--width must be > 0.")
    if args.height <= 0:
        parser.error("--height must be > 0.")
    if args.tail_seconds < 0:
        parser.error("--tail-seconds must be >= 0.")
    if args.duration_seconds is not None and args.duration_seconds < 0:
        parser.error("--duration-seconds must be >= 0 when provided.")
    if args.memory_fade_seconds <= 0:
        parser.error("--memory-fade-seconds must be > 0.")
    if args.min_event_interval_ms <= 0:
        parser.error("--min-event-interval-ms must be > 0.")
    if args.edge_window_ms < 0:
        parser.error("--edge-window-ms must be >= 0.")
    if not (0.0 <= args.face_opacity <= 1.0):
        parser.error("--face-opacity must be between 0 and 1.")
    if args.msaa_samples < 0:
        parser.error("--msaa-samples must be >= 0.")
    if (args.output_width is None) != (args.output_height is None):
        parser.error("--output-width and --output-height must be provided together.")
    if args.output_width is not None and args.output_width <= 0:
        parser.error("--output-width must be > 0.")
    if args.output_height is not None and args.output_height <= 0:
        parser.error("--output-height must be > 0.")


def _validate_preview_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.memory_fade_seconds <= 0:
        parser.error("--memory-fade-seconds must be > 0.")
    if args.min_event_interval_ms <= 0:
        parser.error("--min-event-interval-ms must be > 0.")
    if args.edge_window_ms < 0:
        parser.error("--edge-window-ms must be >= 0.")
    if args.speed <= 0:
        parser.error("--speed must be > 0.")
    if args.audio_sample_rate <= 0:
        parser.error("--audio-sample-rate must be > 0.")
    if not (0.0 <= args.face_opacity <= 1.0):
        parser.error("--face-opacity must be between 0 and 1.")
    if args.msaa_samples < 0:
        parser.error("--msaa-samples must be >= 0.")


def _validate_play_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.sample_rate <= 0:
        parser.error("--sample-rate must be > 0.")
    if args.tail_seconds < 0:
        parser.error("--tail-seconds must be >= 0.")
    if args.duration_seconds is not None and args.duration_seconds < 0:
        parser.error("--duration-seconds must be >= 0 when provided.")


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(raw_argv)

    if args.command in (None, "ui"):
        if args.command == "ui":
            cfg = _load_effective_config(args.config)
            ui_cfg = cfg.get("ui", {})
            ap_cfg = cfg.get("audio_processing", {})
            min_rms_threshold = float(
                _resolve_value(raw_argv, "--min-rms-threshold", args.min_rms_threshold, ap_cfg, "min_rms_threshold")
            )
            min_peak_prominence_ratio = float(
                _resolve_value(
                    raw_argv,
                    "--min-peak-prominence-ratio",
                    args.min_peak_prominence_ratio,
                    ap_cfg,
                    "min_peak_prominence_ratio",
                )
            )
            frequency_smoothing_alpha = float(
                _resolve_value(
                    raw_argv,
                    "--frequency-smoothing-alpha",
                    args.frequency_smoothing_alpha,
                    ap_cfg,
                    "frequency_smoothing_alpha",
                )
            )
            memory_fade_seconds = float(
                _resolve_value(raw_argv, "--memory-fade-seconds", args.memory_fade_seconds, ui_cfg, "memory_fade_seconds")
            )
            min_event_interval_ms = float(
                _resolve_value(
                    raw_argv,
                    "--min-event-interval-ms",
                    args.min_event_interval_ms,
                    ui_cfg,
                    "min_event_interval_ms",
                )
            )
            edge_window_ms = float(
                _resolve_value(raw_argv, "--edge-window-ms", args.edge_window_ms, ui_cfg, "edge_window_ms")
            )
            face_opacity = float(
                _resolve_value(raw_argv, "--face-opacity", args.face_opacity, ui_cfg, "face_opacity")
            )
            camera_follow_note = bool(
                _resolve_value(raw_argv, "--camera-follow-note", args.camera_follow_note, ui_cfg, "camera_follow_note")
            )
            msaa_samples = int(_resolve_value(raw_argv, "--msaa-samples", args.msaa_samples, ui_cfg, "msaa_samples"))
            args.min_rms_threshold = min_rms_threshold
            args.min_peak_prominence_ratio = min_peak_prominence_ratio
            args.frequency_smoothing_alpha = frequency_smoothing_alpha
            args.memory_fade_seconds = memory_fade_seconds
            args.min_event_interval_ms = min_event_interval_ms
            args.edge_window_ms = edge_window_ms
            args.face_opacity = face_opacity
            args.camera_follow_note = camera_follow_note
            args.msaa_samples = msaa_samples
            _validate_ui_args(parser, args)
            resolved_config = {
                "audio_processing": {
                    "min_rms_threshold": min_rms_threshold,
                    "min_peak_prominence_ratio": min_peak_prominence_ratio,
                    "frequency_smoothing_alpha": frequency_smoothing_alpha,
                },
                "ui": {
                    "memory_fade_seconds": memory_fade_seconds,
                    "min_event_interval_ms": min_event_interval_ms,
                    "edge_window_ms": edge_window_ms,
                    "face_opacity": face_opacity,
                    "camera_follow_note": camera_follow_note,
                    "msaa_samples": msaa_samples,
                },
            }
            if args.save_config:
                saved = save_config_file(args.save_config, resolved_config)
                print(f"Saved config to {saved}")
            launch_ui(
                min_rms_threshold=min_rms_threshold,
                min_peak_prominence_ratio=min_peak_prominence_ratio,
                frequency_smoothing_alpha=frequency_smoothing_alpha,
                memory_fade_time_s=memory_fade_seconds,
                min_event_interval_s=min_event_interval_ms / 1000.0,
                edge_window_s=edge_window_ms / 1000.0,
                face_opacity=face_opacity,
                camera_follow_note=camera_follow_note,
                msaa_samples=msaa_samples,
                save_config_path=args.save_config,
            )
        else:
            launch_ui()
        return

    if args.command == "probe":
        raise SystemExit(probe_frequencies(freqs=args.freqs, radius=args.radius, pitch=args.pitch))

    if args.command == "render":
        cfg = _load_effective_config(args.config)
        rcfg = cfg.get("render", {})
        args.fps = int(_resolve_value(raw_argv, "--fps", args.fps, rcfg, "fps"))
        args.width = int(_resolve_value(raw_argv, "--width", args.width, rcfg, "width"))
        args.height = int(_resolve_value(raw_argv, "--height", args.height, rcfg, "height"))
        args.tail_seconds = float(_resolve_value(raw_argv, "--tail-seconds", args.tail_seconds, rcfg, "tail_seconds"))
        args.duration_seconds = _resolve_value(
            raw_argv,
            "--duration-seconds",
            args.duration_seconds,
            rcfg,
            "duration_seconds",
        )
        args.memory_fade_seconds = float(
            _resolve_value(raw_argv, "--memory-fade-seconds", args.memory_fade_seconds, rcfg, "memory_fade_seconds")
        )
        args.edge_window_ms = float(_resolve_value(raw_argv, "--edge-window-ms", args.edge_window_ms, rcfg, "edge_window_ms"))
        args.event_interval_ms = float(
            _resolve_value(raw_argv, "--event-interval-ms", args.event_interval_ms, rcfg, "event_interval_ms")
        )
        args.supersample_scale = int(
            _resolve_value(raw_argv, "--supersample-scale", args.supersample_scale, rcfg, "supersample_scale")
        )
        _validate_render_args(parser, args)
        if args.save_config:
            saved = save_config_file(
                args.save_config,
                {
                    "render": {
                        "fps": args.fps,
                        "width": args.width,
                        "height": args.height,
                        "tail_seconds": args.tail_seconds,
                        "duration_seconds": args.duration_seconds,
                        "memory_fade_seconds": args.memory_fade_seconds,
                        "edge_window_ms": args.edge_window_ms,
                        "event_interval_ms": args.event_interval_ms,
                        "supersample_scale": args.supersample_scale,
                    }
                },
            )
            print(f"Saved config to {saved}")
        raise SystemExit(
            render_offline_video(
                midi_path=args.midi,
                notes_json_path=args.notes_json,
                output_mp4=args.output,
                fps=args.fps,
                width=args.width,
                height=args.height,
                tail_seconds=args.tail_seconds,
                duration_seconds=args.duration_seconds,
                keep_frames=args.keep_frames,
                frames_dir=args.frames_dir,
                memory_fade_time_s=args.memory_fade_seconds,
                edge_window_s=args.edge_window_ms / 1000.0,
                event_interval_s=args.event_interval_ms / 1000.0,
                supersample_scale=args.supersample_scale,
            )
        )

    if args.command == "render-gl":
        cfg = _load_effective_config(args.config)
        rcfg = cfg.get("render_gl", {})
        args.fps = int(_resolve_value(raw_argv, "--fps", args.fps, rcfg, "fps"))
        args.width = int(_resolve_value(raw_argv, "--width", args.width, rcfg, "width"))
        args.height = int(_resolve_value(raw_argv, "--height", args.height, rcfg, "height"))
        args.tail_seconds = float(_resolve_value(raw_argv, "--tail-seconds", args.tail_seconds, rcfg, "tail_seconds"))
        args.duration_seconds = _resolve_value(
            raw_argv,
            "--duration-seconds",
            args.duration_seconds,
            rcfg,
            "duration_seconds",
        )
        args.memory_fade_seconds = float(
            _resolve_value(raw_argv, "--memory-fade-seconds", args.memory_fade_seconds, rcfg, "memory_fade_seconds")
        )
        args.min_event_interval_ms = float(
            _resolve_value(
                raw_argv,
                "--min-event-interval-ms",
                args.min_event_interval_ms,
                rcfg,
                "min_event_interval_ms",
            )
        )
        args.edge_window_ms = float(_resolve_value(raw_argv, "--edge-window-ms", args.edge_window_ms, rcfg, "edge_window_ms"))
        args.face_opacity = float(_resolve_value(raw_argv, "--face-opacity", args.face_opacity, rcfg, "face_opacity"))
        args.msaa_samples = int(_resolve_value(raw_argv, "--msaa-samples", args.msaa_samples, rcfg, "msaa_samples"))
        args.output_width = _resolve_value(raw_argv, "--output-width", args.output_width, rcfg, "output_width")
        args.output_height = _resolve_value(raw_argv, "--output-height", args.output_height, rcfg, "output_height")
        _validate_render_gl_args(parser, args)
        if args.save_config:
            saved = save_config_file(
                args.save_config,
                {
                    "render_gl": {
                        "fps": args.fps,
                        "width": args.width,
                        "height": args.height,
                        "tail_seconds": args.tail_seconds,
                        "duration_seconds": args.duration_seconds,
                        "memory_fade_seconds": args.memory_fade_seconds,
                        "min_event_interval_ms": args.min_event_interval_ms,
                        "edge_window_ms": args.edge_window_ms,
                        "face_opacity": args.face_opacity,
                        "msaa_samples": args.msaa_samples,
                        "output_width": args.output_width,
                        "output_height": args.output_height,
                    }
                },
            )
            print(f"Saved config to {saved}")
        raise SystemExit(
            render_gl_video(
                midi_path=args.midi,
                notes_json_path=args.notes_json,
                output_mp4=args.output,
                fps=args.fps,
                width=args.width,
                height=args.height,
                tail_seconds=args.tail_seconds,
                duration_seconds=args.duration_seconds,
                keep_frames=args.keep_frames,
                frames_dir=args.frames_dir,
                memory_fade_time_s=args.memory_fade_seconds,
                min_event_interval_s=args.min_event_interval_ms / 1000.0,
                edge_window_s=args.edge_window_ms / 1000.0,
                face_opacity=args.face_opacity,
                msaa_samples=args.msaa_samples,
                output_width=args.output_width,
                output_height=args.output_height,
            )
        )

    if args.command == "preview":
        cfg = _load_effective_config(args.config)
        pcfg = cfg.get("preview", {})
        args.memory_fade_seconds = float(
            _resolve_value(raw_argv, "--memory-fade-seconds", args.memory_fade_seconds, pcfg, "memory_fade_seconds")
        )
        args.min_event_interval_ms = float(
            _resolve_value(raw_argv, "--min-event-interval-ms", args.min_event_interval_ms, pcfg, "min_event_interval_ms")
        )
        args.edge_window_ms = float(_resolve_value(raw_argv, "--edge-window-ms", args.edge_window_ms, pcfg, "edge_window_ms"))
        args.speed = float(_resolve_value(raw_argv, "--speed", args.speed, pcfg, "speed"))
        args.audio_sample_rate = int(
            _resolve_value(raw_argv, "--audio-sample-rate", args.audio_sample_rate, pcfg, "audio_sample_rate")
        )
        args.no_audio = bool(_resolve_value(raw_argv, "--no-audio", args.no_audio, pcfg, "no_audio"))
        args.face_opacity = float(_resolve_value(raw_argv, "--face-opacity", args.face_opacity, pcfg, "face_opacity"))
        args.msaa_samples = int(_resolve_value(raw_argv, "--msaa-samples", args.msaa_samples, pcfg, "msaa_samples"))
        _validate_preview_args(parser, args)
        if args.save_config:
            saved = save_config_file(
                args.save_config,
                {
                    "preview": {
                        "memory_fade_seconds": args.memory_fade_seconds,
                        "min_event_interval_ms": args.min_event_interval_ms,
                        "edge_window_ms": args.edge_window_ms,
                        "speed": args.speed,
                        "audio_sample_rate": args.audio_sample_rate,
                        "no_audio": args.no_audio,
                        "face_opacity": args.face_opacity,
                        "msaa_samples": args.msaa_samples,
                    }
                },
            )
            print(f"Saved config to {saved}")
        launch_timeline_preview(
            midi_path=args.midi,
            notes_json_path=args.notes_json,
            memory_fade_time_s=args.memory_fade_seconds,
            min_event_interval_s=args.min_event_interval_ms / 1000.0,
            edge_window_s=args.edge_window_ms / 1000.0,
            speed=args.speed,
            no_audio=args.no_audio,
            audio_sample_rate=args.audio_sample_rate,
            face_opacity=args.face_opacity,
            msaa_samples=args.msaa_samples,
            save_config_path=args.save_config,
        )
        return

    if args.command == "play":
        cfg = _load_effective_config(args.config)
        pcfg = cfg.get("play", {})
        args.sample_rate = int(_resolve_value(raw_argv, "--sample-rate", args.sample_rate, pcfg, "sample_rate"))
        args.tail_seconds = float(_resolve_value(raw_argv, "--tail-seconds", args.tail_seconds, pcfg, "tail_seconds"))
        args.duration_seconds = _resolve_value(
            raw_argv,
            "--duration-seconds",
            args.duration_seconds,
            pcfg,
            "duration_seconds",
        )
        args.no_playback = bool(_resolve_value(raw_argv, "--no-playback", args.no_playback, pcfg, "no_playback"))
        _validate_play_args(parser, args)
        if args.save_config:
            saved = save_config_file(
                args.save_config,
                {
                    "play": {
                        "sample_rate": args.sample_rate,
                        "tail_seconds": args.tail_seconds,
                        "duration_seconds": args.duration_seconds,
                        "no_playback": args.no_playback,
                    }
                },
            )
            print(f"Saved config to {saved}")
        raise SystemExit(
            play_timeline_audio(
                midi_path=args.midi,
                notes_json_path=args.notes_json,
                sample_rate=args.sample_rate,
                tail_seconds=args.tail_seconds,
                duration_seconds=args.duration_seconds,
                output_wav=args.output_wav,
                no_playback=args.no_playback,
            )
        )

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
