import argparse
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from helix_viz.helix_math import D_BOTTOM_OFFSET_RADIANS, frequency_to_xyz

if TYPE_CHECKING:
    from helix_viz.guitar_profiles import GuitarProfile
    from helix_viz.note_timeline import NoteSpan


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
) -> None:
    from PyQt5 import QtCore, QtWidgets

    from helix_viz.audio_devices import select_input_device
    from helix_viz.audio_processor import AudioProcessor
    from helix_viz.pitch_helix_visualizer import PitchHelixVisualizer

    config = select_input_device(config_file=Path("outputs/audio_devices.json"))
    sr = int(config["samplerate"])

    app = QtWidgets.QApplication([])

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
    )
    helix_window.setWindowTitle("Pitch Helix Visualizer")
    helix_window.resize(900, 700)
    helix_window.show()

    processor.start()

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app.aboutToQuit.connect(processor.stop)

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
        try:
            encode_frames_to_mp4(frames_dir=frames_out, output_mp4=output_mp4, fps=fps)
        except FileNotFoundError:
            print("ffmpeg was not found on PATH. Install ffmpeg to encode MP4 output.")
            return 3
        except subprocess.CalledProcessError:
            print("ffmpeg failed while encoding MP4 output.")
            return 3
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
) -> int:
    import math

    from PyQt5 import QtWidgets

    from helix_viz.pitch_helix_visualizer import PitchHelixVisualizer

    spans, base_duration_s = _load_note_spans_from_source(midi_path=midi_path, notes_json_path=notes_json_path)
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

    app = QtWidgets.QApplication([])
    processor = _TimelineProcessor()
    helix_window = PitchHelixVisualizer(
        processor=processor,  # type: ignore[arg-type]
        guitar_profile=_standard_guitar(),
        memory_fade_time_s=memory_fade_time_s,
        min_event_interval_s=min_event_interval_s,
        edge_window_s=edge_window_s,
    )
    # Pull camera back so full helix scale is visible in rendered video.
    helix_window.view.opts["distance"] = 68
    helix_window.setWindowTitle("Pitch Helix Visualizer - OpenGL Render")
    helix_window.resize(width, height)
    helix_window.show()
    app.processEvents()

    try:
        for i in range(num_frames):
            t_s = i / float(fps)
            processor.current_top_k_frequencies = _timeline_top_frequencies(spans=spans, t_s=t_s, top_k=3)
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
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(int(fps)),
            "-i",
            str(frame_path / "frame_%06d.png"),
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("ffmpeg was not found on PATH. Install ffmpeg to encode MP4 output.")
            return 3
        except subprocess.CalledProcessError:
            print("ffmpeg failed while encoding MP4 output.")
            return 3
        print(f"Rendered {num_frames} OpenGL frames to {frame_path}")
        print(f"Wrote video to {output_mp4}")
    finally:
        helix_window.close()
        app.quit()
        if temp_dir is not None and (not keep_frames):
            temp_dir.cleanup()

    return 0


def launch_timeline_preview(
    midi_path: str | None,
    notes_json_path: str | None,
    memory_fade_time_s: float = 2.6,
    min_event_interval_s: float = 0.08,
    edge_window_s: float = 0.22,
    speed: float = 1.0,
) -> None:
    import time

    from PyQt5 import QtCore, QtWidgets

    from helix_viz.pitch_helix_visualizer import PitchHelixVisualizer

    spans, duration_s = _load_note_spans_from_source(midi_path=midi_path, notes_json_path=notes_json_path)

    app = QtWidgets.QApplication([])
    processor = _TimelineProcessor()
    standard_guitar = _standard_guitar()

    helix_window = PitchHelixVisualizer(
        processor=processor,  # type: ignore[arg-type]
        guitar_profile=standard_guitar,
        memory_fade_time_s=memory_fade_time_s,
        min_event_interval_s=min_event_interval_s,
        edge_window_s=edge_window_s,
    )
    # Match render-gl framing for easier comparison between preview and output.
    helix_window.view.opts["distance"] = 68
    helix_window.setWindowTitle("Pitch Helix Visualizer - Timeline Preview")
    helix_window.resize(900, 700)
    helix_window.show()

    start_mono = time.monotonic()

    def tick() -> None:
        elapsed = (time.monotonic() - start_mono) * speed
        processor.current_top_k_frequencies = _timeline_top_frequencies(spans=spans, t_s=elapsed, top_k=3)
        if elapsed > (duration_s + memory_fade_time_s):
            app.quit()

    timer = QtCore.QTimer()
    timer.setInterval(max(10, int(min_event_interval_s * 1000)))
    timer.timeout.connect(tick)
    timer.start()
    tick()

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app.exec()


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


def _validate_preview_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.memory_fade_seconds <= 0:
        parser.error("--memory-fade-seconds must be > 0.")
    if args.min_event_interval_ms <= 0:
        parser.error("--min-event-interval-ms must be > 0.")
    if args.edge_window_ms < 0:
        parser.error("--edge-window-ms must be >= 0.")
    if args.speed <= 0:
        parser.error("--speed must be > 0.")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "ui"):
        if args.command == "ui":
            _validate_ui_args(parser, args)
            launch_ui(
                min_rms_threshold=args.min_rms_threshold,
                min_peak_prominence_ratio=args.min_peak_prominence_ratio,
                frequency_smoothing_alpha=args.frequency_smoothing_alpha,
                memory_fade_time_s=args.memory_fade_seconds,
                min_event_interval_s=args.min_event_interval_ms / 1000.0,
                edge_window_s=args.edge_window_ms / 1000.0,
            )
        else:
            launch_ui()
        return

    if args.command == "probe":
        raise SystemExit(probe_frequencies(freqs=args.freqs, radius=args.radius, pitch=args.pitch))

    if args.command == "render":
        _validate_render_args(parser, args)
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
        _validate_render_gl_args(parser, args)
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
            )
        )

    if args.command == "preview":
        _validate_preview_args(parser, args)
        launch_timeline_preview(
            midi_path=args.midi,
            notes_json_path=args.notes_json,
            memory_fade_time_s=args.memory_fade_seconds,
            min_event_interval_s=args.min_event_interval_ms / 1000.0,
            edge_window_s=args.edge_window_ms / 1000.0,
            speed=args.speed,
        )
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
