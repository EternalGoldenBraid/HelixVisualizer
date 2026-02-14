import argparse
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from helix_viz.helix_math import D_BOTTOM_OFFSET_RADIANS, frequency_to_xyz

if TYPE_CHECKING:
    from helix_viz.guitar_profiles import GuitarProfile


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

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
