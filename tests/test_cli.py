import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from helix_viz.main import main


class TestCli(unittest.TestCase):
    def test_ui_defaults_are_forwarded(self) -> None:
        with patch("helix_viz.main.launch_ui") as launch_ui:
            main(["ui"])
            launch_ui.assert_called_once_with(
                min_rms_threshold=0.008,
                min_peak_prominence_ratio=10.0,
                frequency_smoothing_alpha=0.28,
                memory_fade_time_s=2.6,
                min_event_interval_s=0.08,
                edge_window_s=0.22,
                face_opacity=1.0,
                camera_follow_note=False,
                msaa_samples=4,
                save_config_path=None,
            )

    def test_ui_custom_filter_values_are_forwarded(self) -> None:
        with patch("helix_viz.main.launch_ui") as launch_ui:
            main(
                [
                    "ui",
                    "--min-rms-threshold",
                    "0.02",
                    "--min-peak-prominence-ratio",
                    "14",
                    "--frequency-smoothing-alpha",
                    "0.4",
                    "--memory-fade-seconds",
                    "3.2",
                    "--min-event-interval-ms",
                    "120",
                    "--edge-window-ms",
                    "180",
                    "--face-opacity",
                    "0.7",
                ]
            )
            launch_ui.assert_called_once_with(
                min_rms_threshold=0.02,
                min_peak_prominence_ratio=14.0,
                frequency_smoothing_alpha=0.4,
                memory_fade_time_s=3.2,
                min_event_interval_s=0.12,
                edge_window_s=0.18,
                face_opacity=0.7,
                camera_follow_note=False,
                msaa_samples=4,
                save_config_path=None,
            )

    def test_ui_rejects_invalid_alpha(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["ui", "--frequency-smoothing-alpha", "1.5"])
        self.assertEqual(ctx.exception.code, 2)

    def test_ui_rejects_invalid_memory_fade(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["ui", "--memory-fade-seconds", "0"])
        self.assertEqual(ctx.exception.code, 2)

    def test_ui_rejects_invalid_edge_window(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["ui", "--edge-window-ms", "-1"])
        self.assertEqual(ctx.exception.code, 2)

    def test_ui_loads_values_from_config_when_flags_not_set(self) -> None:
        cfg = {
            "audio_processing": {
                "min_rms_threshold": 0.015,
                "min_peak_prominence_ratio": 13.5,
                "frequency_smoothing_alpha": 0.19,
            },
            "ui": {
                "memory_fade_seconds": 4.0,
                "min_event_interval_ms": 140.0,
                "edge_window_ms": 260.0,
            },
        }
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "profile.json"
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
            with patch("helix_viz.main.launch_ui") as launch_ui:
                main(["ui", "--config", str(cfg_path)])
                launch_ui.assert_called_once_with(
                    min_rms_threshold=0.015,
                    min_peak_prominence_ratio=13.5,
                    frequency_smoothing_alpha=0.19,
                    memory_fade_time_s=4.0,
                    min_event_interval_s=0.14,
                    edge_window_s=0.26,
                    face_opacity=1.0,
                    camera_follow_note=False,
                    msaa_samples=4,
                    save_config_path=None,
                )

    def test_probe_outputs_csv_and_exit_code(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            with self.assertRaises(SystemExit) as ctx:
                main(["probe", "--freq", "82.41", "--freq", "20"])
        self.assertEqual(ctx.exception.code, 2)
        text = buffer.getvalue()
        self.assertIn("freq_hz,x,y,z,status", text)
        self.assertIn("82.410000", text)
        self.assertIn("out_of_range", text)

    def test_render_midi_args_are_forwarded(self) -> None:
        with patch("helix_viz.main.render_offline_video", return_value=0) as render_offline_video:
            with self.assertRaises(SystemExit) as ctx:
                main(
                    [
                        "render",
                        "--midi",
                        "fixtures/test.mid",
                        "--output",
                        "outputs/test.mp4",
                        "--fps",
                        "24",
                        "--width",
                        "640",
                        "--height",
                        "360",
                        "--tail-seconds",
                        "0.5",
                        "--duration-seconds",
                        "2.5",
                        "--frames-dir",
                        "outputs/frames",
                        "--keep-frames",
                    ]
                )
            self.assertEqual(ctx.exception.code, 0)
            render_offline_video.assert_called_once_with(
                midi_path="fixtures/test.mid",
                notes_json_path=None,
                output_mp4="outputs/test.mp4",
                fps=24,
                width=640,
                height=360,
                tail_seconds=0.5,
                duration_seconds=2.5,
                keep_frames=True,
                frames_dir="outputs/frames",
                memory_fade_time_s=2.6,
                edge_window_s=0.22,
                event_interval_s=0.08,
                supersample_scale=2,
            )

    def test_render_rejects_invalid_fps(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["render", "--midi", "in.mid", "--output", "out.mp4", "--fps", "0"])
        self.assertEqual(ctx.exception.code, 2)

    def test_render_gl_midi_args_are_forwarded(self) -> None:
        with patch("helix_viz.main.render_gl_video", return_value=0) as render_gl_video:
            with self.assertRaises(SystemExit) as ctx:
                main(
                    [
                        "render-gl",
                        "--midi",
                        "assets/simple_scale.mid",
                        "--output",
                        "outputs/test_gl.mp4",
                        "--fps",
                        "24",
                        "--width",
                        "800",
                        "--height",
                        "450",
                        "--tail-seconds",
                        "0.5",
                        "--duration-seconds",
                        "2.5",
                        "--frames-dir",
                        "outputs/gl_frames",
                        "--keep-frames",
                        "--memory-fade-seconds",
                        "3.1",
                        "--min-event-interval-ms",
                        "70",
                        "--edge-window-ms",
                        "180",
                        "--face-opacity",
                        "0.6",
                    ]
                )
            self.assertEqual(ctx.exception.code, 0)
            render_gl_video.assert_called_once_with(
                midi_path="assets/simple_scale.mid",
                notes_json_path=None,
                output_mp4="outputs/test_gl.mp4",
                fps=24,
                width=800,
                height=450,
                tail_seconds=0.5,
                duration_seconds=2.5,
                keep_frames=True,
                frames_dir="outputs/gl_frames",
                memory_fade_time_s=3.1,
                min_event_interval_s=0.07,
                edge_window_s=0.18,
                face_opacity=0.6,
                msaa_samples=4,
                output_width=None,
                output_height=None,
            )

    def test_render_gl_rejects_invalid_interval(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(
                [
                    "render-gl",
                    "--midi",
                    "in.mid",
                    "--output",
                    "out.mp4",
                    "--min-event-interval-ms",
                    "0",
                ]
            )
        self.assertEqual(ctx.exception.code, 2)

    def test_preview_midi_args_are_forwarded(self) -> None:
        with patch("helix_viz.main.launch_timeline_preview") as launch_timeline_preview:
            main(
                [
                    "preview",
                    "--midi",
                    "assets/simple_scale.mid",
                    "--memory-fade-seconds",
                    "3.0",
                    "--min-event-interval-ms",
                    "50",
                    "--edge-window-ms",
                    "300",
                    "--speed",
                    "1.5",
                    "--audio-sample-rate",
                    "48000",
                    "--face-opacity",
                    "0.65",
                ]
            )
            launch_timeline_preview.assert_called_once_with(
                midi_path="assets/simple_scale.mid",
                notes_json_path=None,
                memory_fade_time_s=3.0,
                min_event_interval_s=0.05,
                edge_window_s=0.3,
                speed=1.5,
                no_audio=False,
                audio_sample_rate=48000,
                face_opacity=0.65,
                msaa_samples=4,
                save_config_path=None,
            )

    def test_preview_rejects_invalid_speed(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["preview", "--midi", "in.mid", "--speed", "0"])
        self.assertEqual(ctx.exception.code, 2)

    def test_preview_rejects_invalid_audio_sample_rate(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["preview", "--midi", "in.mid", "--audio-sample-rate", "0"])
        self.assertEqual(ctx.exception.code, 2)

    def test_play_midi_args_are_forwarded(self) -> None:
        with patch("helix_viz.main.play_timeline_audio", return_value=0) as play_timeline_audio:
            with self.assertRaises(SystemExit) as ctx:
                main(
                    [
                        "play",
                        "--midi",
                        "assets/simple_scale.mid",
                        "--sample-rate",
                        "48000",
                        "--tail-seconds",
                        "0.8",
                        "--duration-seconds",
                        "2.1",
                        "--output-wav",
                        "outputs/simple_scale.wav",
                        "--no-playback",
                    ]
                )
            self.assertEqual(ctx.exception.code, 0)
            play_timeline_audio.assert_called_once_with(
                midi_path="assets/simple_scale.mid",
                notes_json_path=None,
                sample_rate=48000,
                tail_seconds=0.8,
                duration_seconds=2.1,
                output_wav="outputs/simple_scale.wav",
                no_playback=True,
            )

    def test_play_rejects_invalid_sample_rate(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["play", "--midi", "in.mid", "--sample-rate", "0"])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
