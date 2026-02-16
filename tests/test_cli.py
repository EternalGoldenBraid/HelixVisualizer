import io
import unittest
from contextlib import redirect_stdout
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
                ]
            )
            launch_ui.assert_called_once_with(
                min_rms_threshold=0.02,
                min_peak_prominence_ratio=14.0,
                frequency_smoothing_alpha=0.4,
                memory_fade_time_s=3.2,
                min_event_interval_s=0.12,
                edge_window_s=0.18,
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
                ]
            )
            launch_timeline_preview.assert_called_once_with(
                midi_path="assets/simple_scale.mid",
                notes_json_path=None,
                memory_fade_time_s=3.0,
                min_event_interval_s=0.05,
                edge_window_s=0.3,
                speed=1.5,
            )

    def test_preview_rejects_invalid_speed(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["preview", "--midi", "in.mid", "--speed", "0"])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
