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


if __name__ == "__main__":
    unittest.main()
