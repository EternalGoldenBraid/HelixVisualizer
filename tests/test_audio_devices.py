import unittest

from helix_viz.audio_devices import _find_valid_input_settings, _validated_or_none


class TestAudioDevices(unittest.TestCase):
    def test_find_valid_input_settings_picks_first_working_combo(self) -> None:
        def checker(*, device: int, channels: int, samplerate: int):
            if channels == 1 and samplerate == 44100:
                return
            raise RuntimeError("invalid")

        valid = _find_valid_input_settings(
            device_index=2,
            channels_candidates=[2, 1],
            samplerate_candidates=[48000, 44100],
            checker=checker,
        )
        self.assertEqual(valid, (1, 44100))

    def test_validated_or_none_clamps_channels(self) -> None:
        devices = [{"max_input_channels": 1, "default_samplerate": 44100}]
        config = {"input_device_index": 0, "input_channels": 8, "samplerate": 44100}
        valid = _validated_or_none(config, devices, checker=lambda **_: None)
        self.assertIsNotNone(valid)
        assert valid is not None
        self.assertEqual(valid["input_channels"], 1)
        self.assertEqual(valid["samplerate"], 44100)

    def test_validated_or_none_returns_none_for_invalid_device(self) -> None:
        devices = [{"max_input_channels": 1, "default_samplerate": 44100}]
        config = {"input_device_index": 3, "input_channels": 1, "samplerate": 44100}
        valid = _validated_or_none(config, devices, checker=lambda **_: None)
        self.assertIsNone(valid)


if __name__ == "__main__":
    unittest.main()
