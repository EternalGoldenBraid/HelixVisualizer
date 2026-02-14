from json import dumps, loads
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

def _get_sd():
    import sounddevice as sd

    return sd


def list_devices() -> list[dict[str, Any]]:
    sd = _get_sd()
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        io_flag = []
        if dev["max_input_channels"] > 0:
            io_flag.append("IN")
        if dev["max_output_channels"] > 0:
            io_flag.append("OUT")
        print(f"[{idx}] {dev['name']} ({', '.join(io_flag)})")
    return devices


def prompt_for_input_device(devices: list[dict[str, Any]]) -> Tuple[int, dict[str, Any]]:
    while True:
        try:
            index = int(input("Select input device index: "))
            device = devices[index]
            if device["max_input_channels"] == 0:
                print("Selected device has no input channels.")
                continue
            return index, device
        except (ValueError, IndexError):
            print("Invalid index. Try again.")


def _find_valid_input_settings(
    device_index: int,
    channels_candidates: list[int],
    samplerate_candidates: list[int],
    checker: Callable[..., Any],
) -> Tuple[int, int] | None:
    for channels in channels_candidates:
        if channels <= 0:
            continue
        for samplerate in samplerate_candidates:
            if samplerate <= 0:
                continue
            try:
                checker(device=device_index, channels=channels, samplerate=samplerate)
                return channels, samplerate
            except Exception:
                continue
    return None


def _validated_or_none(
    config: Dict[str, int],
    devices: list[dict[str, Any]],
    checker: Callable[..., Any] | None = None,
) -> Dict[str, int] | None:
    idx = int(config.get("input_device_index", -1))
    if idx < 0 or idx >= len(devices):
        return None
    device = devices[idx]
    max_inputs = int(device.get("max_input_channels", 0))
    if max_inputs <= 0:
        return None

    requested_channels = max(1, int(config.get("input_channels", 1)))
    requested_channels = min(requested_channels, max_inputs)
    default_sr = int(device.get("default_samplerate", 44100))
    requested_sr = int(config.get("samplerate", default_sr))

    channels_candidates = list(range(requested_channels, 0, -1))
    if 1 not in channels_candidates:
        channels_candidates.append(1)

    samplerate_candidates = [requested_sr]
    if default_sr not in samplerate_candidates:
        samplerate_candidates.append(default_sr)

    if checker is None:
        checker = _get_sd().check_input_settings

    valid = _find_valid_input_settings(
        device_index=idx,
        channels_candidates=channels_candidates,
        samplerate_candidates=samplerate_candidates,
        checker=checker,
    )
    if valid is None:
        return None

    channels, samplerate = valid
    return {
        "input_device_index": idx,
        "input_channels": int(channels),
        "samplerate": int(samplerate),
    }


def select_input_device(config_file: Path) -> Dict[str, int]:
    sd = _get_sd()
    devices = sd.query_devices()

    if config_file.exists():
        print(f"Reading configuration from {config_file}")
        loaded = loads(config_file.read_text())
        validated = _validated_or_none(loaded, devices)
        if validated is not None:
            if validated != loaded:
                config_file.write_text(dumps(validated, indent=4))
                print("Adjusted audio config to valid input settings.")
            return validated
        print("Saved audio config is no longer valid. Please select an input device again.")

    if not config_file.parent.exists():
        config_file.parent.mkdir(parents=True, exist_ok=True)

    print("=== Available Audio Devices ===")
    list_devices()
    print("\n--- Select Input Device ---")
    input_index, input_dev = prompt_for_input_device(devices)

    default_sr = int(input_dev["default_samplerate"])
    valid = _find_valid_input_settings(
        device_index=input_index,
        channels_candidates=[1, min(2, int(input_dev["max_input_channels"]))],
        samplerate_candidates=[default_sr],
        checker=sd.check_input_settings,
    )
    if valid is None:
        raise RuntimeError("Selected input device has no valid mono/stereo stream configuration.")
    channels, samplerate = valid

    config = {
        "input_device_index": input_index,
        "input_channels": int(channels),
        "samplerate": int(samplerate),
    }
    config_file.write_text(dumps(config, indent=4))
    return config
