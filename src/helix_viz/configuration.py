from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "audio_processing": {
        "min_rms_threshold": 0.008,
        "min_peak_prominence_ratio": 10.0,
        "frequency_smoothing_alpha": 0.28,
    },
    "ui": {
        "memory_fade_seconds": 2.6,
        "min_event_interval_ms": 80.0,
        "edge_window_ms": 220.0,
        "face_opacity": 1.0,
        "camera_follow_note": False,
        "msaa_samples": 4,
    },
    "preview": {
        "memory_fade_seconds": 2.6,
        "min_event_interval_ms": 80.0,
        "edge_window_ms": 220.0,
        "face_opacity": 1.0,
        "speed": 1.0,
        "audio_sample_rate": 44100,
        "no_audio": False,
        "msaa_samples": 4,
    },
    "render": {
        "fps": 30,
        "width": 960,
        "height": 540,
        "tail_seconds": 1.0,
        "duration_seconds": None,
        "memory_fade_seconds": 2.6,
        "edge_window_ms": 220.0,
        "event_interval_ms": 80.0,
        "supersample_scale": 2,
    },
    "render_gl": {
        "fps": 30,
        "width": 960,
        "height": 540,
        "tail_seconds": 1.0,
        "duration_seconds": None,
        "memory_fade_seconds": 2.6,
        "min_event_interval_ms": 80.0,
        "edge_window_ms": 220.0,
        "face_opacity": 1.0,
        "msaa_samples": 4,
        "output_width": None,
        "output_height": None,
    },
    "play": {
        "sample_rate": 44100,
        "tail_seconds": 0.5,
        "duration_seconds": None,
        "no_playback": False,
    },
}


def get_default_config() -> dict[str, dict[str, Any]]:
    return deepcopy(DEFAULT_CONFIG)


def _deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_config_file(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a JSON object.")
    return _deep_merge_dict(get_default_config(), payload)


def save_config_file(path: str | Path, config: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
