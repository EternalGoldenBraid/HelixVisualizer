# CLI and Testing Workflows

This document covers the command-line endpoints that are useful for:

- verifying behavior during development
- generating reproducible fixtures/assets
- running timeline-based preview/render flows without live audio input

## Commands Overview

- `helix-viz ui`: realtime microphone-driven OpenGL UI
- `helix-viz preview`: realtime UI driven by MIDI/JSON timelines
- `helix-viz render-gl`: video export using OpenGL Qt visualizer
- `helix-viz render`: deterministic CPU renderer (CI-friendly)
- `helix-viz play`: synthesized audio playback from MIDI/JSON timelines
- `helix-viz probe`: print helix coordinates for selected frequencies

Most commands support:

- `--config <path.json>` to load parameters
- `--save-config <path.json>` to save resolved parameters

## Realtime Timeline Preview

Play MIDI in the same OpenGL UI path as live mode:

```bash
uv run helix-viz preview --midi assets/simple_scale.mid --speed 1.0
```

MIDI preview plays synthesized audio by default. Disable with:

```bash
uv run helix-viz preview --midi assets/simple_scale.mid --no-audio
```

Profile example:

```bash
uv run helix-viz preview --midi assets/simple_scale.mid --config configs/preview.json
```

Use JSON note spans instead:

```bash
uv run helix-viz preview --notes-json assets/example_notes.json
```

## OpenGL Video Render

Use this when you want exported video to match the UI look:

```bash
uv run helix-viz render-gl \
  --midi assets/simple_scale.mid \
  --output outputs/demo_gl.mp4 \
  --fps 30 \
  --width 1280 \
  --height 720
```

Important:

- Requires a working Qt/OpenGL environment.
- Requires `ffmpeg` in `PATH` for MP4 encoding.
- Use `--frames-dir` or `--keep-frames` to keep intermediate frame images.

## Deterministic CPU Video Render

Use this for stable regression outputs and headless CI:

```bash
uv run helix-viz render \
  --midi assets/simple_scale.mid \
  --output outputs/demo_cpu.mp4 \
  --fps 30 \
  --width 1280 \
  --height 720 \
  --supersample-scale 2
```

Inputs:

- `--midi <file.mid>`
- `--notes-json <file.json>`

Sample JSON format:

```json
[
  {"start_s": 0.0, "end_s": 0.5, "midi_note": 60, "velocity": 100},
  {"start_s": 0.3, "end_s": 0.9, "midi_note": 64, "velocity": 96}
]
```

## Probe Utility

Useful for checking frequency-to-helix mapping:

```bash
uv run helix-viz probe --freq 82.41 --freq 110 --freq 440
```

Output columns:

- `freq_hz`
- `x,y,z`
- `status` (`ok` or `out_of_range`)

## Timeline Audio Playback

Play MIDI as simple synthesized audio:

```bash
uv run helix-viz play --midi assets/simple_scale.mid
```

Generate WAV without playback:

```bash
uv run helix-viz play \
  --midi assets/simple_scale.mid \
  --output-wav outputs/simple_scale.wav \
  --no-playback
```

## Test Suite

Run all unit tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
