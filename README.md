# Helix Visualizer

Standalone guitar pitch helix visualizer extracted from the AudioViz project.

## Run

```bash
cd helix_visualizer
uv run python main.py
```

## Probe (No UI)

Print helix coordinates for specific frequencies without launching Qt:

```bash
uv run python -m helix_viz.main probe --freq 82.41 --freq 110 --freq 440
```

This returns CSV output with `freq_hz,x,y,z,status`, where status is `ok` when the note is within the configured guitar range.
The circle orientation is rotated so `D` is at the bottom for symmetric whole-tone spacing around that axis.
Recent played-note memory is rendered on the helix in full 3D (octave-aware), with fading nodes, edges, and filled surfaces.

## UI Tuning via CLI

Tune jitter filtering without code edits:

```bash
uv run python -m helix_viz.main ui \
  --min-rms-threshold 0.01 \
  --min-peak-prominence-ratio 12 \
  --frequency-smoothing-alpha 0.24 \
  --memory-fade-seconds 2.8 \
  --min-event-interval-ms 70 \
  --edge-window-ms 180
```

## Unit Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Notes

- Device selection is persisted to `outputs/audio_devices.json`.
- The app listens to live audio input and plots the dominant detected pitch on the helix.
- This folder is structured so it can be moved into a separate git repository directly.
