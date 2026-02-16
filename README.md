# Helix Visualizer

Helix Visualizer is a realtime guitar pitch visualizer.  
Its primary use is to listen to your guitar input and display notes in 3D on a helix.

## Quick Start (User-Facing)

From the project root:

```bash
uv run helix-viz ui
```

What happens on first run:

- You select an audio input device.
- The selection is saved to `outputs/audio_devices.json`.
- The UI starts and visualizes dominant detected pitches from your guitar input.

## Realtime UI Tuning

If your environment is noisy or pitch feels jumpy, tune the UI filters:

```bash
uv run helix-viz ui \
  --min-rms-threshold 0.01 \
  --min-peak-prominence-ratio 12 \
  --frequency-smoothing-alpha 0.24 \
  --memory-fade-seconds 2.8 \
  --min-event-interval-ms 70 \
  --edge-window-ms 180
```

## Optional: Timeline Preview and Video Export

Besides live microphone mode, you can also feed known note timelines (MIDI or JSON note spans):

- `preview`: play timeline in the realtime Qt/OpenGL UI (no microphone required)
- `render-gl`: export video through the OpenGL visualizer (best visual match to UI)
- `render`: deterministic CPU renderer for testing and CI

Example:

```bash
uv run helix-viz preview --midi assets/simple_scale.mid
uv run helix-viz render-gl --midi assets/simple_scale.mid --output outputs/demo_gl.mp4
```

`assets/simple_scale.mid` is included as a basic fixture.

## Developer and CLI Docs

Detailed CLI command reference, verification workflows, and test-oriented endpoints are documented in:

- `docs/cli-testing.md`

## Run Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## License

MIT. See `LICENSE`.

## Credits

Developed by Nicklas, with implementation support from OpenAI Codex (AI pair programmer).
