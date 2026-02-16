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

You can also load/save parameter profiles:

```bash
uv run helix-viz ui --config configs/studio.json
uv run helix-viz ui --save-config configs/studio.json
```

Inside the UI, use the `Helix Controls` window to tweak parameters live (slider + numeric input, with highlighted explanatory tooltips) and click `Save Config...`.

The same controls window includes `Start Recording` / `Stop Recording` for manual guitar sessions:

- captures live helix visuals + microphone input
- writes outputs under `outputs/live_recordings/`
- saves final MP4 as `outputs/live_recordings/session_<timestamp>.mp4`

## Optional: Timeline Preview and Video Export

Besides live microphone mode, you can also feed known note timelines (MIDI or JSON note spans):

- `preview`: play timeline in the realtime Qt/OpenGL UI (no microphone required)
- `render-gl`: export video through the OpenGL visualizer (best visual match to UI)
- `render`: deterministic CPU renderer for testing and CI
- `play`: hear MIDI/JSON timelines with a minimal built-in synth

Example:

```bash
uv run helix-viz preview --midi assets/simple_scale.mid
uv run helix-viz render-gl --midi assets/simple_scale.mid --output outputs/demo_gl.mp4
uv run helix-viz play --midi assets/simple_scale.mid
```

When `preview` is run with `--midi`, synthesized timeline audio is played by default.
For multi-track/channel MIDI, `preview` and `render-gl` color memory activations per channel.
`preview` includes a matching dark `Timeline Controls` window and saves shared visual tuning for `preview` and `render-gl`.

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
