from collections import deque
from typing import Any, Optional

import numpy as np


class AudioProcessor:
    def __init__(
        self,
        sr: int,
        input_device_index: int,
        input_channels: int = 1,
        io_blocksize: int = 2048,
        fft_size: int = 4096,
        number_top_k_frequencies: int = 3,
        min_rms_threshold: float = 0.008,
        min_peak_prominence_ratio: float = 10.0,
        frequency_smoothing_alpha: float = 0.28,
    ):
        self.sr = sr
        self.input_device_index = input_device_index
        self.input_channels = input_channels
        self.io_blocksize = io_blocksize
        self.fft_size = fft_size
        self.num_top_frequencies = number_top_k_frequencies
        self.min_rms_threshold = float(min_rms_threshold)
        self.min_peak_prominence_ratio = float(min_peak_prominence_ratio)
        self.frequency_smoothing_alpha = float(frequency_smoothing_alpha)
        self._smoothed_dominant_frequency: Optional[float] = None

        self.current_top_k_frequencies: list[Optional[float]] = [None] * self.num_top_frequencies
        self.raw_input_queue: deque[np.ndarray] = deque(maxlen=64)
        self.audio_buffer = np.zeros((self.fft_size, max(1, self.input_channels)), dtype=np.float32)

        import sounddevice as sd

        self._sd: Any = sd
        self.input_stream = sd.InputStream(
            device=self.input_device_index,
            channels=self.input_channels,
            samplerate=self.sr,
            callback=self.audio_input_callback,
            blocksize=self.io_blocksize,
            dtype="float32",
            latency="low",
        )

    def start(self) -> bool:
        self.input_stream.start()
        return True

    def stop(self) -> None:
        if self.input_stream is not None:
            self.input_stream.stop()
            self.input_stream.close()
            print("Input stream stopped.")

    def audio_input_callback(self, indata: np.ndarray, frames: int, time, status):
        if status:
            print(f"Input stream status: {status}")
        self.raw_input_queue.append(indata.copy())

    def process_pending_audio(self) -> None:
        while self.raw_input_queue:
            chunk = self.raw_input_queue.popleft()
            self.process_audio(chunk)

    def process_audio(self, indata: np.ndarray) -> None:
        frames = indata.shape[0]
        self.audio_buffer = np.roll(self.audio_buffer, -frames, axis=0)
        self.audio_buffer[-frames:] = indata

        mono = self.audio_buffer[:, 0].astype(np.float64)
        mono = mono - np.mean(mono)
        rms = self.compute_rms(mono)
        if rms < self.min_rms_threshold:
            self._set_no_pitch()
            return

        window = np.hanning(self.fft_size)
        spectrum = np.abs(np.fft.rfft(mono * window))
        freqs = np.fft.rfftfreq(self.fft_size, d=1 / self.sr)

        valid = (freqs >= 60.0) & (freqs <= 1500.0)
        if not np.any(valid):
            self._set_no_pitch()
            return

        valid_spectrum = spectrum[valid]
        valid_freqs = freqs[valid]
        if valid_spectrum.size < 3:
            self._set_no_pitch()
            return

        dominant_idx = int(np.argmax(valid_spectrum))
        dominant_mag = float(valid_spectrum[dominant_idx])
        noise_floor = float(np.median(valid_spectrum) + 1e-12)
        if dominant_mag / noise_floor < self.min_peak_prominence_ratio:
            self._set_no_pitch()
            return

        dominant_freq = self._interpolate_peak_frequency(valid_freqs, valid_spectrum, dominant_idx)
        smoothed = self._smooth_frequency(self._smoothed_dominant_frequency, dominant_freq, self.frequency_smoothing_alpha)
        self._smoothed_dominant_frequency = smoothed

        top_idxs = np.argsort(valid_spectrum)[-self.num_top_frequencies :]
        top_freqs = valid_freqs[top_idxs][::-1]
        if len(top_freqs) > 0:
            top_freqs[0] = smoothed

        for i in range(self.num_top_frequencies):
            self.current_top_k_frequencies[i] = float(top_freqs[i]) if i < len(top_freqs) else None

    def _set_no_pitch(self) -> None:
        self._smoothed_dominant_frequency = None
        for i in range(self.num_top_frequencies):
            self.current_top_k_frequencies[i] = None

    @staticmethod
    def compute_rms(samples: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(samples))))

    @staticmethod
    def _smooth_frequency(previous: Optional[float], current: float, alpha: float) -> float:
        if previous is None:
            return float(current)
        prev_log = np.log2(previous)
        curr_log = np.log2(current)
        return float(2 ** ((1.0 - alpha) * prev_log + alpha * curr_log))

    @staticmethod
    def _interpolate_peak_frequency(freqs: np.ndarray, mags: np.ndarray, idx: int) -> float:
        if idx <= 0 or idx >= len(mags) - 1:
            return float(freqs[idx])
        left = mags[idx - 1]
        center = mags[idx]
        right = mags[idx + 1]
        denom = left - 2.0 * center + right
        if abs(denom) < 1e-12:
            return float(freqs[idx])
        delta = 0.5 * (left - right) / denom
        delta = float(np.clip(delta, -1.0, 1.0))
        step = float(freqs[idx + 1] - freqs[idx])
        return float(freqs[idx] + delta * step)
