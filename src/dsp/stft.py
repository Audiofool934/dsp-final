import math
from typing import Literal

import numpy as np

from .fft import rfft

WindowType = Literal["hann", "hamming", "rect"]


def _get_window(window: WindowType, frame_length: int) -> np.ndarray:
    if window == "hann":
        n = np.arange(frame_length)
        return 0.5 - 0.5 * np.cos(2 * math.pi * n / frame_length)
    if window == "hamming":
        n = np.arange(frame_length)
        return 0.54 - 0.46 * np.cos(2 * math.pi * n / frame_length)
    if window == "rect":
        return np.ones(frame_length)
    raise ValueError(f"Unsupported window: {window}")


def frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    n_frames = 1 + max(0, (len(signal) - frame_length) // hop_length)
    frames = np.zeros((n_frames, frame_length), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_length
        frames[i] = signal[start : start + frame_length]
    return frames


def stft(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int,
    window: WindowType = "hann",
    n_fft: int | None = None,
) -> np.ndarray:
    """Short-time Fourier transform using custom FFT."""
    frames = frame_signal(signal, frame_length, hop_length)
    win = _get_window(window, frame_length)
    if n_fft is None:
        n_fft = frame_length
    out = np.empty((frames.shape[0], n_fft // 2 + 1), dtype=np.complex128)
    for i, frame in enumerate(frames):
        windowed = frame * win
        out[i] = rfft(windowed, n=n_fft)
    return out
