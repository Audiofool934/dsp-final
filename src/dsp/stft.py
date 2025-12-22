import math
from functools import lru_cache
from typing import Literal

import numpy as np

from .fft import rfft

WindowType = Literal["hann", "hamming", "rect"]


@lru_cache(maxsize=32)
def _get_window(window: WindowType, frame_length: int) -> np.ndarray:
    """Cached window generation to avoid recomputation."""
    if frame_length <= 0:
        raise ValueError("frame_length must be positive")
    n = np.arange(frame_length)
    if window == "hann":
        return 0.5 - 0.5 * np.cos(2 * math.pi * n / frame_length)
    if window == "hamming":
        return 0.54 - 0.46 * np.cos(2 * math.pi * n / frame_length)
    if window == "rect":
        return np.ones(frame_length)
    raise ValueError(f"Unsupported window: {window}")


def frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Frame signal using stride tricks; still pure NumPy (no librosa)."""
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    if signal.ndim != 1:
        signal = np.asarray(signal).reshape(-1)
    n_frames = 1 + max(0, (len(signal) - frame_length) // hop_length)
    if n_frames <= 0:
        return np.empty((0, frame_length), dtype=np.float64)
    stride = signal.strides[0]
    shape = (n_frames, frame_length)
    strides = (hop_length * stride, stride)
    frames_view = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    return np.array(frames_view, dtype=np.float64, copy=True)


def stft(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int,
    window: WindowType = "hann",
    n_fft: int | None = None,
) -> np.ndarray:
    """Short-time Fourier transform using our own RFFT (no numpy.fft fast-path)."""
    frames = frame_signal(signal, frame_length, hop_length)
    win = _get_window(window, frame_length)
    if n_fft is None:
        n_fft = frame_length
    # Still loop over frames (requirement: own FFT), but keep it NumPy-heavy
    return np.vstack([rfft(frame * win, n=n_fft) for frame in frames]) if len(frames) else np.empty((0, n_fft // 2 + 1), dtype=np.complex128)


# Legacy per-frame loop is now folded into the list comprehension above.
