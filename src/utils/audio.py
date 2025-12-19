from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - fallback
    sf = None

try:
    from scipy.signal import resample_poly
except ImportError:  # pragma: no cover - fallback
    resample_poly = None


def load_audio(path: str | Path, target_sr: int | None = None) -> Tuple[np.ndarray, int]:
    path = Path(path)
    if sf is None:
        raise ImportError("soundfile is required for audio loading")
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_sr is not None and sr != target_sr:
        if resample_poly is None:
            raise ImportError("scipy is required for resampling")
        audio = resample_poly(audio, target_sr, sr)
        sr = target_sr
    return audio.astype(np.float32), sr


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio
