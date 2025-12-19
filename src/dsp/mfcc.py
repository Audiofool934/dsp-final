import math
from dataclasses import dataclass

import numpy as np

from .stft import stft


@dataclass
class MfccConfig:
    sample_rate: int
    frame_length: int
    hop_length: int
    n_fft: int | None = None
    n_mels: int = 40
    n_mfcc: int = 13
    f_min: float = 0.0
    f_max: float | None = None
    pre_emphasis: float = 0.97
    window: str = "hann"


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> np.ndarray:
    if f_max is None:
        f_max = sample_rate / 2
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_freqs = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left = bin_freqs[m - 1]
        center = bin_freqs[m]
        right = bin_freqs[m + 1]
        if right <= left:
            continue
        for k in range(left, center):
            fbank[m - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            fbank[m - 1, k] = (right - k) / max(1, right - center)
    return fbank


def dct_type_2(x: np.ndarray, n_mfcc: int) -> np.ndarray:
    n = x.shape[-1]
    k = np.arange(n_mfcc).reshape(-1, 1)
    n_idx = np.arange(n).reshape(1, -1)
    basis = np.cos(math.pi / n * (n_idx + 0.5) * k)
    return 2.0 * np.dot(x, basis.T)


def log_mel_spectrogram(signal: np.ndarray, cfg: MfccConfig) -> np.ndarray:
    if cfg.pre_emphasis > 0:
        signal = np.append(signal[0], signal[1:] - cfg.pre_emphasis * signal[:-1])
    n_fft = cfg.n_fft or cfg.frame_length
    spec = stft(
        signal,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
        window=cfg.window,
        n_fft=n_fft,
    )
    power = np.abs(spec) ** 2
    fbank = mel_filterbank(cfg.n_mels, n_fft, cfg.sample_rate, cfg.f_min, cfg.f_max)
    mel_spec = np.dot(power, fbank.T)
    mel_spec = np.maximum(mel_spec, 1e-10)
    return np.log(mel_spec)


def mfcc(signal: np.ndarray, cfg: MfccConfig) -> np.ndarray:
    log_mel = log_mel_spectrogram(signal, cfg)
    cepstrum = dct_type_2(log_mel, cfg.n_mfcc)
    return cepstrum
