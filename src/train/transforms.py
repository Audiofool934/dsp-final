from __future__ import annotations

import numpy as np
import torch

from src.dsp.mfcc import MfccConfig, log_mel_spectrogram


class LogMelTransform:
    def __init__(self, cfg: MfccConfig):
        self.cfg = cfg

    def __call__(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        if sr != self.cfg.sample_rate:
            raise ValueError(f"Sample rate mismatch: {sr} != {self.cfg.sample_rate}")
        log_mel = log_mel_spectrogram(audio, self.cfg)
        # (frames, n_mels) -> (1, n_mels, frames)
        feat = torch.tensor(log_mel.T, dtype=torch.float32).unsqueeze(0)
        return feat
