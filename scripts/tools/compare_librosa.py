from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.fftpack

from src.datasets.esc50 import Esc50Meta
from src.dsp.mfcc import MfccConfig, log_mel_spectrogram, mfcc
from src.dsp.stft import stft
from src.utils.audio import load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare custom DSP outputs with librosa")
    parser.add_argument("--data-root", type=str, default="data/ESC-50-master")
    parser.add_argument("--file", type=str, default=None, help="Specific ESC-50 filename to use")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-fft", type=int, default=None)
    parser.add_argument("--n-mels", type=int, default=40)
    parser.add_argument("--n-mfcc", type=int, default=13)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", type=str, default="outputs/validation/librosa_compare.json")
    return parser.parse_args()


def next_pow_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))


def pick_item(meta: Esc50Meta, filename: str | None):
    if filename:
        for item in meta.items:
            if item.filename == filename:
                return item
        raise ValueError(f"File not found in ESC-50 metadata: {filename}")
    return meta.by_folds([5])[0]


def main() -> None:
    args = parse_args()
    try:
        import librosa
    except ImportError as exc:
        raise ImportError("librosa is required for this comparison script") from exc

    meta = Esc50Meta(args.data_root)
    item = pick_item(meta, args.file)
    audio, sr = load_audio(item.path, target_sr=args.sample_rate)

    frame_length = args.frame_length
    hop_length = args.hop_length
    n_fft = args.n_fft or frame_length
    n_fft_eff = next_pow_two(n_fft)

    if args.max_frames:
        max_len = frame_length + hop_length * (args.max_frames - 1)
        audio = audio[:max_len]

    cfg = MfccConfig(
        sample_rate=args.sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        n_fft=n_fft_eff,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
    )

    # STFT comparison (no pre-emphasis)
    stft_custom = stft(audio, frame_length=frame_length, hop_length=hop_length, n_fft=n_fft_eff)
    stft_lib = librosa.stft(
        audio,
        n_fft=n_fft_eff,
        hop_length=hop_length,
        win_length=frame_length,
        window="hann",
        center=False,
    )
    stft_lib = stft_lib.T

    stft_mag_error = relative_error(np.abs(stft_custom), np.abs(stft_lib))
    stft_complex_error = relative_error(stft_custom, stft_lib)

    # MFCC comparison (match mel scale and log base)
    log_mel_custom = log_mel_spectrogram(audio, cfg)
    mfcc_custom = mfcc(audio, cfg)

    if cfg.pre_emphasis > 0:
        audio_pe = np.append(audio[0], audio[1:] - cfg.pre_emphasis * audio[:-1])
    else:
        audio_pe = audio

    mel_filter = librosa.filters.mel(
        sr=cfg.sample_rate,
        n_fft=n_fft_eff,
        n_mels=cfg.n_mels,
        fmin=cfg.f_min,
        fmax=cfg.f_max,
        htk=True,
        norm=None,
    )
    stft_lib_pe = librosa.stft(
        audio_pe,
        n_fft=n_fft_eff,
        hop_length=hop_length,
        win_length=frame_length,
        window="hann",
        center=False,
    )
    power = np.abs(stft_lib_pe) ** 2
    mel_spec = mel_filter @ power
    log_mel_lib = np.log(np.maximum(mel_spec, 1e-10)).T
    mfcc_lib = scipy.fftpack.dct(log_mel_lib, type=2, axis=1, norm=None)[:, : cfg.n_mfcc]

    log_mel_error = relative_error(log_mel_custom, log_mel_lib)
    mfcc_error = relative_error(mfcc_custom, mfcc_lib)

    report = {
        "file": item.filename,
        "sample_rate": args.sample_rate,
        "frame_length": frame_length,
        "hop_length": hop_length,
        "n_fft": n_fft_eff,
        "stft_complex_rel_error": stft_complex_error,
        "stft_mag_rel_error": stft_mag_error,
        "log_mel_rel_error": log_mel_error,
        "mfcc_rel_error": mfcc_error,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
