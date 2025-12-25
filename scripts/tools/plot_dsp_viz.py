import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
from pathlib import Path

# Add src to path
sys.path.append(".")

from src.dsp.stft import stft
from src.dsp.mfcc import log_mel_spectrogram, MfccConfig, dct_type_2

def plot_spectrogram(spec, title, output_path, y_label="Frequency"):
    plt.figure(figsize=(10, 4))
    # Flip Y axis for correct frequency display (low to high)
    plt.imshow(np.flipud(spec.T), aspect='auto', cmap='magma')
    plt.title(title)
    plt.xlabel('Time (Frames)')
    plt.ylabel(y_label)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")

def main():
    # Audio Path
    audio_path = "outputs/results/run_20251221_003516/errors/audio/ast_transfer/5-160614-E-48__gt_fireworks__pred_footsteps.wav"
    output_dir = Path("reports/latex/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Audio
    y, sr = librosa.load(audio_path, sr=44100)
    # Normalize
    y = y / np.max(np.abs(y))

    # STFT
    frame_length = 2048
    hop_length = 512
    # Custom STFT returns complex (N, F)
    # n_fft default is frame_length in our stft unless specified
    spec_complex = stft(y, frame_length, hop_length, window="hann", n_fft=frame_length)
    # Magnitude Spectrogram in dB
    spec_mag = np.abs(spec_complex)
    spec_db = librosa.amplitude_to_db(spec_mag, ref=np.max)

    plot_spectrogram(
        spec_db, 
        f"Custom STFT Spectrogram (Fireworks) | Frame {frame_length}", 
        output_dir / "viz_stft.png",
        y_label="Frequency Bin"
    )

    # Log Mel
    cfg = MfccConfig(
        sample_rate=sr,
        frame_length=frame_length,
        hop_length=hop_length,
        n_mels=40,
        n_mfcc=13,
        n_fft=frame_length
    )
    # Custom log_mel returns (N, n_mels)
    spec_mel = log_mel_spectrogram(y, cfg)
    
    plot_spectrogram(
        spec_mel, 
        f"Custom Log-Mel Spectrogram (Fireworks) | 40 Mels", 
        output_dir / "viz_logmel.png",
        y_label="Mel Filter Index"
    )

    # MFCC
    mfccs = dct_type_2(spec_mel, cfg.n_mfcc)
    
    plot_spectrogram(
        mfccs,
        f"Custom MFCC (Fireworks) | {cfg.n_mfcc} Coefficients",
        output_dir / "viz_mfcc.png",
        y_label="MFCC Coefficient Index"
    )

if __name__ == "__main__":
    main()
