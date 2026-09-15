"""Spectrogram reader - render an audio file as a spectrogram PNG."""

import argparse
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def build(path, n_fft, hop, mel, n_mels, sr):
    """Load audio and return (dB matrix, sample rate, y-axis type)."""
    y, sr = librosa.load(path, sr=sr, mono=True)

    if mel:
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
        )
        return librosa.power_to_db(S, ref=np.max), sr, "mel"

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    return librosa.amplitude_to_db(S, ref=np.max), sr, "log"


def render(D, sr, hop, y_axis, title, out, dynamic_range):
    """Draw the dB matrix and save it to disk."""
    fig, ax = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(
        D, sr=sr, hop_length=hop, x_axis="time", y_axis=y_axis,
        ax=ax, cmap="magma", vmin=-dynamic_range, vmax=0,
    )
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Render an audio file as a spectrogram.")
    p.add_argument("audio", type=Path, help="input audio file")
    p.add_argument("-o", "--out", type=Path, help="output PNG (default: <audio>.png)")
    p.add_argument("-n", "--n-fft", type=int, default=2048, help="FFT window size")
    p.add_argument("-H", "--hop", type=int, default=512, help="hop length in samples")
    p.add_argument("-m", "--mel", action="store_true", help="use a mel scale")
    p.add_argument("--n-mels", type=int, default=128, help="mel bands (with --mel)")
    p.add_argument("--sr", type=int, default=None, help="resample rate (default: native)")
    p.add_argument("--db", type=float, default=80.0, help="dynamic range shown, in dB")
    a = p.parse_args()

    if not a.audio.exists():
        p.error(f"no such file: {a.audio}")

    out = a.out or a.audio.with_suffix(".png")
    D, sr, y_axis = build(a.audio, a.n_fft, a.hop, a.mel, a.n_mels, a.sr)

    scale = f"mel ({a.n_mels} bands)" if a.mel else "linear"
    title = f"{a.audio.name}  |  n_fft={a.n_fft}  hop={a.hop}  {scale}"
    render(D, sr, a.hop, y_axis, title, out, a.db)

    secs = D.shape[1] * a.hop / sr
    print(f"{a.audio.name}: {sr} Hz, {secs:.1f}s")
    print(f"matrix: {D.shape[0]} bins x {D.shape[1]} frames")
    print(f"freq resolution: {sr / a.n_fft:.1f} Hz/bin")
    print(f"time resolution: {a.hop / sr * 1000:.1f} ms/frame")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
