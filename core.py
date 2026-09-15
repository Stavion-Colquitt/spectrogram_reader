"""Shared helpers for the spectrogram tools."""

import tempfile
from pathlib import Path

import librosa
import numpy as np
import streamlit as st

BG = "#14181A"
FG = "#D9E0E4"
GRID = "#3A464C"
MAX_COLS = 2000


@st.cache_data(show_spinner="Decoding audio...")
def load_audio(raw, name, size, target_sr=None):
    """Decode once per file. Keyed on name+size so sliders never re-decode."""
    suffix = Path(name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(raw)
        tmp = f.name
    try:
        return librosa.load(tmp, sr=target_sr, mono=True)
    finally:
        Path(tmp).unlink(missing_ok=True)


def magnitude(y, sr, n_fft, hop, mel, n_mels):
    """Linear-magnitude spectrogram, before any dB conversion."""
    if mel:
        return librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
        ), "mel"
    return np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)), "log"


def to_db(S, ref, mel):
    """Convert to dB against a SHARED reference so two files stay comparable."""
    if mel:
        return librosa.power_to_db(S, ref=ref)
    return librosa.amplitude_to_db(S, ref=ref)


def decimate(D, cap=MAX_COLS):
    """Thin the time axis for display. Returns (matrix, stride)."""
    if D.shape[1] <= cap:
        return D, 1
    stride = int(np.ceil(D.shape[1] / cap))
    return D[:, ::stride], stride


def style(fig, ax, cb=None, title=None):
    """Apply the dark theme to a matplotlib axes."""
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    if cb is not None:
        cb.ax.tick_params(colors=FG, labelsize=8)
        cb.outline.set_edgecolor(GRID)
    if title:
        ax.set_title(title, color=FG, fontsize=11, pad=10)


# A diverging colormap with a DARK midpoint, so "no change" recedes into the
# background and real changes glow. Matplotlib's diverging maps all put white
# at zero, which floods a dark UI when most of a difference is near zero.
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import matplotlib  # noqa: E402

DIFF_DARK = LinearSegmentedColormap.from_list("diffdark", [
    "#7FD4FF", "#2E86AB", "#17394B", BG,
    "#4A2417", "#C9563C", "#FFB08A",
])
if "diffdark" not in matplotlib.colormaps:
    matplotlib.colormaps.register(DIFF_DARK, name="diffdark")
