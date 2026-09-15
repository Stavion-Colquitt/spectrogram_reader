"""Interactive spectrogram reader. Run: python -m streamlit run app.py"""

import io
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.signal import butter, sosfiltfilt

from core import BG, FG, GRID, decimate, load_audio

st.set_page_config(page_title="Spectrogram Reader", layout="wide")
st.title("Spectrogram Reader")

up = st.file_uploader("Audio file", type=["wav", "mp3", "flac", "ogg", "m4a", "aiff"])


def analyze(y, sr, n_fft, hop, scale, n_mels, n_mfcc):
    """Return (matrix, y_axis, is_db)."""
    if scale == "Linear":
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
        return librosa.amplitude_to_db(S, ref=np.max), "log", True

    M = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    )
    M_db = librosa.power_to_db(M, ref=np.max)
    if scale == "Mel":
        return M_db, "mel", True
    return librosa.feature.mfcc(S=M_db, n_mfcc=n_mfcc), None, False


if not up:
    st.info("Upload a file to begin.")
    st.stop()

raw = up.getvalue()
y_full, sr = load_audio(raw, up.name, len(raw))
dur = len(y_full) / sr

with st.sidebar:
    st.header("Segment")
    lo, hi = st.slider("Seconds", 0.0, round(dur, 1), (0.0, round(dur, 1)), 0.5)

    st.header("STFT")
    n_fft = st.select_slider("Window (n_fft)", [256, 512, 1024, 2048, 4096, 8192], 2048)
    overlap = st.select_slider("Overlap", ["75%", "87.5%", "93.75%"], "75%")
    scale = st.radio("Scale", ["Linear", "Mel", "MFCC"], horizontal=True)
    n_mels = st.slider("Mel bands", 32, 256, 128, 32, disabled=scale == "Linear")
    n_mfcc = st.slider("MFCC count", 8, 40, 20, 2, disabled=scale != "MFCC")

    st.header("Preprocess")
    hpf = st.checkbox("High-pass filter")
    cutoff = st.slider("Cutoff (Hz)", 20, 500, 80, 10, disabled=not hpf)
    order = st.select_slider("Slope", [2, 4, 6, 8], 4, disabled=not hpf)

    st.header("Display")
    show_wave = st.checkbox("Waveform", value=True)
    show_rms = st.checkbox("Energy curve", value=False)
    thresh = st.slider("Silence threshold (dB)", -80, -10, -40, 2, disabled=not show_rms)
    db = st.slider("Dynamic range (dB)", 40, 120, 80, 10)
    cmap = st.selectbox("Colormap", ["turbo", "magma", "inferno", "plasma",
                                     "viridis", "cividis", "CMRmap", "gray_r"])

    st.header("Probe")
    probe = st.checkbox("Spectrum at a point in time")
    if hi - lo >= 0.2:
        p_t = st.slider("Probe time (s)", float(lo), float(hi),
                        float(lo + (hi - lo) / 2), 0.05, disabled=not probe)
    else:
        p_t = float(lo)
        if probe:
            st.caption("Segment too short to probe.")

hop = n_fft // {"75%": 4, "87.5%": 8, "93.75%": 16}[overlap]
y = y_full[int(lo * sr):int(hi * sr)]

if len(y) < n_fft:
    st.warning("Segment is shorter than the window. Widen it or shrink n_fft.")
    st.stop()

if hpf:
    sos = butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    y = sosfiltfilt(sos, y).astype(np.float32)

st.audio(raw)
D, y_axis, is_db = analyze(y, sr, n_fft, hop, scale, n_mels, n_mfcc)
Dp, stride = decimate(D)
t_end = len(y) / sr

rms_db = None
if show_rms:
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

rows = [r for r, on in [("wave", show_wave), ("spec", True), ("rms", show_rms)] if on]
heights = {"wave": 1, "spec": 4, "rms": 1.2}
fig, axs = plt.subplots(
    len(rows), 1, figsize=(14, 1.6 * sum(heights[r] for r in rows)),
    facecolor=BG, sharex=True, squeeze=False,
    gridspec_kw={"height_ratios": [heights[r] for r in rows], "hspace": 0.08},
)
axs = {name: axs[i][0] for i, name in enumerate(rows)}


def dress(ax, ylabel=None):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    if ylabel:
        ax.set_ylabel(ylabel, color=FG, fontsize=9)


if show_wave:
    ax = axs["wave"]
    ax.plot(np.arange(len(y)) / sr, y, color="#3DD68C", linewidth=0.4)
    ax.set_xlim(0, t_end)
    dress(ax, "amp")

ax = axs["spec"]
lim = {"vmin": -db, "vmax": 0} if is_db else {}
img = librosa.display.specshow(Dp, sr=sr, hop_length=hop * stride, x_axis="time",
                               y_axis=y_axis, ax=ax, cmap=cmap, **lim)
dress(ax)
if not is_db:
    ax.set_ylabel("MFCC coefficient", color=FG, fontsize=9)

if show_rms:
    ax = axs["rms"]
    t_rms = np.arange(len(rms_db)) * hop / sr
    ax.plot(t_rms, rms_db, color="#3DD68C", linewidth=1.0)
    ax.axhline(thresh, color="#FF7A5C", linewidth=0.9, linestyle="--")
    ax.fill_between(t_rms, -90, rms_db, where=rms_db >= thresh,
                    color="#3DD68C", alpha=0.18)
    ax.set_ylim(-90, 2)
    ax.set_xlim(0, t_end)
    dress(ax, "RMS dB")

if probe:
    for ax in axs.values():
        ax.axvline(p_t - lo, color="#FFFFFF", linewidth=0.9, alpha=0.7)

cb = fig.colorbar(img, ax=list(axs.values()),
                  format="%+2.0f dB" if is_db else "%+.0f", pad=0.01)
cb.ax.tick_params(colors=FG, labelsize=8)
cb.outline.set_edgecolor(GRID)

hp = f"   HPF {cutoff} Hz" if hpf else ""
axs[rows[0]].set_title(
    f"{up.name}   {lo:.1f}-{hi:.1f}s   {scale}   n_fft={n_fft}  hop={hop}{hp}",
    color=FG, fontsize=11, pad=10)
st.pyplot(fig)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Freq resolution", f"{sr / n_fft:.1f} Hz/bin")
c2.metric("Time resolution", f"{hop / sr * 1000:.1f} ms/frame")
c3.metric("Full matrix", f"{D.shape[0]} x {D.shape[1]}")
if show_rms:
    c4.metric("Above threshold", f"{100 * (rms_db >= thresh).mean():.1f}%")
else:
    c4.metric("Drawn", f"1 of every {stride}" if stride > 1 else "all frames")

if probe:
    frame = int((p_t - lo) * sr / hop)
    frame = max(0, min(frame, D.shape[1] - 1))
    col = D[:, frame]

    f2, a2 = plt.subplots(figsize=(14, 3), facecolor=BG)
    if scale == "Linear":
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        a2.semilogx(freqs[1:], col[1:], color="#3DD68C", linewidth=0.9)
        a2.set_xlim(20, sr / 2)
        a2.set_xlabel("Hz")
        peak = freqs[1:][np.argmax(col[1:])]
        a2.axvline(peak, color="#FF7A5C", linewidth=0.9, linestyle="--")
    elif scale == "Mel":
        freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=sr / 2)
        a2.semilogx(freqs, col, color="#3DD68C", linewidth=0.9)
        a2.set_xlabel("Hz (mel bands)")
        peak = freqs[np.argmax(col)]
        a2.axvline(peak, color="#FF7A5C", linewidth=0.9, linestyle="--")
    else:
        a2.bar(np.arange(len(col)), col, color="#3DD68C")
        a2.set_xlabel("MFCC coefficient")
        peak = None
    dress(a2, "dB" if is_db else "value")
    a2.set_title(f"Spectrum at {p_t:.2f}s" + (f"   peak {peak:.0f} Hz" if peak else ""),
                 color=FG, fontsize=10, pad=8)
    f2.tight_layout()
    st.pyplot(f2)
    if peak:
        st.caption(f"Loudest bin at {p_t:.2f}s is {peak:.1f} Hz "
                   f"({librosa.hz_to_note(peak)}) at {col.max():.1f} dB.")
    plt.close(f2)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, facecolor=BG, bbox_inches="tight")

npy = io.BytesIO()
np.save(npy, D.astype(np.float32))
stem = Path(up.name).stem
tag = f"{scale.lower()}_n{n_fft}_h{hop}"

d1, d2 = st.columns(2)
d1.download_button("Download PNG", buf.getvalue(),
                   f"{stem}_{tag}.png", "image/png", width="stretch")
d2.download_button(f"Download .npy  ({D.shape[0]}x{D.shape[1]})",
                   npy.getvalue(), f"{stem}_{tag}.npy",
                   "application/octet-stream", width="stretch")
st.caption(
    f"The .npy holds the full-resolution matrix for {lo:.1f}-{hi:.1f}s"
    f"{', high-passed' if hpf else ''}, not the decimated display copy."
)
plt.close(fig)
