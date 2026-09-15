"""A/B compare two audio files on a shared dB scale."""

import io
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from core import BG, decimate, load_audio, magnitude, style, to_db

st.set_page_config(page_title="Compare", layout="wide")
st.title("Compare")
st.caption("Both files share one dB reference, so level differences stay visible.")

c1, c2 = st.columns(2)
up_a = c1.file_uploader("A (reference)", type=["wav", "mp3", "flac", "ogg", "m4a"])
up_b = c2.file_uploader("B (edited)", type=["wav", "mp3", "flac", "ogg", "m4a"])

if not (up_a and up_b):
    st.info("Upload two files. Original in A, processed version in B.")
    st.stop()

raw_a, raw_b = up_a.getvalue(), up_b.getvalue()
y_a, sr = load_audio(raw_a, up_a.name, len(raw_a))
y_b, _ = load_audio(raw_b, up_b.name, len(raw_b), target_sr=sr)

n = min(len(y_a), len(y_b))
if abs(len(y_a) - len(y_b)) / max(len(y_a), len(y_b)) > 0.02:
    st.warning(f"Lengths differ; trimming both to {n / sr:.1f}s.")
y_a, y_b = y_a[:n], y_b[:n]
dur = n / sr

with st.sidebar:
    st.header("Segment")
    lo, hi = st.slider("Seconds", 0.0, round(dur, 1), (0.0, round(dur, 1)), 0.5)
    st.header("STFT")
    n_fft = st.select_slider("Window (n_fft)", [256, 512, 1024, 2048, 4096, 8192], 2048)
    overlap = st.select_slider("Overlap", ["75%", "87.5%", "93.75%"], "75%")
    mel = st.checkbox("Mel scale")
    n_mels = st.slider("Mel bands", 32, 256, 128, 32, disabled=not mel)
    st.header("Display")
    db = st.slider("Dynamic range (dB)", 40, 120, 80, 10)
    cmap = st.selectbox("Colormap", ["turbo", "magma", "inferno", "plasma",
                                     "viridis", "cividis", "gray_r"])
    st.header("Difference")
    show_diff = st.checkbox("Show B minus A", value=True)
    diff_span = st.slider("Diff scale (+/- dB)", 6, 60, 24, 6, disabled=not show_diff)
    diff_cmap = st.selectbox("Diff colormap", [
        "diffdark", "RdBu_r", "coolwarm", "bwr", "seismic",
        "PuOr_r", "BrBG_r", "PiYG_r", "turbo",
    ], disabled=not show_diff)

hop = n_fft // {"75%": 4, "87.5%": 8, "93.75%": 16}[overlap]
sl = slice(int(lo * sr), int(hi * sr))
seg_a, seg_b = y_a[sl], y_b[sl]

if len(seg_a) < n_fft:
    st.warning("Segment is shorter than the window.")
    st.stop()

S_a, y_axis = magnitude(seg_a, sr, n_fft, hop, mel, n_mels)
S_b, _ = magnitude(seg_b, sr, n_fft, hop, mel, n_mels)
ref = max(S_a.max(), S_b.max())
D_a, D_b = to_db(S_a, ref, mel), to_db(S_b, ref, mel)

rows = 3 if show_diff else 2
fig, axes = plt.subplots(rows, 1, figsize=(14, 4 * rows), facecolor=BG, sharex=True)

for ax, D, label in zip(axes, [D_a, D_b], [f"A - {up_a.name}", f"B - {up_b.name}"]):
    Dp, stride = decimate(D)
    img = librosa.display.specshow(Dp, sr=sr, hop_length=hop * stride,
                                   x_axis="time", y_axis=y_axis, ax=ax,
                                   cmap=cmap, vmin=-db, vmax=0)
    cb = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
    style(fig, ax, cb, label)

if show_diff:
    diff = D_b - D_a
    Dp, stride = decimate(diff)
    ax = axes[2]
    img = librosa.display.specshow(Dp, sr=sr, hop_length=hop * stride,
                                   x_axis="time", y_axis=y_axis, ax=ax,
                                   cmap=diff_cmap, vmin=-diff_span, vmax=diff_span)
    cb = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
    style(fig, ax, cb, "B minus A   (warm = B louder, cool = B quieter)")

fig.tight_layout()
st.pyplot(fig)


def panel_png(D, title, cm, vmin, vmax):
    """Render one panel on its own so each download stands alone."""
    f, a = plt.subplots(figsize=(14, 6), facecolor=BG)
    Dd, sd = decimate(D)
    im = librosa.display.specshow(Dd, sr=sr, hop_length=hop * sd, x_axis="time",
                                  y_axis=y_axis, ax=a, cmap=cm, vmin=vmin, vmax=vmax)
    c = f.colorbar(im, ax=a, format="%+2.0f dB", pad=0.01)
    style(f, a, c, title)
    f.tight_layout()
    b = io.BytesIO()
    f.savefig(b, format="png", dpi=150, facecolor=BG)
    plt.close(f)
    return b.getvalue()


stem_a, stem_b = Path(up_a.name).stem, Path(up_b.name).stem
tag = f"{'mel' if mel else 'stft'}_n{n_fft}_h{hop}_{lo:.0f}-{hi:.0f}s"

st.subheader("Downloads")
g1, g2, g3 = st.columns(3)
g1.download_button("A panel", panel_png(D_a, f"A - {up_a.name}", cmap, -db, 0),
                   f"{stem_a}_A_{tag}.png", "image/png", width="stretch")
g2.download_button("B panel", panel_png(D_b, f"B - {up_b.name}", cmap, -db, 0),
                   f"{stem_b}_B_{tag}.png", "image/png", width="stretch")

if show_diff:
    g3.download_button(
        "Difference panel",
        panel_png(D_b - D_a, "B minus A   (warm = B louder, cool = B quieter)",
                  diff_cmap, -diff_span, diff_span),
        f"{stem_b}_minus_{stem_a}_{tag}.png", "image/png", width="stretch")

e1, e2, e3 = st.columns(3)
stack = io.BytesIO()
fig.savefig(stack, format="png", dpi=150, facecolor=BG, bbox_inches="tight")
e1.download_button("All panels (stacked)", stack.getvalue(),
                   f"{stem_a}_vs_{stem_b}_{tag}.png", "image/png", width="stretch")

nb = io.BytesIO()
np.save(nb, (D_b - D_a).astype(np.float32))
e2.download_button(f"Difference .npy  ({D_a.shape[0]}x{D_a.shape[1]})", nb.getvalue(),
                   f"{stem_b}_minus_{stem_a}_{tag}.npy",
                   "application/octet-stream", width="stretch",
                   disabled=not show_diff)

both = io.BytesIO()
np.savez_compressed(both, A=D_a.astype(np.float32), B=D_b.astype(np.float32),
                    sr=sr, n_fft=n_fft, hop=hop)
e3.download_button("Both matrices .npz", both.getvalue(),
                   f"{stem_a}_vs_{stem_b}_{tag}.npz",
                   "application/octet-stream", width="stretch")
st.caption("Panels export at full resolution on a shared dB reference. "
           "The .npz holds A, B, sr, n_fft and hop.")

d = D_b - D_a
m1, m2, m3, m4 = st.columns(4)
m1.metric("Mean change", f"{d.mean():+.2f} dB")
m2.metric("Largest boost", f"{d.max():+.1f} dB")
m3.metric("Largest cut", f"{d.min():+.1f} dB")
m4.metric("Bins changed >1 dB", f"{100 * (np.abs(d) > 1).mean():.1f}%")

a1, a2 = st.columns(2)
a1.audio(raw_a)
a2.audio(raw_b)
