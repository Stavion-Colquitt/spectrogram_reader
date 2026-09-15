"""Generate the demo audio and every figure used in the README.

Run: python make_figures.py
Everything here is synthetic, so the repo ships no real recordings.
"""

import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

from core import BG, FG, GRID

SR = 44100
OUT = "docs"


def tone(f0, dur, sr=SR, vibrato=5.0, depth=0.004, partials=6):
    """A voice-ish tone: harmonic stack, vibrato, soft attack and release."""
    t = np.arange(int(dur * sr)) / sr
    phase = 2 * np.pi * f0 * (t + depth * np.sin(2 * np.pi * vibrato * t))
    y = sum(np.sin(k * phase) / (k ** 1.6) for k in range(1, partials + 1))
    env = np.minimum(1.0, t / 0.04) * np.exp(-t / (dur * 0.9))
    return (y * env).astype(np.float32)


def consonant(dur=0.05, sr=SR):
    """A short noise burst standing in for a transient."""
    n = np.random.default_rng(0).normal(0, 1, int(dur * sr))
    sos = butter(2, 2500, btype="highpass", fs=sr, output="sos")
    env = np.exp(-np.linspace(0, 12, len(n)))
    return (sosfilt(sos, n) * env * 0.5).astype(np.float32)


def build_demo():
    """A short phrase, a silent gap, then a second phrase."""
    parts = []
    for f0, d in [(220.0, 0.9), (261.63, 0.7), (329.63, 1.1)]:
        parts += [consonant(), tone(f0, d)]
    parts.append(np.zeros(int(1.4 * SR), dtype=np.float32))
    for f0, d in [(293.66, 0.8), (349.23, 1.2)]:
        parts += [consonant(), tone(f0, d)]

    y = np.concatenate(parts)
    y += np.random.default_rng(1).normal(0, 3e-4, len(y)).astype(np.float32)  # room floor
    return (y / np.abs(y).max() * 0.85).astype(np.float32)


def process(y, sr=SR):
    """Stand-in for a mix move: air boost plus a sibilance cut."""
    hi = sosfilt(butter(2, 4000, btype="highpass", fs=sr, output="sos"), y)
    boosted = y + 1.1 * hi
    cut = sosfilt(butter(4, [5500, 8500], btype="bandstop", fs=sr, output="sos"), boosted)
    return (cut * 0.9).astype(np.float32)


def dress(ax, ylabel=None, title=None):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    if ylabel:
        ax.set_ylabel(ylabel, color=FG, fontsize=9)
    if title:
        ax.set_title(title, color=FG, fontsize=10, pad=8)


def spec_db(y, n_fft, hop, sr=SR, ref=None):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    return librosa.amplitude_to_db(S, ref=ref if ref is not None else np.max)


def fig_main(y, hop=512, n_fft=2048):
    """Waveform + spectrogram + energy curve, the default single-file view."""
    D = spec_db(y, n_fft, hop)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    thresh = -40

    fig, axs = plt.subplots(3, 1, figsize=(13, 8.5), facecolor=BG, sharex=True,
                            gridspec_kw={"height_ratios": [1, 4, 1.2], "hspace": 0.08})
    axs[0].plot(np.arange(len(y)) / SR, y, color="#3DD68C", linewidth=0.4)
    axs[0].set_xlim(0, len(y) / SR)
    dress(axs[0], "amp", "demo.wav   Linear   n_fft=2048  hop=512")

    img = librosa.display.specshow(D, sr=SR, hop_length=hop, x_axis="time",
                                   y_axis="log", ax=axs[1], cmap="turbo",
                                   vmin=-80, vmax=0)
    dress(axs[1])

    t = np.arange(len(rms_db)) * hop / SR
    axs[2].plot(t, rms_db, color="#3DD68C", linewidth=1.0)
    axs[2].axhline(thresh, color="#FF7A5C", linewidth=0.9, linestyle="--")
    axs[2].fill_between(t, -90, rms_db, where=rms_db >= thresh,
                        color="#3DD68C", alpha=0.18)
    axs[2].set_ylim(-90, 2)
    dress(axs[2], "RMS dB")

    cb = fig.colorbar(img, ax=list(axs), format="%+2.0f dB", pad=0.01)
    cb.ax.tick_params(colors=FG, labelsize=8)
    cb.outline.set_edgecolor(GRID)
    fig.savefig(f"{OUT}/main-view.png", dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    pct = 100 * (rms_db >= thresh).mean()
    print(f"main-view.png   {pct:.1f}% of frames above {thresh} dB")


def fig_tradeoff(y):
    """The same audio at a narrow and a wide window."""
    fig, axs = plt.subplots(2, 1, figsize=(13, 8), facecolor=BG, sharex=True,
                            gridspec_kw={"hspace": 0.18})
    for ax, (n_fft, hop) in zip(axs, [(512, 128), (8192, 2048)]):
        D = spec_db(y, n_fft, hop)
        img = librosa.display.specshow(D, sr=SR, hop_length=hop, x_axis="time",
                                       y_axis="log", ax=ax, cmap="turbo",
                                       vmin=-80, vmax=0)
        cb = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
        cb.ax.tick_params(colors=FG, labelsize=8)
        cb.outline.set_edgecolor(GRID)
        dress(ax, None, f"n_fft={n_fft}  hop={hop}    "
                        f"{SR / n_fft:.1f} Hz/bin, {hop / SR * 1000:.1f} ms/frame")
    fig.savefig(f"{OUT}/window-tradeoff.png", dpi=140, facecolor=BG,
                bbox_inches="tight")
    plt.close(fig)
    print("window-tradeoff.png")


def fig_compare(a, b, n_fft=2048, hop=512):
    """A, B and the signed difference on one shared dB reference."""
    Sa = np.abs(librosa.stft(a, n_fft=n_fft, hop_length=hop))
    Sb = np.abs(librosa.stft(b, n_fft=n_fft, hop_length=hop))
    ref = max(Sa.max(), Sb.max())
    Da = librosa.amplitude_to_db(Sa, ref=ref)
    Db = librosa.amplitude_to_db(Sb, ref=ref)
    diff = Db - Da

    fig, axs = plt.subplots(3, 1, figsize=(13, 11), facecolor=BG, sharex=True,
                            gridspec_kw={"hspace": 0.2})
    specs = [(Da, "turbo", -80, 0, "A - demo.wav"),
             (Db, "turbo", -80, 0, "B - demo_processed.wav"),
             (diff, "diffdark", -14, 14,
              "B minus A   (warm = B louder, cool = B quieter)")]
    for ax, (D, cm, lo, hi, title) in zip(axs, specs):
        img = librosa.display.specshow(D, sr=SR, hop_length=hop, x_axis="time",
                                       y_axis="log", ax=ax, cmap=cm,
                                       vmin=lo, vmax=hi)
        cb = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
        cb.ax.tick_params(colors=FG, labelsize=8)
        cb.outline.set_edgecolor(GRID)
        dress(ax, None, title)
    fig.savefig(f"{OUT}/compare-diff.png", dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"compare-diff.png   mean {diff.mean():+.2f} dB, "
          f"max {diff.max():+.1f}, min {diff.min():+.1f}")


def fig_probe(y, t_probe=1.2, n_fft=2048, hop=512):
    """The spectrum at one instant, with the peak marked."""
    D = spec_db(y, n_fft, hop)
    frame = min(int(t_probe * SR / hop), D.shape[1] - 1)
    col = D[:, frame]
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    peak = freqs[1:][np.argmax(col[1:])]

    fig, ax = plt.subplots(figsize=(13, 3.2), facecolor=BG)
    ax.semilogx(freqs[1:], col[1:], color="#3DD68C", linewidth=0.9)
    ax.axvline(peak, color="#FF7A5C", linewidth=0.9, linestyle="--")
    ax.set_xlim(20, SR / 2)
    ax.set_xlabel("Hz")
    note = librosa.hz_to_note(peak, unicode=False)
    dress(ax, "dB", f"Spectrum at {t_probe:.2f}s    peak {peak:.0f} Hz ({note})")
    fig.savefig(f"{OUT}/probe.png", dpi=140, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"probe.png   peak {peak:.1f} Hz ({note})")


if __name__ == "__main__":
    demo = build_demo()
    proc = process(demo)
    sf.write(f"{OUT}/demo.wav", demo, SR)
    sf.write(f"{OUT}/demo_processed.wav", proc, SR)
    print(f"demo.wav   {len(demo) / SR:.1f}s @ {SR} Hz")

    fig_main(demo)
    fig_tradeoff(demo)
    fig_compare(demo, proc)
    fig_probe(demo)
    print("done")
