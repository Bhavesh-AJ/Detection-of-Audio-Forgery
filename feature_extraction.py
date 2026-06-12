import os

os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(
        os.path.dirname(__file__),
        ".numba_cache"
    )
)

import numpy as np
import librosa
import soundfile as sf
from scipy.fftpack import dct
from scipy.signal import resample_poly


def load_audio(audio_path, target_sr=16000):

    try:

        y, sr = sf.read(
            audio_path,
            dtype="float32"
        )

        if y.ndim > 1:
            y = np.mean(
                y,
                axis=1
            )

        if sr != target_sr:
            gcd = np.gcd(sr, target_sr)

            y = resample_poly(
                y,
                target_sr // gcd,
                sr // gcd
            ).astype("float32")

            sr = target_sr

        return y, sr

    except Exception:

        return librosa.load(
            audio_path,
            sr=target_sr
        )


# -------------------------
# Chroma Features
# -------------------------

def extract_chroma(audio_path):

    y, sr = load_audio(
        audio_path,
        target_sr=16000
    )

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    chroma_mean = np.mean(
        chroma,
        axis=1
    )

    return chroma_mean


# -------------------------
# MFCC Features
# -------------------------

def _hz_to_mel(hz):

    return 2595 * np.log10(
        1 + hz / 700
    )


def _mel_to_hz(mel):

    return 700 * (
        10 ** (mel / 2595) - 1
    )


def _mel_filterbank(
        sr,
        n_fft,
        n_mels=128,
        fmin=0,
        fmax=None):

    if fmax is None:
        fmax = sr / 2

    mel_points = np.linspace(
        _hz_to_mel(fmin),
        _hz_to_mel(fmax),
        n_mels + 2
    )

    hz_points = _mel_to_hz(
        mel_points
    )

    bins = np.floor(
        (n_fft + 1) * hz_points / sr
    ).astype(int)

    filters = np.zeros(
        (n_mels, n_fft // 2 + 1),
        dtype=np.float32
    )

    for i in range(1, n_mels + 1):

        left = bins[i - 1]
        center = bins[i]
        right = bins[i + 1]

        if center > left:
            filters[i - 1, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)

        if right > center:
            filters[i - 1, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)

    return filters


def _fast_mfcc(
        y,
        sr,
        n_mfcc=40,
        n_fft=2048,
        hop_length=512,
        n_mels=128):

    if len(y) < n_fft:
        y = np.pad(
            y,
            (0, n_fft - len(y)),
            mode="constant"
        )

    frame_count = 1 + max(
        0,
        (len(y) - n_fft) // hop_length
    )

    frames = np.lib.stride_tricks.sliding_window_view(
        y,
        n_fft
    )[::hop_length][:frame_count]

    window = np.hanning(
        n_fft
    ).astype(np.float32)

    spectrum = np.fft.rfft(
        frames * window,
        n=n_fft,
        axis=1
    )

    power = (
        np.abs(spectrum) ** 2
    ) / n_fft

    mel_filters = _mel_filterbank(
        sr,
        n_fft,
        n_mels=n_mels
    )

    mel_power = np.dot(
        power,
        mel_filters.T
    )

    log_mel = np.log(
        np.maximum(mel_power, 1e-10)
    )

    mfcc = dct(
        log_mel,
        type=2,
        axis=1,
        norm="ortho"
    )[:, :n_mfcc]

    return mfcc.T.astype(
        np.float32
    )

def extract_mfcc(
        audio_path,
        max_pad_len=300):

    y, sr = load_audio(
        audio_path,
        target_sr=16000
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    if mfcc.shape[1] < max_pad_len:

        pad_width = max_pad_len - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )

    else:

        mfcc = mfcc[:, :max_pad_len]

    return mfcc


# -------------------------
# Deep Learning MFCC
# -------------------------

def extract_mfcc_for_dl(audio_path):

    mfcc = extract_mfcc(audio_path)

    # (40,300) -> (300,40)
    return mfcc.T


# -------------------------
# Testing
# -------------------------

if __name__ == "__main__":

    sample_file = r"Data\release_in_the_wild\0.wav"

    chroma = extract_chroma(sample_file)

    mfcc = extract_mfcc(sample_file)

    mfcc_dl = extract_mfcc_for_dl(sample_file)

    print("Chroma Shape :", chroma.shape)
    print("MFCC Shape :", mfcc.shape)
    print("MFCC DL Shape :", mfcc_dl.shape)
