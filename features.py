"""Modul ekstraksi fitur akustik - dipakai SAMA PERSIS oleh notebook (training)
dan app.py (inference) supaya tidak ada perbedaan pra-pemrosesan."""
import numpy as np
import librosa

SR = 16000                      # sample rate seragam (16 kHz)
CHUNK_SEC = 5                   # durasi chunk analisis (detik)
CHUNK_LEN = SR * CHUNK_SEC      # 80.000 sampel per chunk
TOP_DB = 25                     # ambang batas pemangkasan hening
MIN_CHUNK_SEC = 1.0             # sisa potongan terakhir < 1 detik dibuang
USE_SILENCE_REMOVAL = True      # aktifkan pembersihan jeda sunyi

# Daftar 14 Fitur Terbaik Terpilih (Hasil Seleksi Fitur ANOVA F-Score)
# Memangkas lebih dari 55% dimensi awal (dari 32 ke 14) untuk efisiensi & akurasi optimal (~89%)
FEATURE_NAMES = (
    "rms_mean", "rms_std", "zcr_mean", "mfcc1_mean", "mfcc3_mean",
    "mfcc4_mean", "mfcc5_mean", "mfcc6_mean", "mfcc7_mean", "mfcc8_mean",
    "mfcc11_mean", "mfcc11_std", "mfcc12_std", "mfcc13_std"
)


def load_audio(path):
    """Memuat file audio menjadi format mono dengan sample rate seragam 16 kHz."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def remove_silence(y, top_db=None):
    """Membuang bagian hening. Jika audio hasil potong < 0.5 detik, gunakan sinyal asli."""
    if not USE_SILENCE_REMOVAL:
        return y
    intervals = librosa.effects.split(y, top_db=top_db or TOP_DB)
    if len(intervals) == 0:
        return y
    out = np.concatenate([y[s:e] for s, e in intervals])
    return out if len(out) >= SR * 0.5 else y


def make_chunks(y):
    """Memotong audio menjadi segmen 5 detik (80.000 sampel). Sisa >= 1 detik di-pad (wrap)."""
    chunks = []
    for s in range(0, len(y), CHUNK_LEN):
        c = y[s:s + CHUNK_LEN]
        if len(c) == CHUNK_LEN:
            chunks.append(c)
        elif len(c) >= MIN_CHUNK_SEC * SR:
            chunks.append(np.pad(c, (0, CHUNK_LEN - len(c)), mode="wrap"))
    return chunks


def extract_features(chunk):
    """Mengekstrak vektor 14 fitur akustik terpilih dari satu segmen audio 5 detik."""
    rms = librosa.feature.rms(y=chunk)[0]
    cent = librosa.feature.spectral_centroid(y=chunk, sr=SR)[0]
    zcr = librosa.feature.zero_crossing_rate(chunk)[0]
    mfcc = librosa.feature.mfcc(y=chunk, sr=SR, n_mfcc=13)

    raw = {
        "rms_mean": float(rms.mean()),
        "rms_std": float(rms.std()),
        "centroid_mean": float(cent.mean()),
        "centroid_std": float(cent.std()),
        "zcr_mean": float(zcr.mean()),
        "zcr_std": float(zcr.std()),
    }
    for i in range(1, 14):
        raw[f"mfcc{i}_mean"] = float(mfcc[i - 1].mean())
        raw[f"mfcc{i}_std"] = float(mfcc[i - 1].std())

    return np.array([raw[f] for f in FEATURE_NAMES], dtype=np.float32)


def file_to_features(path):
    """Mengekstrak seluruh chunk dari sebuah file rekaman audio menjadi matriks fitur."""
    y = remove_silence(load_audio(path))
    feats = [extract_features(c) for c in make_chunks(y)]
    return np.array(feats) if feats else np.empty((0, len(FEATURE_NAMES)))


def signal_to_features(y):
    """Mengekstrak fitur dari sinyal live mikrofon (16 kHz, 5 detik) saat inferensi web."""
    y = remove_silence(y)
    return extract_features(y).reshape(1, -1)
