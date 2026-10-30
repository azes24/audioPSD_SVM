"""Modul ekstraksi fitur akustik - dipakai SAMA PERSIS oleh notebook (training)
dan app.py (inference) supaya tidak ada perbedaan pra-pemrosesan."""
import numpy as np
import librosa

SR = 16000                      # sample rate seragam
CHUNK_SEC = 3                   # durasi chunk (detik)
CHUNK_LEN = SR * CHUNK_SEC      # 48.000 sampel
TOP_DB = 25                     # ambang silence removal
MIN_CHUNK_SEC = 1.0             # sisa potongan terakhir < ini dibuang
USE_SILENCE_REMOVAL = True      # ubah ke False untuk membandingkan

FEATURE_NAMES = (
    ["rms_mean", "rms_std", "centroid_mean", "centroid_std", "zcr_mean", "zcr_std"]
    + [f"mfcc{i}_{s}" for i in range(1, 14) for s in ("mean", "std")]
)  # total 32 fitur


def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def remove_silence(y, top_db=None):
    """Buang bagian hening. Jika hasilnya < 0.5 detik, pakai sinyal asli."""
    if not USE_SILENCE_REMOVAL:
        return y
    intervals = librosa.effects.split(y, top_db=top_db or TOP_DB)
    if len(intervals) == 0:
        return y
    out = np.concatenate([y[s:e] for s, e in intervals])
    return out if len(out) >= SR * 0.5 else y


def make_chunks(y):
    """Potong jadi segmen 3 detik. Sisa >= 1 detik di-padding (wrap), sisanya dibuang."""
    chunks = []
    for s in range(0, len(y), CHUNK_LEN):
        c = y[s:s + CHUNK_LEN]
        if len(c) == CHUNK_LEN:
            chunks.append(c)
        elif len(c) >= MIN_CHUNK_SEC * SR:
            chunks.append(np.pad(c, (0, CHUNK_LEN - len(c)), mode="wrap"))
    return chunks


def extract_features(chunk):
    """1 chunk audio -> vektor 32 fitur (mean & std)."""
    rms = librosa.feature.rms(y=chunk)[0]
    cent = librosa.feature.spectral_centroid(y=chunk, sr=SR)[0]
    zcr = librosa.feature.zero_crossing_rate(chunk)[0]
    mfcc = librosa.feature.mfcc(y=chunk, sr=SR, n_mfcc=13)
    feats = [rms.mean(), rms.std(), cent.mean(), cent.std(), zcr.mean(), zcr.std()]
    for row in mfcc:
        feats += [row.mean(), row.std()]
    return np.array(feats, dtype=np.float32)


def file_to_features(path):
    """File audio -> matriks (n_chunk, 32). Dipakai saat membangun dataset."""
    y = remove_silence(load_audio(path))
    feats = [extract_features(c) for c in make_chunks(y)]
    return np.array(feats) if feats else np.empty((0, len(FEATURE_NAMES)))


def signal_to_features(y):
    """Sinyal live (16 kHz, ~3 detik) -> vektor fitur (1, 32). Dipakai saat inference."""
    y = remove_silence(y)
    return extract_features(y).reshape(1, -1)
