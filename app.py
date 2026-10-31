"""Smart Noise Alert System - Backend Flask.
Laboratorium Teknik Informatika UTM - PSD Kelompok 10
Jalankan: python app.py -> http://localhost:8010
"""
import csv
import datetime
import io
import math
import os
import sys
import tempfile
import warnings
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import scipy.signal
from flask import Flask, jsonify, render_template, request, send_file
from sklearn.exceptions import InconsistentVersionWarning

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Dynamic path resolution to safely run from current folder or container /app
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import features as ft


def resolve_file(filename):
    """Cari file dari BASE_DIR atau current working directory."""
    candidates = [
        BASE_DIR / filename,
        Path(filename)
    ]
    for c in candidates:
        if c.exists():
            return c
    return BASE_DIR / filename


# Load Model Bundle
model_path = resolve_file("model_noise.pkl")
bundle = joblib.load(model_path)
model = bundle["model"]
THRESHOLD = float(bundle.get("threshold", 0.6))
LOG_FILE = BASE_DIR / "log_kebisingan.csv"

# Inisialisasi Flask dengan template_folder yang aman
template_dir = resolve_file("templates")
if not template_dir.is_dir():
    template_dir = BASE_DIR / "templates"

app = Flask(__name__, template_folder=str(template_dir))
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


def resample_audio(y, orig_sr, target_sr):
    """Resample audio cepat menggunakan resample_poly."""
    if orig_sr == target_sr:
        return y
    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return scipy.signal.resample_poly(y, up, down).astype(np.float32)


def log_event(prob, noisy, db):
    new = not LOG_FILE.exists()
    try:
        with LOG_FILE.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["waktu", "probabilitas", "berisik", "level_dbfs"])
            w.writerow([
                datetime.datetime.now().isoformat(timespec="seconds"),
                round(prob, 4),
                int(noisy),
                round(db, 1)
            ])
    except Exception as e:
        app.logger.warning(f"Gagal mencatat log event: {e}")


@app.get("/")
def index():
    raw_name = bundle.get("model_name", "SVM")
    clean_name = raw_name.replace("(RBF)", "").replace("(rbf)", "").strip() or "SVM"
    accuracy = bundle.get("accuracy", "89%")
    return render_template(
        "index.html",
        model_name=clean_name,
        accuracy=accuracy,
        threshold=round(THRESHOLD, 3)
    )


@app.get("/alarm.mp3")
def get_alarm_sound():
    alarm_path = resolve_file("alarm.mp3")
    if alarm_path.exists():
        return send_file(alarm_path, mimetype="audio/mpeg")
    return jsonify(error="alarm.mp3 tidak ditemukan"), 404


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """Terima PCM float32 mentah (mono) dari browser, query ?sr=<sample rate browser>&th=<threshold>."""
    if request.method == "OPTIONS":
        return "", 204

    try:
        sr_param = request.args.get("sr", "48000")
        sr = int(float(sr_param))
    except (ValueError, TypeError):
        sr = 48000

    try:
        th_param = request.args.get("th", str(THRESHOLD))
        th = float(th_param)
    except (ValueError, TypeError):
        th = THRESHOLD

    raw_data = request.get_data()
    if not raw_data:
        return jsonify(error="data audio kosong"), 400

    # Pastikan byte buffer kelipatan 4 untuk konversi ke float32
    valid_bytes = len(raw_data) - (len(raw_data) % 4)
    if valid_bytes < (sr * 4 * 0.5):  # minimal 0.5 detik
        return jsonify(error="audio terlalu pendek"), 400

    y = np.frombuffer(raw_data[:valid_bytes], dtype=np.float32).copy()
    if not np.isfinite(y).all():
        return jsonify(error="audio mengandung nilai NaN/Inf"), 400

    # Resample ke 16 kHz jika sample rate browser berbeda
    if sr != ft.SR:
        try:
            y = resample_audio(y, orig_sr=sr, target_sr=ft.SR)
        except Exception as e:
            return jsonify(error=f"gagal resample: {e}"), 500

    # Ekstraksi fitur dan inferensi model SVM
    try:
        feats = ft.signal_to_features(y)
        X = pd.DataFrame(feats, columns=ft.FEATURE_NAMES)
        prob = float(model.predict_proba(X)[0, 1])
        noisy = bool(prob >= th)
        db = float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9))
        log_event(prob, noisy, db)
        return jsonify(prob=prob, noisy=noisy, db=db, threshold=th)
    except Exception as e:
        return jsonify(error=f"inferensi gagal: {e}"), 500


@app.route("/predict_file", methods=["POST", "OPTIONS"])
def predict_file():
    """Analisis file audio unggahan: segmentasi dan prediksi per chunk 5 detik."""
    if request.method == "OPTIONS":
        return "", 204

    f = request.files.get("file")
    if not f:
        return jsonify(error="file tidak ditemukan dalam request"), 400

    try:
        th_param = request.form.get("th", str(THRESHOLD))
        th = float(th_param)
    except (ValueError, TypeError):
        th = THRESHOLD

    # Simpan sementara ke disk agar librosa & audioread/ffmpeg dapat membaca format MP3/WAV secara andal
    suffix = Path(f.filename).suffix if f.filename else ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        y, _ = librosa.load(tmp_path, sr=ft.SR, mono=True)
    except Exception as e:
        return jsonify(error=f"gagal membaca audio: {e}"), 400
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    try:
        y_clean = ft.remove_silence(y)
        chunks = ft.make_chunks(y_clean)
        if not chunks:
            # Jika audio sangat pendek atau hampir seluruhnya sunyi, fallback ke sinyal awal
            chunks = ft.make_chunks(y)
        if not chunks:
            return jsonify(error="audio terlalu pendek / hampir seluruhnya hening"), 400

        X_mat = np.vstack([ft.extract_features(c) for c in chunks])
        X = pd.DataFrame(X_mat, columns=ft.FEATURE_NAMES)
        probs = model.predict_proba(X)[:, 1]
        flags = [bool(p >= th) for p in probs]
        noisy_ratio = float(np.mean(flags))
        verdict = "BERISIK" if noisy_ratio >= 0.5 else "TENANG"

        return jsonify(
            chunks=[{"prob": float(p), "noisy": b} for p, b in zip(probs, flags)],
            noisy_ratio=noisy_ratio,
            verdict=verdict,
            threshold=th
        )
    except Exception as e:
        return jsonify(error=f"analisis fitur gagal: {e}"), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8010))
    print(f"Starting Smart Noise Alert System (Flask) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
