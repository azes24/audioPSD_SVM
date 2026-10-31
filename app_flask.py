"""Smart Noise Alert System - backend Flask.
Jalankan: python app.py  ->  http://localhost:5000
"""
import csv, datetime, io, math, warnings
from pathlib import Path

import joblib
import librosa
import numpy as np
import scipy.signal
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import features as ft

bundle = joblib.load("model_noise.pkl")
model = bundle["model"]
THRESHOLD = float(bundle.get("threshold", 0.6))
LOG_FILE = Path("log_kebisingan.csv")

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


def resample_audio(y, orig_sr, target_sr):
    """Resample audio cepat menggunakan resample_poly (jauh lebih efisien dari default librosa)."""
    if orig_sr == target_sr:
        return y
    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return scipy.signal.resample_poly(y, up, down).astype(np.float32)


def log_event(prob, noisy, db):
    new = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["waktu", "probabilitas", "berisik", "level_dbfs"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"),
                    round(prob, 4), int(noisy), round(db, 1)])


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
    alarm_path = Path("alarm.mp3")
    if alarm_path.exists():
        return send_file(alarm_path, mimetype="audio/mpeg")
    return jsonify(error="alarm.mp3 tidak ditemukan"), 404


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """Terima PCM float32 mentah (mono) dari browser, query ?sr=<sample rate browser>&th=<threshold>."""
    if request.method == "OPTIONS":
        return "", 204
    sr = int(request.args.get("sr", 48000))
    th = float(request.args.get("th", THRESHOLD))
    y = np.frombuffer(request.get_data(), dtype=np.float32)
    if len(y) < sr or not np.isfinite(y).all():
        return jsonify(error="audio tidak valid / terlalu pendek"), 400
    if sr != ft.SR:
        y = resample_audio(y, orig_sr=sr, target_sr=ft.SR)
    feats = ft.signal_to_features(y)
    X = pd.DataFrame(feats, columns=ft.FEATURE_NAMES)
    prob = float(model.predict_proba(X)[0, 1])
    noisy = prob >= th
    db = float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9))
    log_event(prob, noisy, db)
    return jsonify(prob=prob, noisy=bool(noisy), db=db, threshold=th)


@app.route("/predict_file", methods=["POST", "OPTIONS"])
def predict_file():
    """Analisis file audio unggahan: prediksi per chunk 5 detik."""
    if request.method == "OPTIONS":
        return "", 204
    f = request.files.get("file")
    th = float(request.form.get("th", THRESHOLD))
    if not f:
        return jsonify(error="file tidak ada"), 400
    try:
        y, _ = librosa.load(io.BytesIO(f.read()), sr=ft.SR, mono=True)
    except Exception as e:
        return jsonify(error=f"gagal membaca audio: {e}"), 400
    chunks = ft.make_chunks(ft.remove_silence(y))
    if not chunks:
        return jsonify(error="audio terlalu pendek / hampir seluruhnya hening"), 400
    X_mat = np.vstack([ft.extract_features(c) for c in chunks])
    X = pd.DataFrame(X_mat, columns=ft.FEATURE_NAMES)
    probs = model.predict_proba(X)[:, 1]
    flags = probs >= th
    return jsonify(chunks=[{"prob": float(p), "noisy": bool(b)} for p, b in zip(probs, flags)],
                   noisy_ratio=float(flags.mean()), verdict="BERISIK" if flags.mean() >= 0.5 else "TENANG",
                   threshold=th)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
