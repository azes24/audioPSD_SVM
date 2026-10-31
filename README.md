# Smart Noise Alert System (Audio PSD - SVM)
**Klasifikasi Kebisingan Laboratorium Berbasis Analisis Sinyal Akustik & Support Vector Machine (SVM)**  
**Mata Kuliah:** Penambangan Sains Data (PSD) — **Kelompok 10**

---

## 📌 Ringkasan Proyek
Aplikasi sistem peringatan kebisingan cerdas (*Smart Noise Alert System*) yang dirancang untuk memantau ketertiban suasana laboratorium Teknik Informatika UTM secara otomatis, objektif, dan *real-time*.

Backend dibangun menggunakan **Flask** dan terintegrasi dengan model **Support Vector Machine (SVM - RBF Kernel)** dengan **14 Fitur Akustik Terbaik** hasil seleksi fitur **ANOVA F-Score**.

- **Framework Web:** Flask (HTML5 / JavaScript Web Audio API)
- **Port:** **8010** (Kelompok 10)
- **Sample Rate Seragam:** 16.000 Hz (16 kHz mono)
- **Segmentasi (Chunking):** Segmen 5 detik (80.000 sampel per chunk)
- **Decision Threshold:** 0.60 (60%)
- **Akurasi Model:** ~89% ($88.65\%$)

---

## 🚀 Fitur Utama Aplikasi Web (Flask)
1. **Pemantauan Mikrofon Real-Time (Live Mic):** Menangkap sinyal mikrofon browser via Web Audio API dan menganalisis kebisingan per window 5 detik secara langsung.
2. **Unggah File Audio (Upload):** Menganalisis file rekaman audio (`.wav`, `.mp3`) per chunk 5 detik secara komprehensif.
3. **Peringatan & Alarm Otomatis:** Alert visual dan pemutar suara alarm (`alarm.mp3`) otomatis aktif jika suasana ruangan terdeteksi berisik.
4. **Visualisasi Dinamis:** Live waveform audio meter, indikator dBFS, dan riwayat log deteksi kebisingan.

---

## 🐳 Deployment Menggunakan Docker & Docker Compose

Proyek ini telah dikonfigurasi lengkap untuk dideploy menggunakan Docker pada **Port 8010** (Kelompok 10).

### 1. Menjalankan dengan Docker Compose (Direkomendasikan)
Di direktori root project, jalankan:
```bash
docker-compose up --build -d
```
Atau jika menggunakan Docker Compose v2:
```bash
docker compose up --build -d
```

Aplikasi Flask dapat langsung diakses melalui browser di:
👉 **http://localhost:8010** atau **http://0.0.0.0:8010**

Untuk menghentikan container:
```bash
docker-compose down
```

### 2. Menjalankan Manual dengan Docker CLI
Build Docker image:
```bash
docker build -t audio_psd_svm_kel10 .
```

Jalankan container:
```bash
docker run -d -p 8010:8010 --name audio_psd_svm_kel10 --restart unless-stopped audio_psd_svm_kel10
```

---

## 📁 Struktur Direktori Proyek
```text
.
├── app.py                     # Aplikasi Web Flask utama (Port 8010)
├── templates/                 # Direktori template Flask
│   └── index.html             # Antarmuka web dashboard interaktif
├── Dockerfile                 # Konfigurasi container Docker (python:3.10-slim, port 8010)
├── docker-compose.yml         # Konfigurasi service Docker Compose (Port 8010:8010)
├── .dockerignore              # Mengabaikan file yang tidak perlu untuk image ringan
├── requirements.txt           # Dependensi pustaka Python
├── requerements.txt           # Duplikat requirements.txt (toleransi typo)
├── features.py                # Modul ekstraksi 14 fitur akustik
├── model_noise.pkl            # Model SVM Pipeline (StandardScaler + SVC RBF)
├── alarm.mp3                  # Aset audio peringatan alarm kebisingan
├── README.md                  # Dokumentasi proyek dan panduan deployment
└── Audio-APP/audioPSD_SVM/    # Salinan modul, dataset, dan notebook awal
```
