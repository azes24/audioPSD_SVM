FROM python:3.10-slim

WORKDIR /app

# Install dependency sistem Linux untuk pemrosesan audio
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh file project ke dalam container
COPY . /app

# Konfigurasi environment untuk Flask
ENV PORT=8010
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8010
ENV PYTHONUNBUFFERED=1

# Port untuk Kelompok 10
EXPOSE 8010

# Jalankan aplikasi Flask pada port 8010
CMD ["python", "app.py"]
