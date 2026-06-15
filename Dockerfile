FROM python:3.11-slim

# ffmpeg + libsndfile (librosa 의존성)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py index.html ./

EXPOSE 5000

CMD ["python", "app.py"]
