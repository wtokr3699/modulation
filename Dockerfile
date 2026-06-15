FROM python:3.11-slim

# ffmpeg + libsndfile + Node.js + git (bgutil 빌드에 필요)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 nodejs npm git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# bgutil PO Token 서버 빌드 (Node.js, 개인 계정 불필요)
RUN git clone --depth 1 --branch 1.3.1 \
    https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git /bgutil && \
    cd /bgutil/server && \
    npm ci --no-audit --no-fund && \
    npx tsc && \
    npm prune --omit=dev

COPY app.py index.html ./

EXPOSE 5000

CMD ["python", "app.py"]
