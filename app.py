"""
반음 전조 도구 — 로컬 서버

사용법:
  pip install -r requirements.txt
  python app.py
  → 브라우저에서 http://localhost:5000 열기
"""

import atexit
import io
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
import warnings

import librosa
import numpy as np
import yt_dlp
from flask import Flask, jsonify, request, send_file

warnings.simplefilter("ignore")

# pytubefix SSL 인증서 경로 설정 (Linux 환경에서는 자동으로 처리됨)
try:
    import certifi as _certifi
    os.environ.setdefault('SSL_CERT_FILE', _certifi.where())
    os.environ.setdefault('REQUESTS_CA_BUNDLE', _certifi.where())
except ImportError:
    pass

# ── bgutil PO Token 서버 (YouTube 봇 차단 우회, 개인 계정 불필요) ──────────────
_BGUTIL_SCRIPT = '/bgutil/server/build/main.js'
_BGUTIL_PROC = None


def _start_bgutil():
    global _BGUTIL_PROC
    if not os.path.exists(_BGUTIL_SCRIPT):
        return  # 로컬 개발 환경에서는 스킵
    try:
        _BGUTIL_PROC = subprocess.Popen(
            ['node', _BGUTIL_SCRIPT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)  # 서버 초기화 대기
        print('[bgutil] PO Token 서버 시작 완료 (port 4416)')
    except Exception as e:
        print(f'[bgutil] 서버 시작 실패: {e}')


def _stop_bgutil():
    if _BGUTIL_PROC:
        _BGUTIL_PROC.terminate()


atexit.register(_stop_bgutil)
_start_bgutil()

app = Flask(__name__, static_folder='.')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 파일 업로드 최대 50MB

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

# 임시 파일 레지스트리: {file_id: filepath}
TEMP = {}

# 허용된 오디오 확장자
ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.flac', '.m4a', '.webm', '.opus', '.aac'}

# 동시 분석 요청 제한
_analyze_semaphore = threading.Semaphore(3)

# YouTube URL 화이트리스트 (SSRF 방지)
_YT_PATTERN = re.compile(
    r'^https://(www\.)?(youtube\.com/(watch|shorts|embed)|youtu\.be/)',
    re.IGNORECASE,
)


def is_valid_youtube_url(url: str) -> bool:
    return bool(_YT_PATTERN.match(url))


def safe_extension(filename: str) -> str:
    """허용된 확장자만 반환. 그 외에는 .tmp."""
    ext = os.path.splitext(filename or '')[1].lower()
    return ext if ext in ALLOWED_EXTENSIONS else '.tmp'

# ── 음악 분석 함수 ─────────────────────────────────────────────────────────────
# 출처: /Users/applw/Desktop/coding/bpm/analyzer.py (단순화·확장)

CHORD_NAMES   = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJ_TEMPLATE  = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
MIN_TEMPLATE  = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)

# 스케일 degree → Roman numeral
MAJOR_DEGREES = {0: 'I',  2: 'ii', 4: 'iii', 5: 'IV', 7: 'V', 9: 'vi', 11: 'vii°'}
MINOR_DEGREES = {0: 'i', 2: 'ii°', 3: 'III', 5: 'iv', 7: 'v',  8: 'VI', 10: 'VII'}


def estimate_key(y, sr):
    """Krumhansl-Schmuckler 알고리즘으로 조(Key) 추정.
    Returns: (key_str, key_root_idx, is_major)
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)

    best_corr = -1
    best_key_str = 'C Major'
    best_root = 0
    best_is_major = True

    for i in range(12):
        corr_major = np.corrcoef(chroma_avg, np.roll(MAJOR_PROFILE, i))[0, 1]
        if corr_major > best_corr:
            best_corr = corr_major
            best_key_str = f"{CHORD_NAMES[i]} Major"
            best_root = i
            best_is_major = True

        corr_minor = np.corrcoef(chroma_avg, np.roll(MINOR_PROFILE, i))[0, 1]
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key_str = f"{CHORD_NAMES[i]} Minor"
            best_root = i
            best_is_major = False

    return best_key_str, best_root, best_is_major


def estimate_meter(y, sr, beats):
    """Autocorrelation으로 4/4 vs 6/8 판별."""
    try:
        if len(beats) < 10:
            return "4/4"
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_intervals = np.diff(beats)
        avg_beat_interval = int(np.round(np.mean(beat_intervals)))
        if avg_beat_interval == 0:
            return "4/4"
        max_lag = min(len(onset_env), avg_beat_interval * 8)
        ac = librosa.autocorrelate(onset_env, max_size=max_lag)
        window = max(1, avg_beat_interval // 8)

        def get_peak(mult):
            c = mult * avg_beat_interval
            if c >= len(ac):
                return 0
            return np.max(ac[max(0, c - window): min(len(ac), c + window + 1)])

        if get_peak(3) > get_peak(2) and get_peak(3) > get_peak(4):
            return "6/8"
        return "4/4"
    except Exception:
        return "4/4"


def calc_numeral(chord_root, chord_is_major, key_root, key_is_major):
    """코드의 스케일 degree를 Roman numeral로 변환."""
    degree = (chord_root - key_root) % 12
    table = MAJOR_DEGREES if key_is_major else MINOR_DEGREES
    return table.get(degree, f'♭{degree}')


def get_chord_progression(y, sr, beat_times, key_root, is_major):
    """비트 단위 코드 진행 추출. 연속 중복은 병합해서 반환."""
    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    raw = []

    for t in beat_times:
        f = librosa.time_to_frames(t, sr=sr, hop_length=hop)
        end = min(f + 8, chroma.shape[1])
        segment = np.mean(chroma[:, f:end], axis=1) if end > f else chroma[:, f]

        best_score, best_chord_str, best_root, best_is_maj = -1, 'N', 0, True
        for root in range(12):
            for tmpl, is_maj in [(MAJ_TEMPLATE, True), (MIN_TEMPLATE, False)]:
                score = np.dot(segment, np.roll(tmpl, root))
                if score > best_score:
                    best_score = score
                    best_chord_str = CHORD_NAMES[root] + ('' if is_maj else 'm')
                    best_root = root
                    best_is_maj = is_maj

        numeral = calc_numeral(best_root, best_is_maj, key_root, is_major)
        raw.append({'time': round(float(t), 3), 'chord': best_chord_str, 'numeral': numeral})

    # 연속 중복 병합
    deduped = []
    for item in raw:
        if not deduped or item['chord'] != deduped[-1]['chord']:
            deduped.append(item)

    return deduped


def load_audio_via_ffmpeg(filepath, max_duration=120):
    """ffmpeg로 오디오 파일 또는 URL을 float32 PCM으로 변환."""
    TARGET_SR = 22050
    is_url = filepath.startswith('http://') or filepath.startswith('https://')
    timeout = 300 if is_url else 180  # URL은 네트워크 다운로드 시간 포함
    proc = subprocess.run(
        [FFMPEG, '-i', filepath,
         '-ac', '1', '-ar', str(TARGET_SR),
         '-f', 'f32le', '-t', str(max_duration), '-'],
        capture_output=True, timeout=timeout,
    )
    if proc.returncode != 0 or len(proc.stdout) < 1000:
        raise RuntimeError(f"ffmpeg 변환 실패: {proc.stderr[-200:].decode(errors='ignore')}")
    y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return y, TARGET_SR


# ── YouTube 쿠키 설정 ─────────────────────────────────────────────────────────
# Render.com 등 클라우드 서버 IP는 YouTube 봇 차단 대상. 실제 계정 쿠키로 우회.
# 설정 방법 (우선순위 순):
#   1. 서버 파일: ./cookies.txt  (로컬 개발용)
#   2. 환경변수:  YT_COOKIES_PATH = "/etc/yt-cookies.txt"  (Render Secret Files)
#   3. 환경변수:  YT_COOKIES = "<cookies.txt 내용 전체>"    (Render 환경변수)

_YT_COOKIES_LOCK = threading.Lock()
_YT_COOKIES_FILE_CACHE = None


def _get_yt_cookies():
    """사용 가능한 YouTube 쿠키 파일 경로 반환. 없으면 None."""
    global _YT_COOKIES_FILE_CACHE

    # 1. 로컬 파일
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cookies.txt')
    if os.path.exists(local):
        return local

    # 2. 파일 경로 환경변수 (Render.com Secret Files 권장)
    env_path = os.environ.get('YT_COOKIES_PATH', '').strip()
    if env_path and os.path.exists(env_path):
        return env_path

    # 3. 쿠키 내용 환경변수 → 임시 파일로 저장 (최초 1회)
    cookie_content = os.environ.get('YT_COOKIES', '').strip()
    if cookie_content:
        with _YT_COOKIES_LOCK:
            if _YT_COOKIES_FILE_CACHE is None:
                try:
                    import base64
                    content = base64.b64decode(cookie_content).decode()
                except Exception:
                    content = cookie_content
                tmp = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.txt', delete=False, prefix='yt_cookies_'
                )
                tmp.write(content)
                tmp.close()
                _YT_COOKIES_FILE_CACHE = tmp.name
        return _YT_COOKIES_FILE_CACHE

    return None


# ── YouTube 오디오 추출 엔진 ───────────────────────────────────────────────────

def _try_pytubefix(yt_url):
    """pytubefix (InnerTube API)로 오디오 스트림 URL 추출."""
    from pytubefix import YouTube
    yt = YouTube(yt_url)
    audio = yt.streams.filter(only_audio=True).order_by('abr').last()
    if not audio:
        raise RuntimeError("오디오 스트림을 찾을 수 없습니다")
    return audio.url, yt.title or '유튜브 오디오', int(yt.length or 0)


_YDL_CLIENTS = [['web'], ['ios'], ['android'], ['web_creator'], ['mweb']]


def _try_ytdlp(yt_url):
    """yt-dlp로 여러 클라이언트를 순차 시도해 오디오 URL 추출. 쿠키 있으면 사용."""
    cookies_file = _get_yt_cookies()
    last_err = None
    for clients in _YDL_CLIENTS:
        try:
            opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio/best',
                'quiet': True,
                'no_warnings': True,
                'extractor_args': {'youtube': {'player_client': clients}},
            }
            if cookies_file:
                opts['cookiefile'] = cookies_file
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(yt_url, download=False)
            audio_url = None
            for f in reversed(info.get('formats', [])):
                if f.get('acodec') != 'none' and f.get('vcodec') in (None, 'none'):
                    audio_url = f.get('url')
                    break
            if not audio_url:
                audio_url = info.get('url')
            if audio_url:
                return audio_url, info.get('title', '유튜브 오디오'), int(info.get('duration', 0))
        except Exception as e:
            last_err = e
    raise last_err or RuntimeError("yt-dlp 모든 클라이언트 실패")


def _extract_yt_audio(yt_url):
    """쿠키 유무에 따라 최적 엔진을 선택해 오디오 스트림 URL 추출."""
    cookies_file = _get_yt_cookies()

    # 쿠키가 있으면 yt-dlp+cookies가 가장 신뢰성 높음 → 먼저 시도
    if cookies_file:
        engines = [('yt-dlp+cookies', _try_ytdlp), ('pytubefix', _try_pytubefix)]
    else:
        engines = [('pytubefix', _try_pytubefix), ('yt-dlp', _try_ytdlp)]

    for name, fn in engines:
        try:
            result = fn(yt_url)
            app.logger.info(f'YouTube 추출 성공: {name}')
            return result
        except Exception as e:
            app.logger.warning(f'YouTube 추출 실패 ({name}): {e}')

    raise RuntimeError(
        "유튜브 영상을 불러올 수 없습니다.\n"
        "서버 IP가 YouTube에 차단되어 있습니다. "
        "관리자에게 YouTube 쿠키 설정을 요청하거나, "
        "파일을 직접 업로드해 주세요."
    )


# ── Flask 라우트 ───────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': '파일 크기가 너무 큽니다 (최대 50MB).'}), 413


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/')
def index():
    return send_file('index.html')


@app.route('/yt/download', methods=['POST'])
def yt_download():
    """YouTube URL → 스트림 URL 추출 (다중 엔진 폴백) → ID + audio_url 반환."""
    data = request.get_json(silent=True) or {}
    url = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'URL이 비어 있습니다.'}), 400
    if not is_valid_youtube_url(url):
        return jsonify({'error': '유튜브 URL만 허용됩니다.'}), 400

    try:
        audio_url, title, duration = _extract_yt_audio(url)
    except Exception as e:
        return jsonify({'error': str(e)[:300]}), 422

    fid = str(uuid.uuid4())
    TEMP[fid] = {'stream_url': audio_url, 'title': title}
    threading.Thread(target=lambda: (time.sleep(600), TEMP.pop(fid, None)), daemon=True).start()

    return jsonify({'id': fid, 'title': title, 'duration': duration, 'audio_url': audio_url})


@app.route('/yt/audio/<fid>')
def serve_audio(fid):
    """Plan B 폴백: 브라우저 직접 fetch 실패 시 서버가 스트리밍 프록시 역할."""
    try:
        uuid.UUID(fid)
    except ValueError:
        return 'Invalid ID', 400

    entry = TEMP.get(fid)
    if not entry or not isinstance(entry, dict):
        return 'Not found', 404

    stream_url = entry.get('stream_url')
    if not stream_url:
        return 'No stream URL', 404

    def generate():
        req = urllib.request.Request(stream_url, headers={
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15',
            'Referer': 'https://www.youtube.com/',
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                yield chunk

    return app.response_class(generate(), mimetype='audio/mp4')


@app.route('/analyze', methods=['POST'])
def analyze():
    """BPM / Key / Meter / 코드 진행 분석.

    입력 A: multipart 파일 업로드  (form field: "file")
    입력 B: JSON {"file_id": "<yt-download-id>"}
    """
    cleanup_tmp = False
    audio_path = None

    try:
        if 'file' in request.files:
            f = request.files['file']
            ext = safe_extension(f.filename)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            f.save(tmp)
            tmp.close()
            audio_path = tmp.name
            cleanup_tmp = True
        elif request.is_json:
            fid = request.json.get('file_id', '')
            try:
                uuid.UUID(fid)
            except ValueError:
                return jsonify({'error': '잘못된 file_id'}), 400
            entry = TEMP.get(fid)
            if not entry:
                return jsonify({'error': '파일을 찾을 수 없습니다. 먼저 YouTube 불러오기를 실행하세요.'}), 404
            # TEMP 값이 dict(스트림 URL)인 경우와 filepath(구형)인 경우 모두 처리
            if isinstance(entry, dict):
                audio_path = entry['stream_url']
            else:
                audio_path = entry
                if not os.path.exists(audio_path):
                    return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404
        else:
            return jsonify({'error': '파일 또는 file_id가 필요합니다.'}), 400

        if not _analyze_semaphore.acquire(blocking=False):
            return jsonify({'error': '현재 분석 요청이 많습니다. 잠시 후 다시 시도하세요.'}), 429

        try:
            y, sr = load_audio_via_ffmpeg(audio_path)
        finally:
            _analyze_semaphore.release()

        # BPM
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo[0] if hasattr(tempo, '__len__') else tempo)
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        dur = librosa.get_duration(y=y, sr=sr)
        if bpm >= 110 and (len(onsets) / dur if dur > 0 else 0) < 2.3:
            bpm /= 2.0

        # Key + Meter
        key_str, key_root, is_major = estimate_key(y, sr)
        meter = estimate_meter(y, sr, beats)

        # 코드 진행
        beat_times = librosa.frames_to_time(beats, sr=sr)
        chords = get_chord_progression(y, sr, beat_times, key_root, is_major)

        return jsonify({
            'bpm':    round(bpm, 1),
            'key':    key_str,
            'meter':  meter,
            'chords': chords,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if cleanup_tmp and audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


if __name__ == '__main__':
    print('=' * 52)
    print('  반음 전조 도구 + 음악 분석 서버 시작')
    print('  http://localhost:5000 을 브라우저에서 열어주세요')
    ck = _get_yt_cookies()
    bgutil_ok = _BGUTIL_PROC is not None
    print(f'  YouTube 쿠키  : {"✅ " + ck if ck else "❌ 미설정"}')
    print(f'  bgutil PO 토큰: {"✅ 실행 중 (port 4416)" if bgutil_ok else "❌ 미설치 (로컬 모드)"}')
    print('=' * 52)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
