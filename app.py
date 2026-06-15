"""
반음 전조 도구 — 로컬 서버

사용법:
  pip install -r requirements.txt
  python app.py
  → 브라우저에서 http://localhost:5000 열기
"""

import atexit
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import warnings

import librosa
import numpy as np
import yt_dlp
from flask import Flask, jsonify, request, send_file
from werkzeug.utils import secure_filename

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
_BGUTIL_READY = False
_BGUTIL_BASE_URL = os.environ.get('BGUTIL_BASE_URL', 'http://127.0.0.1:4416').rstrip('/')


def _start_bgutil():
    global _BGUTIL_PROC, _BGUTIL_READY
    if not os.path.exists(_BGUTIL_SCRIPT):
        return  # 로컬 개발 환경에서는 스킵
    try:
        import socket
        _BGUTIL_PROC = subprocess.Popen(
            ['node', _BGUTIL_SCRIPT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(30):
            time.sleep(0.5)
            if _BGUTIL_PROC.poll() is not None:
                raise RuntimeError(f'프로세스 종료됨 (code={_BGUTIL_PROC.returncode})')
            try:
                with socket.create_connection(('127.0.0.1', 4416), timeout=1):
                    _BGUTIL_READY = True
                    print('[bgutil] PO Token 서버 준비 완료 (port 4416)')
                    return
            except OSError:
                pass
        print('[bgutil] 15초 내 포트 준비 실패')
    except Exception as e:
        print(f'[bgutil] 서버 시작 실패: {e}')


def _stop_bgutil():
    if _BGUTIL_PROC:
        _BGUTIL_PROC.terminate()


atexit.register(_stop_bgutil)
_start_bgutil()

# ── Tor SOCKS5 프록시 (YouTube IP 차단 우회용) ─────────────────────────────────
_TOR_READY = False


def _start_tor():
    global _TOR_READY
    if not shutil.which('tor'):
        return
    try:
        import socket
        os.makedirs('/tmp/tor_data', exist_ok=True)
        subprocess.Popen(
            ['tor', '--SocksPort', '9050',
             '--DataDirectory', '/tmp/tor_data',
             '--Log', 'err stderr'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(90):  # 최대 90초 대기
            time.sleep(1)
            try:
                with socket.create_connection(('127.0.0.1', 9050), timeout=1):
                    _TOR_READY = True
                    print('[tor] SOCKS5 프록시 준비 완료 (port 9050)')
                    return
            except OSError:
                pass
        print('[tor] 90초 내 부트스트랩 실패')
    except Exception as e:
        print(f'[tor] 시작 오류: {e}')


threading.Thread(target=_start_tor, daemon=True).start()

app = Flask(__name__, static_folder='.')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 파일 업로드 최대 50MB

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

# 임시 파일 레지스트리: {file_id: {'path': filepath, 'title': str, ...}}
TEMP = {}
TEMP_LOCK = threading.Lock()

# 허용된 오디오 확장자
ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.ogg', '.flac', '.m4a', '.webm', '.opus', '.aac'}

# 동시 분석 요청 제한
_analyze_semaphore = threading.Semaphore(3)

# YouTube 임시 캐시 설정
YT_CACHE_TTL_SECONDS = int(os.environ.get('YT_CACHE_TTL_SECONDS', '1800'))
YT_MAX_DURATION_SECONDS = int(os.environ.get('YT_MAX_DURATION_SECONDS', '1200'))
YT_MAX_FILESIZE = int(os.environ.get('YT_MAX_FILESIZE', str(80 * 1024 * 1024)))
YT_TOR_WAIT_SECONDS = int(os.environ.get('YT_TOR_WAIT_SECONDS', '0'))
YT_SOCKET_TIMEOUT = int(os.environ.get('YT_SOCKET_TIMEOUT', '15'))
YT_RETRIES = int(os.environ.get('YT_RETRIES', '1'))
YT_FRAGMENT_RETRIES = int(os.environ.get('YT_FRAGMENT_RETRIES', '1'))

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


def estimate_key(chroma):
    """Krumhansl-Schmuckler 알고리즘으로 조(Key) 추정.
    Returns: (key_str, key_root_idx, is_major)
    """
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


def get_chord_progression(chroma, sr, beat_times, key_root, is_major):
    """비트 단위 코드 진행 추출. 연속 중복은 병합해서 반환."""
    hop = 512
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


# ── YouTube 오디오 준비 엔진 ───────────────────────────────────────────────────

def _try_pytubefix(yt_url):
    """pytubefix (InnerTube API)로 오디오 스트림 URL 추출."""
    from pytubefix import YouTube
    yt = YouTube(yt_url)
    audio = yt.streams.filter(only_audio=True).order_by('abr').last()
    if not audio:
        raise RuntimeError("오디오 스트림을 찾을 수 없습니다")
    return audio.url, yt.title or '유튜브 오디오', int(yt.length or 0)


# ── YouTube 우회 엔진 2: Invidious ───────────────────────────────────────────
_INVIDIOUS_INSTANCES = [
    'https://invidious.privacyredirect.com',
    'https://iv.datura.network',
    'https://invidious.fdn.fr',
    'https://inv.tux.pizza',
    'https://invidious.nerdvpn.de',
    'https://invidious.io',
]

_YT_VIDEO_ID_RE = re.compile(
    r'(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})'
)


def _extract_video_id(yt_url):
    m = _YT_VIDEO_ID_RE.search(yt_url)
    if not m:
        raise RuntimeError('YouTube 동영상 ID를 추출할 수 없습니다')
    return m.group(1)


def _try_invidious(yt_url):
    """Invidious 오픈소스 프론트엔드 공개 인스턴스로 오디오 URL 추출."""
    import json
    import urllib.request as _req
    video_id = _extract_video_id(yt_url)
    last_err = None
    for base in _INVIDIOUS_INSTANCES:
        try:
            req = _req.Request(
                f'{base}/api/v1/videos/{video_id}',
                headers={'User-Agent': 'Mozilla/5.0'},
            )
            with _req.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            title    = data.get('title', '유튜브 오디오')
            duration = int(data.get('lengthSeconds', 0))
            formats  = [f for f in data.get('adaptiveFormats', [])
                        if f.get('type', '').startswith('audio/')]
            if not formats:
                raise RuntimeError('오디오 스트림 없음')
            formats.sort(key=lambda f: int(f.get('bitrate', 0)), reverse=True)
            audio_url = formats[0]['url']
            app.logger.info(f'[invidious] {base} 성공')
            return audio_url, title, duration
        except Exception as e:
            app.logger.warning(f'[invidious] {base} 실패: {e}')
            last_err = e
    raise last_err or RuntimeError('Invidious 모든 인스턴스 실패')


# ── YouTube 우회 엔진 3: Cobalt ───────────────────────────────────────────────

def _try_cobalt(yt_url):
    """cobalt.tools 공개 API로 오디오 URL 추출."""
    import json
    import urllib.request as _req
    payload = json.dumps({
        'url': yt_url,
        'downloadMode': 'audio',
        'audioFormat': 'best',
    }).encode()
    req = _req.Request(
        'https://api.cobalt.tools/',
        data=payload,
        headers={
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }
    )
    with _req.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    status = data.get('status')
    if status not in ('tunnel', 'stream', 'redirect'):
        code = data.get('error', {}).get('code', '') or str(data)
        raise RuntimeError(f'cobalt 응답 오류: {code}')
    audio_url = data.get('url')
    if not audio_url:
        raise RuntimeError('cobalt URL 없음')
    app.logger.info('[cobalt] 성공')
    return audio_url, '유튜브 오디오', 0


def _try_y2mate(yt_url):
    """y2mate 2단계 API: analyze → convert → 자사 CDN URL 반환.
    반환 URL이 YouTube CDN이 아닌 y2mate CDN이므로 Render.com IP 제한 우회 가능.
    """
    import json
    import urllib.request as _req
    import urllib.parse

    hdrs = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'X-Requested-With': 'XMLHttpRequest',
        'Referer': 'https://www.y2mate.com/',
        'Origin': 'https://www.y2mate.com',
    }

    # Step 1: 영상 분석 — vid, title, 변환 토큰 획득
    data1 = urllib.parse.urlencode({
        'k_query': yt_url,
        'k_page': 'home',
        'hl': 'en',
        'q_auto': '0',
    }).encode()
    req1 = _req.Request(
        'https://www.y2mate.com/mates/analyzeV2/ajax',
        data=data1, headers=hdrs,
    )
    with _req.urlopen(req1, timeout=20) as r:
        info = json.loads(r.read())

    if info.get('status') != 'Ok':
        raise RuntimeError(f"y2mate 분석 실패: {info.get('mess') or info.get('status')}")

    vid   = info['vid']
    title = info.get('title', '유튜브 오디오')
    dur   = int(info.get('t', 0))

    links_mp3 = info.get('links', {}).get('mp3', {})
    if not links_mp3:
        raise RuntimeError('y2mate: MP3 포맷 링크 없음')

    token = None
    for q in ['mp3128', 'mp3320', 'mp364']:
        if q in links_mp3:
            token = links_mp3[q]['k']
            break
    if not token:
        token = next(iter(links_mp3.values()))['k']

    # Step 2: 변환 요청 — dlink (y2mate CDN URL) 획득
    data2 = urllib.parse.urlencode({'vid': vid, 'k': token}).encode()
    req2 = _req.Request(
        'https://www.y2mate.com/mates/convertV2/index',
        data=data2, headers=hdrs,
    )
    with _req.urlopen(req2, timeout=30) as r:
        result = json.loads(r.read())

    if result.get('c_status') != 'CONVERTED':
        raise RuntimeError(f"y2mate 변환 실패: {result.get('mess') or result.get('c_status')}")

    dlink = result.get('dlink')
    if not dlink:
        raise RuntimeError('y2mate: dlink 없음')

    app.logger.info('[y2mate] 성공')
    return dlink, title, dur


_YDL_CLIENTS = [['web'], ['ios'], ['android'], ['web_creator'], ['mweb']]


class _YtdlpLogger:
    def debug(self, msg):
        pass

    def warning(self, msg):
        app.logger.debug(msg)

    def error(self, msg):
        app.logger.debug(msg)


def _should_use_tor(cookies_file):
    mode = os.environ.get('YT_USE_TOR', 'auto').strip().lower()
    if mode in {'0', 'false', 'no', 'off'}:
        return False
    if mode in {'1', 'true', 'yes', 'on'}:
        return _TOR_READY
    return _TOR_READY and not cookies_file


def _wait_for_tor_if_needed():
    mode = os.environ.get('YT_USE_TOR', 'auto').strip().lower()
    if mode not in {'1', 'true', 'yes', 'on'}:
        return
    if not shutil.which('tor') or _TOR_READY or YT_TOR_WAIT_SECONDS <= 0:
        return
    deadline = time.time() + YT_TOR_WAIT_SECONDS
    while time.time() < deadline:
        if _TOR_READY:
            return
        time.sleep(0.5)


def _yt_match_filter(info_dict, *args, **kwargs):
    duration = info_dict.get('duration')
    if duration and duration > YT_MAX_DURATION_SECONDS:
        return f"영상이 너무 깁니다. 최대 {YT_MAX_DURATION_SECONDS // 60}분까지 지원합니다."

    size = info_dict.get('filesize') or info_dict.get('filesize_approx')
    if size and size > YT_MAX_FILESIZE:
        return "오디오 파일이 너무 큽니다. 더 짧은 영상을 사용해 주세요."

    return None


def _ytdlp_base_opts(clients):
    """yt-dlp 공통 옵션. 쿠키/Tor는 추출과 다운로드에 동일하게 적용한다."""
    cookies_file = _get_yt_cookies()
    bgutil_configured = _BGUTIL_READY or bool(os.environ.get('BGUTIL_BASE_URL'))
    fetch_pot = os.environ.get('YT_FETCH_POT', 'always' if bgutil_configured else 'auto')
    youtube_args = {
        'player_client': clients,
        # Render 같은 클라우드 IP에서 PO Token provider가 자동 판단 전에 필요할 때가 있어 강제로 요청한다.
        'fetch_pot': [fetch_pot],
        'pot_trace': [os.environ.get('YT_POT_TRACE', 'true')],
    }
    extractor_args = {'youtube': youtube_args}
    if bgutil_configured:
        extractor_args['youtubepot-bgutilhttp'] = {'base_url': [_BGUTIL_BASE_URL]}

    opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'socket_timeout': YT_SOCKET_TIMEOUT,
        'retries': YT_RETRIES,
        'fragment_retries': YT_FRAGMENT_RETRIES,
        'continuedl': False,
        'overwrites': True,
        'max_filesize': YT_MAX_FILESIZE,
        'match_filter': _yt_match_filter,
        'noprogress': True,
        'logger': _YtdlpLogger(),
        'extractor_args': extractor_args,
    }
    if os.path.exists(FFMPEG):
        opts['ffmpeg_location'] = FFMPEG
    if cookies_file:
        opts['cookiefile'] = cookies_file
    if _should_use_tor(cookies_file):
        opts['proxy'] = 'socks5://127.0.0.1:9050'
    return opts


def _find_downloaded_audio(workdir):
    """yt-dlp가 만든 오디오 파일 중 브라우저에서 디코딩하기 좋은 결과를 고른다."""
    candidates = []
    for name in os.listdir(workdir):
        path = os.path.join(workdir, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in ALLOWED_EXTENSIONS:
            candidates.append(path)
    if not candidates:
        raise RuntimeError("다운로드된 오디오 파일을 찾을 수 없습니다.")
    return max(candidates, key=lambda p: os.path.getsize(p))


def _download_with_ytdlp(yt_url, workdir):
    """yt-dlp로 여러 클라이언트를 순차 시도해 서버에 오디오 파일을 저장한다."""
    last_err = None
    for clients in _YDL_CLIENTS:
        try:
            opts = _ytdlp_base_opts(clients)
            opts.update({
                'outtmpl': os.path.join(workdir, '%(id)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
            })
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(yt_url, download=True)
            path = _find_downloaded_audio(workdir)
            return path, info.get('title', '유튜브 오디오'), int(info.get('duration') or 0)
        except Exception as e:
            last_err = e
            for name in os.listdir(workdir):
                try:
                    os.remove(os.path.join(workdir, name))
                except OSError:
                    pass
    raise last_err or RuntimeError("yt-dlp 모든 클라이언트 실패")


def _download_stream_url(stream_url, workdir, title, duration):
    """pytubefix 스트림 URL 폴백. 브라우저가 아닌 서버가 직접 받아 CORS를 피한다."""
    import urllib.request

    path = os.path.join(workdir, 'youtube-audio.m4a')
    req = urllib.request.Request(stream_url, headers={
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15',
        'Referer': 'https://www.youtube.com/',
    })
    downloaded = 0
    with urllib.request.urlopen(req, timeout=120) as resp, open(path, 'wb') as out:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            downloaded += len(chunk)
            if downloaded > YT_MAX_FILESIZE:
                raise RuntimeError("오디오 파일이 너무 큽니다. 더 짧은 영상을 사용해 주세요.")
            out.write(chunk)
    return path, title, duration


def _prepare_yt_audio(yt_url):
    """YouTube URL을 서버의 임시 오디오 파일로 준비한다.
    엔진 순서: yt-dlp → pytubefix → invidious (다중 인스턴스) → cobalt → y2mate
    """
    _wait_for_tor_if_needed()
    workdir = tempfile.mkdtemp(prefix='yt_audio_')

    # 1차: yt-dlp (서버 직접 다운로드, Tor 프록시 포함)
    try:
        path, title, duration = _download_with_ytdlp(yt_url, workdir)
        if duration > YT_MAX_DURATION_SECONDS:
            raise RuntimeError(f"영상이 너무 깁니다. 최대 {YT_MAX_DURATION_SECONDS // 60}분까지 지원합니다.")
        app.logger.info('YouTube 오디오 준비 성공: yt-dlp')
        return path, title, duration, workdir
    except Exception as e:
        app.logger.warning(f'YouTube 오디오 준비 실패 (yt-dlp): {e}')

    # 2차: 스트림 URL 추출 후 서버 다운로드 (pytubefix → invidious → cobalt → y2mate)
    for name, fn in [
        ('pytubefix', _try_pytubefix),
        ('invidious',  _try_invidious),
        ('cobalt',     _try_cobalt),
        ('y2mate',     _try_y2mate),
    ]:
        try:
            audio_url, title, duration = fn(yt_url)
            if duration > 0 and duration > YT_MAX_DURATION_SECONDS:
                raise RuntimeError(f"영상이 너무 깁니다. 최대 {YT_MAX_DURATION_SECONDS // 60}분까지 지원합니다.")
            path, title, duration = _download_stream_url(audio_url, workdir, title, duration)
            app.logger.info(f'YouTube 오디오 준비 성공: {name}')
            return path, title, duration, workdir
        except Exception as e:
            app.logger.warning(f'YouTube 오디오 준비 실패 ({name}): {e}')

    shutil.rmtree(workdir, ignore_errors=True)
    raise RuntimeError(
        "유튜브 영상을 불러올 수 없습니다. "
        "yt-dlp, pytubefix, Invidious, Cobalt, y2mate 모두 실패했습니다. "
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


def _cleanup_temp_entry(fid):
    with TEMP_LOCK:
        entry = TEMP.pop(fid, None)
    if not isinstance(entry, dict):
        return
    workdir = entry.get('workdir')
    path = entry.get('path')
    if workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    elif path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _schedule_cleanup(fid):
    def cleanup_later():
        time.sleep(YT_CACHE_TTL_SECONDS)
        _cleanup_temp_entry(fid)

    threading.Thread(target=cleanup_later, daemon=True).start()


@app.route('/yt/download', methods=['POST'])
def yt_download():
    """YouTube URL → 서버 임시 오디오 파일 준비 → ID 반환."""
    data = request.get_json(silent=True) or {}
    url = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'URL이 비어 있습니다.'}), 400
    if not is_valid_youtube_url(url):
        return jsonify({'error': '유튜브 URL만 허용됩니다.'}), 400

    try:
        path, title, duration, workdir = _prepare_yt_audio(url)
    except Exception as e:
        return jsonify({'error': str(e)[:300]}), 422

    fid = str(uuid.uuid4())
    mimetype = mimetypes.guess_type(path)[0] or 'audio/mpeg'
    with TEMP_LOCK:
        TEMP[fid] = {
            'path': path,
            'title': title,
            'duration': duration,
            'workdir': workdir,
            'mimetype': mimetype,
            'created_at': time.time(),
        }
    _schedule_cleanup(fid)

    return jsonify({
        'id': fid,
        'title': title,
        'duration': duration,
        'size': os.path.getsize(path),
    })


@app.route('/yt/audio/<fid>')
def serve_audio(fid):
    """서버에 준비된 임시 오디오 파일을 같은 출처에서 제공."""
    try:
        uuid.UUID(fid)
    except ValueError:
        return 'Invalid ID', 400

    with TEMP_LOCK:
        entry = TEMP.get(fid)
    if not entry or not isinstance(entry, dict):
        return 'Not found', 404

    path = entry.get('path')
    if not path or not os.path.exists(path):
        return 'Audio file expired', 404

    title = secure_filename(entry.get('title') or 'youtube-audio') or 'youtube-audio'
    ext = os.path.splitext(path)[1].lower() or '.mp3'
    return send_file(
        path,
        mimetype=entry.get('mimetype') or 'audio/mpeg',
        download_name=f'{title}{ext}',
        conditional=True,
        max_age=0,
    )


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
            with TEMP_LOCK:
                entry = TEMP.get(fid)
            if not entry:
                return jsonify({'error': '파일을 찾을 수 없습니다. 먼저 YouTube 불러오기를 실행하세요.'}), 404
            # TEMP 값이 dict(임시 파일)인 경우와 filepath(구형)인 경우 모두 처리
            if isinstance(entry, dict):
                audio_path = entry.get('path')
                if not audio_path or not os.path.exists(audio_path):
                    return jsonify({'error': '오디오 파일이 만료되었습니다. 다시 불러와 주세요.'}), 404
            else:
                audio_path = entry
                if not os.path.exists(audio_path):
                    return jsonify({'error': '파일을 찾을 수 없습니다.'}), 404
        else:
            return jsonify({'error': '파일 또는 file_id가 필요합니다.'}), 400

        if not _analyze_semaphore.acquire(blocking=False):
            return jsonify({'error': '현재 분석 요청이 많습니다. 잠시 후 다시 시도하세요.'}), 429

        try:
            y, sr = load_audio_via_ffmpeg(audio_path, max_duration=30)

            # BPM
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(tempo[0] if hasattr(tempo, '__len__') else tempo)
            onsets = librosa.onset.onset_detect(y=y, sr=sr)
            dur = librosa.get_duration(y=y, sr=sr)
            if bpm >= 110 and (len(onsets) / dur if dur > 0 else 0) < 2.3:
                bpm /= 2.0

            # Chroma 한 번만 계산 (Key + 코드 진행 공유)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)

            # Key + Meter
            key_str, key_root, is_major = estimate_key(chroma)
            meter = estimate_meter(y, sr, beats)

            # 코드 진행
            beat_times = librosa.frames_to_time(beats, sr=sr)
            chords = get_chord_progression(chroma, sr, beat_times, key_root, is_major)

            return jsonify({
                'bpm':    round(bpm, 1),
                'key':    key_str,
                'meter':  meter,
                'chords': chords,
            })
        finally:
            _analyze_semaphore.release()

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if cleanup_tmp and audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


def _run_hpss(audio_path, workdir=None):
    """HPSS로 오디오를 Harmonic / Percussive로 분리. 최대 60초."""
    import soundfile as sf
    y, sr = load_audio_via_ffmpeg(audio_path, max_duration=60)
    y_h, y_p = librosa.effects.hpss(y)
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix='stems_')
    path_h = os.path.join(workdir, 'harmonic.wav')
    path_p = os.path.join(workdir, 'percussive.wav')
    sf.write(path_h, y_h, sr)
    sf.write(path_p, y_p, sr)
    return {'harmonic': path_h, 'percussive': path_p}


@app.route('/stems/<fid>')
def get_stems(fid):
    """YouTube 파일 ID로 줄기 분리 수행."""
    try:
        uuid.UUID(fid)
    except ValueError:
        return jsonify({'error': '잘못된 ID'}), 400

    with TEMP_LOCK:
        entry = TEMP.get(fid)
    if not entry or not isinstance(entry, dict):
        return jsonify({'error': '파일을 찾을 수 없습니다. 다시 불러와 주세요.'}), 404

    if entry.get('stems'):
        return jsonify({
            'harmonic': f'/stem-audio/{fid}/harmonic',
            'percussive': f'/stem-audio/{fid}/percussive',
        })

    audio_path = entry.get('path')
    if not audio_path or not os.path.exists(audio_path):
        return jsonify({'error': '오디오 파일이 만료되었습니다. 다시 불러와 주세요.'}), 404

    try:
        workdir = entry.get('workdir') or tempfile.mkdtemp(prefix='stems_')
        stems = _run_hpss(audio_path, workdir)
        with TEMP_LOCK:
            entry['stems'] = stems
        return jsonify({
            'harmonic': f'/stem-audio/{fid}/harmonic',
            'percussive': f'/stem-audio/{fid}/percussive',
        })
    except Exception as e:
        return jsonify({'error': str(e)[:200]}), 500


@app.route('/stems/file', methods=['POST'])
def stems_from_file():
    """업로드 파일로 줄기 분리 수행."""
    if 'file' not in request.files:
        return jsonify({'error': '파일이 필요합니다.'}), 400

    f = request.files['file']
    ext = safe_extension(f.filename)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    f.save(tmp)
    tmp.close()
    audio_path = tmp.name

    try:
        stems = _run_hpss(audio_path)
        fid = str(uuid.uuid4())
        workdir = os.path.dirname(stems['harmonic'])
        with TEMP_LOCK:
            TEMP[fid] = {'stems': stems, 'workdir': workdir}
        _schedule_cleanup(fid)
        return jsonify({
            'id': fid,
            'harmonic': f'/stem-audio/{fid}/harmonic',
            'percussive': f'/stem-audio/{fid}/percussive',
        })
    except Exception as e:
        return jsonify({'error': str(e)[:200]}), 500
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


@app.route('/stem-audio/<fid>/<stem_type>')
def serve_stem(fid, stem_type):
    """분리된 줄기 오디오 파일 서빙."""
    try:
        uuid.UUID(fid)
    except ValueError:
        return 'Invalid ID', 400

    if stem_type not in ('harmonic', 'percussive'):
        return 'Invalid stem type', 400

    with TEMP_LOCK:
        entry = TEMP.get(fid)
    if not entry or not isinstance(entry, dict):
        return 'Not found', 404

    stems = entry.get('stems', {})
    path = stems.get(stem_type)
    if not path or not os.path.exists(path):
        return 'Stem not found', 410

    return send_file(path, mimetype='audio/wav', conditional=True, max_age=0)


if __name__ == '__main__':
    print('=' * 52)
    print('  반음 전조 도구 + 음악 분석 서버 시작')
    print('  http://localhost:5000 을 브라우저에서 열어주세요')
    ck = _get_yt_cookies()
    bgutil_ok = _BGUTIL_PROC is not None
    tor_installed = bool(shutil.which('tor'))
    print(f'  YouTube 쿠키  : {"✅ " + ck if ck else "❌ 미설정"}')
    print(f'  bgutil PO 토큰: {"✅ 준비됨 " + _BGUTIL_BASE_URL if _BGUTIL_READY else ("⚠️ 실행 확인 안 됨" if bgutil_ok else "❌ 미설치 (로컬 모드)")}')
    print(f'  Tor 프록시    : {"✅ 부트스트랩 중 (port 9050)..." if tor_installed else "❌ 미설치"}')
    print('=' * 52)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
