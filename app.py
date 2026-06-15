"""
반음 전조 도구 — 로컬 서버

사용법:
  pip install -r requirements.txt
  python app.py
  → 브라우저에서 http://localhost:5000 열기
"""

import io
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

warnings.simplefilter("ignore")

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
    """ffmpeg로 오디오 파일을 float32 PCM으로 변환. 모든 포맷 지원."""
    TARGET_SR = 22050
    proc = subprocess.run(
        [FFMPEG, '-i', filepath,
         '-ac', '1', '-ar', str(TARGET_SR),
         '-f', 'f32le', '-t', str(max_duration), '-'],
        capture_output=True, timeout=180,
    )
    if proc.returncode != 0 or len(proc.stdout) < 1000:
        raise RuntimeError(f"ffmpeg 변환 실패: {proc.stderr[-200:].decode(errors='ignore')}")
    y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return y, TARGET_SR


# ── Flask 라우트 ───────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': '파일 크기가 너무 큽니다 (최대 50MB).'}), 413


@app.route('/')
def index():
    return send_file('index.html')


@app.route('/yt/download', methods=['POST'])
def yt_download():
    """YouTube URL → 오디오 다운로드 → 임시 저장 → ID 반환."""
    data = request.get_json(silent=True) or {}
    url = data.get('url', '').strip()
    if not url:
        return jsonify({'error': 'URL이 비어 있습니다.'}), 400
    if not is_valid_youtube_url(url):
        return jsonify({'error': '유튜브 URL만 허용됩니다.'}), 400

    fid = str(uuid.uuid4())
    tmpdir = tempfile.mkdtemp(prefix='yt_audio_')

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(tmpdir, f'{fid}.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as e:
        return jsonify({'error': f'다운로드 실패: {str(e)[:200]}'}), 422
    except Exception as e:
        return jsonify({'error': f'오류: {str(e)[:200]}'}), 500

    files = os.listdir(tmpdir)
    if not files:
        return jsonify({'error': '파일이 생성되지 않았습니다.'}), 500

    filepath = os.path.join(tmpdir, files[0])
    TEMP[fid] = filepath

    def cleanup():
        time.sleep(600)
        try:
            os.remove(filepath)
            os.rmdir(tmpdir)
        except OSError:
            pass
        TEMP.pop(fid, None)

    threading.Thread(target=cleanup, daemon=True).start()

    title    = info.get('title', '유튜브 오디오')
    duration = info.get('duration', 0)

    return jsonify({'id': fid, 'title': title, 'duration': duration})


@app.route('/yt/audio/<fid>')
def serve_audio(fid):
    """임시 오디오 파일 서빙."""
    try:
        uuid.UUID(fid)
    except ValueError:
        return 'Invalid ID', 400

    filepath = TEMP.get(fid)
    if not filepath or not os.path.exists(filepath):
        return 'Not found', 404

    return send_file(filepath)


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
            audio_path = TEMP.get(fid)
            if not audio_path or not os.path.exists(audio_path):
                return jsonify({'error': '파일을 찾을 수 없습니다. 먼저 YouTube 불러오기를 실행하세요.'}), 404
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
    print('=' * 52)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
