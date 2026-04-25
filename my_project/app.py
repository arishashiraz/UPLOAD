import io
import os
import uuid
import time
import struct
import wave

import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

try:
    from pydub import AudioSegment
    PYDUB_OK = True
except ImportError:
    PYDUB_OK = False

# ── CONFIG ────────────────────────────────────────────────────────────────────
MAX_MB         = 50
ALLOWED        = {"mp3", "wav", "ogg", "flac", "aac", "aiff", "m4a", "opus"}
TARGET_SR      = 22050
TARGET_MONO    = True
MP3_BITRATE    = "64k"
EXPORT_FORMAT  = "mp3"

DRC_THRESHOLD  = 0.35
DRC_RATIO      = 4.0
DRC_KNEE       = 0.05

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

# ── AUDIO PROCESSING ──────────────────────────────────────────────────────────

def load_audio(file_bytes: bytes, ext: str) -> "AudioSegment":
    fmt = ext.lstrip(".")
    if fmt == "mp3":
        fmt = "mp3"
    elif fmt in ("aiff",):
        fmt = "aiff"
    buf = io.BytesIO(file_bytes)
    return AudioSegment.from_file(buf, format=fmt)


def segment_to_numpy(seg: "AudioSegment") -> tuple:
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    max_val = float(2 ** (seg.sample_width * 8 - 1))
    samples /= max_val
    channels = seg.channels
    if channels > 1:
        samples = samples.reshape(-1, channels)
    return samples, seg.frame_rate, channels


def numpy_to_segment(samples: np.ndarray, sr: int, channels: int, sample_width: int = 2) -> "AudioSegment":
    max_val = float(2 ** (sample_width * 8 - 1)) - 1
    pcm = np.clip(samples * max_val, -max_val, max_val).astype(np.int16)
    if channels == 1:
        raw = pcm.tobytes()
    else:
        raw = pcm.flatten().tobytes()
    return AudioSegment(
        data=raw,
        sample_width=sample_width,
        frame_rate=sr,
        channels=channels,
    )


def soft_knee_compress(samples: np.ndarray, threshold: float, ratio: float, knee: float) -> np.ndarray:
    abs_s = np.abs(samples)
    sign  = np.sign(samples)
    lower = threshold - knee / 2
    upper = threshold + knee / 2
    compressed = np.copy(abs_s)
    knee_mask = (abs_s > lower) & (abs_s <= upper)
    if knee_mask.any():
        x = abs_s[knee_mask]
        t = (x - lower) / knee
        gain_reduction = (1 - 1 / ratio) * (t ** 2) / 2
        compressed[knee_mask] = x - gain_reduction * (x - threshold)
    above_mask = abs_s > upper
    if above_mask.any():
        x = abs_s[above_mask]
        excess = x - threshold
        compressed[above_mask] = threshold + excess / ratio
    return compressed * sign


def downsample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr <= target_sr:
        return samples
    ratio = orig_sr / target_sr
    if samples.ndim == 1:
        indices = np.round(np.arange(0, len(samples), ratio)).astype(int)
        indices = indices[indices < len(samples)]
        return samples[indices]
    else:
        indices = np.round(np.arange(0, samples.shape[0], ratio)).astype(int)
        indices = indices[indices < samples.shape[0]]
        return samples[indices, :]


def mix_to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    return samples.mean(axis=1)


def normalize(samples: np.ndarray, peak: float = 0.92) -> np.ndarray:
    m = np.max(np.abs(samples))
    if m > 0:
        samples = samples * (peak / m)
    return samples


def compress_audio(file_bytes: bytes, ext: str) -> tuple:
    if not PYDUB_OK:
        raise RuntimeError("pydub is not installed. Run: pip install pydub numpy")
    seg = load_audio(file_bytes, ext)
    orig_sr    = seg.frame_rate
    orig_ch    = seg.channels
    orig_sw    = seg.sample_width
    samples, sr, ch = segment_to_numpy(seg)
    if TARGET_MONO and ch > 1:
        samples = mix_to_mono(samples)
        ch = 1
    new_sr = min(sr, TARGET_SR)
    if sr > TARGET_SR:
        samples = downsample(samples, sr, TARGET_SR)
        sr = TARGET_SR
    samples = soft_knee_compress(samples, threshold=DRC_THRESHOLD, ratio=DRC_RATIO, knee=DRC_KNEE)
    samples = normalize(samples, peak=0.92)
    out_seg = numpy_to_segment(samples, sr, ch, sample_width=2)
    buf = io.BytesIO()
    if EXPORT_FORMAT == "mp3":
        out_seg.export(buf, format="mp3", bitrate=MP3_BITRATE)
    else:
        out_seg.export(buf, format="wav")
    output_bytes = buf.getvalue()
    stats = {
        "orig_sr":    orig_sr,
        "new_sr":     new_sr,
        "orig_ch":    orig_ch,
        "new_ch":     ch,
        "drc":        f"threshold={DRC_THRESHOLD}, ratio={DRC_RATIO}:1, knee={DRC_KNEE}",
        "bitrate":    MP3_BITRATE if EXPORT_FORMAT == "mp3" else "lossless",
        "format":     EXPORT_FORMAT,
    }
    return output_bytes, stats


# ── HELPERS ───────────────────────────────────────────────────────────────────

def ok_ext(name: str) -> bool:
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED


def hsize(n: int) -> str:
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} GB"


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML


@app.route("/api/status")
def api_status():
    return jsonify({"ok": True, "pydub": PYDUB_OK})


@app.route("/api/compress", methods=["POST"])
def api_compress():
    if not PYDUB_OK:
        return jsonify({"error": "pydub not installed. Run: pip install pydub numpy"}), 500
    if "file" not in request.files:
        return jsonify({"error": "No file in request."}), 400
    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"error": "No filename received."}), 400
    if not ok_ext(f.filename):
        return jsonify({"error": f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED))}"}), 400
    safe = secure_filename(f.filename)
    ext  = safe.rsplit(".", 1)[1].lower()
    file_bytes = f.read()
    orig_size  = len(file_bytes)
    t0 = time.time()
    try:
        out_bytes, stats = compress_audio(file_bytes, ext)
    except Exception as e:
        return jsonify({"error": f"Compression failed: {e}"}), 500
    elapsed = round(time.time() - t0, 1)
    comp_size = len(out_bytes)
    saved     = round((1 - comp_size / orig_size) * 100, 1) if orig_size else 0
    tok = uuid.uuid4().hex
    out_filename = os.path.splitext(safe)[0] + "_compressed." + EXPORT_FORMAT
    _store[tok] = (out_bytes, out_filename)
    return jsonify({
        "token":      tok,
        "filename":   out_filename,
        "orig_size":  hsize(orig_size),
        "comp_size":  hsize(comp_size),
        "saved":      saved,
        "elapsed":    elapsed,
        "stats":      stats,
    })


@app.route("/api/download/<token>/<filename>")
def api_download(token, filename):
    if token not in _store:
        return jsonify({"error": "File not found or already downloaded."}), 404
    out_bytes, out_filename = _store.pop(token)
    buf = io.BytesIO(out_bytes)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=out_filename,
                     mimetype="audio/mpeg" if EXPORT_FORMAT == "mp3" else "audio/wav")


_store: dict = {}

# ── EMBEDDED HTML ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Audio Compression</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #f4f4f5; display: flex; justify-content: center; padding: 40px 16px; }
.wrap { width: 100%; max-width: 480px; }
h1 { font-size: 20px; font-weight: 600; margin-bottom: 4px; }
p.sub { font-size: 13px; color: #71717a; margin-bottom: 24px; }
.box { background: #fff; border: 1px solid #e4e4e7; border-radius: 10px; padding: 20px; margin-bottom: 12px; }
input[type=file] { display: block; width: 100%; font-size: 14px; color: #3f3f46; cursor: pointer; }
input[type=file]::file-selector-button {
  margin-right: 12px; padding: 6px 14px; border: 1px solid #d4d4d8;
  border-radius: 6px; background: #fff; font-size: 13px; cursor: pointer;
}
input[type=file]::file-selector-button:hover { background: #f4f4f5; }
label.lbl { display: block; font-size: 13px; font-weight: 500; margin-bottom: 6px; color: #3f3f46; }
button {
  width: 100%; padding: 11px; background: #2563eb; color: #fff;
  border: none; border-radius: 8px; font-size: 14px; font-weight: 600;
  cursor: pointer; margin-top: 4px;
}
button:hover { background: #1d4ed8; }
button:disabled { opacity: 0.5; cursor: not-allowed; }
#status { font-size: 13px; color: #52525b; margin-top: 10px; min-height: 18px; }
.bar-wrap { height: 5px; background: #e4e4e7; border-radius: 3px; margin-top: 8px; overflow: hidden; }
.bar { height: 100%; width: 0; background: #2563eb; border-radius: 3px; transition: width .3s; }
#result { display: none; }
.row { display: flex; justify-content: space-between; font-size: 13px; padding: 6px 0; border-bottom: 1px solid #f4f4f5; }
.row:last-child { border: none; }
.row span:first-child { color: #71717a; }
.row span:last-child { font-weight: 500; }
.saved { color: #16a34a !important; }
a.dl {
  display: block; text-align: center; margin-top: 14px; padding: 11px;
  background: #2563eb; color: #fff; border-radius: 8px; text-decoration: none;
  font-size: 14px; font-weight: 600;
}
a.dl:hover { background: #1d4ed8; }
.err { font-size: 13px; color: #dc2626; margin-top: 8px; display: none; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Audio Compression</h1>
  <p class="sub">Compress mp3, wav, flac, ogg, aac, m4a, aiff, opus &mdash; up to 50 MB</p>

  <div class="box">
    <label class="lbl">Select audio file</label>
    <input type="file" id="fi" accept=".mp3,.wav,.ogg,.flac,.aac,.aiff,.m4a,.opus,audio/*"/>
    <div class="err" id="err"></div>
    <button id="btn" onclick="go()" style="margin-top:14px">Compress</button>
    <div id="status"></div>
    <div class="bar-wrap" style="display:none" id="bw"><div class="bar" id="bar"></div></div>
  </div>

  <div class="box" id="result">
    <div id="rows"></div>
    <a class="dl" id="dl" href="#">Download compressed file</a>
    <button onclick="reset()" style="margin-top:8px;background:#f4f4f5;color:#3f3f46;border:1px solid #e4e4e7">Compress another</button>
  </div>
</div>

<script>
function go() {
  var fi = document.getElementById('fi');
  if (!fi.files || !fi.files[0]) { showErr('Please select a file first.'); return; }
  var f = fi.files[0];
  var ext = f.name.split('.').pop().toLowerCase();
  var ok = ['mp3','wav','ogg','flac','aac','aiff','m4a','opus'];
  if (!ok.includes(ext)) { showErr('Unsupported format: .' + ext); return; }

  hideErr();
  document.getElementById('btn').disabled = true;
  document.getElementById('result').style.display = 'none';
  setStatus('Uploading...', 10);

  var fd = new FormData();
  fd.append('file', f, f.name);

  fetch('/api/compress', { method: 'POST', body: fd })
    .then(function(r) {
      setStatus('Processing...', 60);
      return r.json();
    })
    .then(function(d) {
      if (d.error) { showErr(d.error); return; }
      setStatus('Done!', 100);
      setTimeout(function() { showResult(d); }, 300);
    })
    .catch(function(e) {
      showErr('Error: ' + e.message);
    });
}

function showResult(d) {
  var s = d.stats || {};
  document.getElementById('rows').innerHTML =
    row('Original size', d.orig_size) +
    row('Compressed size', d.comp_size) +
    row('Space saved', d.saved + '%', true) +
    row('Sample rate', (s.orig_sr||'?') + ' Hz \u2192 ' + (s.new_sr||'?') + ' Hz') +
    row('Channels', (s.orig_ch||'?') + ' \u2192 ' + (s.new_ch||'?') + ' (mono)') +
    row('Format', (s.format||'mp3').toUpperCase() + ' @ ' + (s.bitrate||'64k')) +
    row('Time', d.elapsed + 's');
  var a = document.getElementById('dl');
  a.href = '/api/download/' + d.token + '/' + encodeURIComponent(d.filename);
  a.download = d.filename;
  document.getElementById('result').style.display = 'block';
  document.getElementById('bw').style.display = 'none';
  document.getElementById('status').textContent = '';
}

function row(label, val, highlight) {
  return '<div class="row"><span>' + label + '</span><span' + (highlight ? ' class="saved"' : '') + '>' + val + '</span></div>';
}

function setStatus(msg, pct) {
  document.getElementById('status').textContent = msg;
  document.getElementById('bw').style.display = 'block';
  document.getElementById('bar').style.width = pct + '%';
}

function showErr(msg) {
  var e = document.getElementById('err');
  e.textContent = msg;
  e.style.display = 'block';
  document.getElementById('btn').disabled = false;
  document.getElementById('bw').style.display = 'none';
  document.getElementById('status').textContent = '';
}

function hideErr() { document.getElementById('err').style.display = 'none'; }

function reset() {
  document.getElementById('fi').value = '';
  document.getElementById('result').style.display = 'none';
  document.getElementById('btn').disabled = false;
  document.getElementById('status').textContent = '';
  document.getElementById('bw').style.display = 'none';
  document.getElementById('bar').style.width = '0';
  hideErr();
}
</script>
</body>
</html>"""

# ── ENTRY ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Audio Compression")
    print("  -----------------")
    if not PYDUB_OK:
        print("  WARNING: pydub not found!")
        print("  Run: pip install pydub numpy")
    else:
        print("  pydub OK. Ready.")
    print("\n  http://localhost:5000\n")
    app.run(debug=True, port=5000, host="0.0.0.0")