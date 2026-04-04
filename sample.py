"""
audio_compressor.py
────────────────────────────────────────────────────────────
A proper audio compression tool using industry-standard codecs.

MODES
  1. LOSSLESS  → FLAC   (identical audio, ~40-60% smaller than WAV)
  2. LOSSY HQ  → Opus   (transparent quality, ~80-90% smaller than WAV)
  3. LOSSY MP3 → MP3    (universal compatibility, ~70-85% smaller than WAV)

REQUIREMENTS
  pip install pydub soundfile numpy
  Also needs ffmpeg installed:
    Windows : https://ffmpeg.org/download.html  (add to PATH)
    Ubuntu  : sudo apt install ffmpeg
    macOS   : brew install ffmpeg

USAGE
  python audio_compressor.py                     # uses CONFIG below
  python audio_compressor.py my_audio.wav        # compress a specific file
  python audio_compressor.py my_audio.wav lossy  # lossy mode
"""

import os
import sys
import time
import subprocess
import shutil

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

UPLOAD_FOLDER   = r"D:\UPLOAD"
UPLOAD_FILENAME = "123.mp3"

# Compression mode: "lossless", "lossy_hq", or "lossy_mp3"
DEFAULT_MODE = "lossy_hq"

# Quality settings (you can tune these)
OPUS_BITRATE    = "64k"    # 48k=small/good, 64k=great, 96k=transparent
MP3_BITRATE     = "128k"   # 96k=small, 128k=good, 192k=high, 320k=best
FLAC_COMPRESSION = 8       # 0=fastest/larger, 8=slowest/smallest (0-8)

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def check_ffmpeg():
    """Verify ffmpeg is installed and accessible."""
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found in PATH.")
        print("  Windows : Download from https://ffmpeg.org/download.html and add to PATH")
        print("  Ubuntu  : sudo apt install ffmpeg")
        print("  macOS   : brew install ffmpeg")
        sys.exit(1)

def file_size_str(path):
    """Return human-readable file size."""
    size = os.path.getsize(path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"

def print_result(input_path, output_path, elapsed):
    """Print a clean compression summary."""
    in_size  = os.path.getsize(input_path)
    out_size = os.path.getsize(output_path)
    saved    = (1 - out_size / in_size) * 100

    print()
    print("─" * 50)
    print(f"  Input  : {os.path.basename(input_path):<30} {file_size_str(input_path):>8}")
    print(f"  Output : {os.path.basename(output_path):<30} {file_size_str(output_path):>8}")
    print(f"  Saved  : {saved:.1f}%  |  Time: {elapsed:.1f}s")
    print("─" * 50)

    if saved < 0:
        print("  Note: Output is larger — your source is already compressed.")
        print("  Best practice: always start from a raw WAV file.")
    print()

# ─── CORE COMPRESSION FUNCTIONS ───────────────────────────────────────────────

def compress_lossless(input_path, output_path=None):
    """
    FLAC compression — mathematically lossless.
    Decompressed audio is bit-for-bit identical to the original.
    Typical savings vs WAV: 40-60%.
    Works best on: WAV, AIFF files.
    """
    if output_path is None:
        base       = os.path.splitext(input_path)[0]
        output_path = base + ".flac"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:a", "flac",
        "-compression_level", str(FLAC_COMPRESSION),
        output_path
    ]

    print(f"Compressing (FLAC lossless) → {os.path.basename(output_path)} ...")
    _run(cmd)
    return output_path


def compress_lossy_opus(input_path, output_path=None):
    """
    Opus compression — state-of-the-art lossy codec.
    At 64k it's perceptually transparent for most content.
    At 48k it's excellent for voice/podcast.
    Typical savings vs WAV: 80-90%.
    """
    if output_path is None:
        base        = os.path.splitext(input_path)[0]
        output_path = base + "_opus.ogg"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:a", "libopus",
        "-b:a", OPUS_BITRATE,
        "-vbr", "on",              # variable bitrate — better quality/size
        "-compression_level", "10",
        output_path
    ]

    print(f"Compressing (Opus {OPUS_BITRATE} VBR) → {os.path.basename(output_path)} ...")
    _run(cmd)
    return output_path


def compress_lossy_mp3(input_path, output_path=None):
    """
    MP3 compression — universal compatibility.
    Plays everywhere; slightly less efficient than Opus at same bitrate.
    Typical savings vs WAV: 70-85%.
    """
    if output_path is None:
        base        = os.path.splitext(input_path)[0]
        output_path = base + "_compressed.mp3"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:a", "libmp3lame",
        "-b:a", MP3_BITRATE,
        "-q:a", "2",               # VBR quality 0-9 (2 = high, ~190kbps avg)
        output_path
    ]

    print(f"Compressing (MP3 {MP3_BITRATE}) → {os.path.basename(output_path)} ...")
    _run(cmd)
    return output_path


def decompress(input_path, output_path=None):
    """
    Decompress any audio format back to WAV (PCM).
    Works on FLAC, Opus, MP3, OGG, AAC, etc.
    """
    if output_path is None:
        base        = os.path.splitext(input_path)[0]
        output_path = base + "_decoded.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:a", "pcm_s16le",       # 16-bit PCM WAV — universally playable
        output_path
    ]

    print(f"Decompressing → {os.path.basename(output_path)} ...")
    _run(cmd)
    return output_path


def compress_batch(folder, mode="lossy_hq", extensions=(".wav", ".mp3", ".ogg", ".flac", ".aiff")):
    """
    Compress every audio file in a folder.
    Skips files that already look like compressed outputs.
    """
    compress_fn = _mode_fn(mode)
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(extensions)
        and not f.endswith(("_compressed.mp3", "_opus.ogg", ".flac"))
    ]

    if not files:
        print(f"No matching audio files found in {folder}")
        return

    print(f"Found {len(files)} file(s) to compress in {folder}\n")
    for fname in files:
        path = os.path.join(folder, fname)
        t0   = time.time()
        try:
            out = compress_fn(path)
            print_result(path, out, time.time() - t0)
        except Exception as e:
            print(f"  FAILED: {fname} — {e}\n")


# ─── PRIVATE ──────────────────────────────────────────────────────────────────

def _run(cmd):
    """Run an ffmpeg command, suppressing noisy output unless it fails."""
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        print("\nffmpeg error output:")
        print(result.stderr[-2000:])  # last 2000 chars of error
        raise RuntimeError(f"ffmpeg failed (code {result.returncode})")


def _mode_fn(mode):
    return {
        "lossless" : compress_lossless,
        "lossy_hq" : compress_lossy_opus,
        "lossy_mp3": compress_lossy_mp3,
    }.get(mode, compress_lossy_opus)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    check_ffmpeg()

    # Allow overriding via command line: script.py [file] [mode]
    input_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(UPLOAD_FOLDER, UPLOAD_FILENAME)
    mode       = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODE

    if not os.path.exists(input_path):
        print(f"ERROR: File not found — {input_path}")
        sys.exit(1)

    compress_fn = _mode_fn(mode)

    t0         = time.time()
    output     = compress_fn(input_path)
    elapsed    = time.time() - t0

    print_result(input_path, output, elapsed)

    # Also decode back to WAV so you can verify playback
    wav_out = decompress(output)
    print(f"Decoded WAV for playback → {wav_out}")


if __name__ == "__main__":
    main()