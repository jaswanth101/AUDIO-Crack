# -*- coding: utf-8 -*-
"""
Audio Splitter - Flask Backend
===============================
A local web app that splits large audio files into smaller parts using FFmpeg.
No database, no login, runs entirely on your own computer.

HOW IT WORKS (big picture):
1. User uploads a file → saved to uploads/<job_id>/original.<ext>
2. ffprobe inspects the file and returns metadata (duration, codec, etc.)
3. User picks split settings → we calculate the cut points
4. FFmpeg runs in a background thread, splitting the file
5. User downloads individual parts or a ZIP of all parts
6. Temporary job folders are auto-deleted after 24 hours
"""

import sys

# Make stdout/stderr handle UTF-8 on Windows (avoids UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os

import re
import json
import math
import uuid
import time
import shutil
import zipfile
import threading
import subprocess
import unicodedata
from pathlib import Path
from datetime import datetime, timedelta

from flask import (
    Flask, request, jsonify, send_file,
    send_from_directory, render_template
)
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Where all job folders live
UPLOAD_ROOT = Path("uploads")
UPLOAD_ROOT.mkdir(exist_ok=True)

# Maximum upload size – 4 GB (we stream anyway, this is a safety cap)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024

# Supported input extensions (we double-check with ffprobe)
SUPPORTED_EXTENSIONS = {
    "mp3", "wav", "m4a", "aac", "flac", "ogg", "opus",
    "wma", "aiff", "aif", "mp4", "mov", "mkv", "webm",
}

# Output format → FFmpeg codec mapping
OUTPUT_CODECS = {
    "mp3":  {"codec": "libmp3lame", "ext": "mp3"},
    "wav":  {"codec": "pcm_s16le",  "ext": "wav"},
    "flac": {"codec": "flac",       "ext": "flac"},
    "m4a":  {"codec": "aac",        "ext": "m4a"},
    "ogg":  {"codec": "libvorbis",  "ext": "ogg"},
}

# In-memory job store: job_id → dict with status, progress, errors, etc.
# Keyed by job_id (string UUID).  Thread-safe writes via a lock.
JOBS: dict = {}
JOBS_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------

def check_ffmpeg() -> dict:
    """
    Verify that ffmpeg and ffprobe are available on PATH.
    Returns {"ok": True} or {"ok": False, "message": "..."}.
    """
    errors = []
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run(
                [tool, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except FileNotFoundError:
            errors.append(tool)
        except subprocess.CalledProcessError:
            pass  # version flag worked, binary exists

    if errors:
        missing = " and ".join(errors)
        msg = (
            f"Could not find: {missing}\n\n"
            "INSTALL INSTRUCTIONS\n"
            "--------------------\n"
            "Windows : https://www.gyan.dev/ffmpeg/builds/  (add to PATH)\n"
            "macOS   : brew install ffmpeg\n"
            "Linux   : sudo apt install ffmpeg   (Debian/Ubuntu)\n"
            "          sudo dnf install ffmpeg   (Fedora)\n"
        )
        return {"ok": False, "message": msg}
    return {"ok": True, "message": ""}


FFMPEG_STATUS = check_ffmpeg()  # checked once at startup

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """
    Make a filename safe: strip unicode accents, remove special characters,
    collapse spaces, truncate to 200 chars.
    """
    # Normalise unicode → ASCII equivalents where possible
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    # Keep letters, digits, dots, dashes, underscores
    name = re.sub(r"[^\w.\-]", "_", name)
    # Collapse multiple underscores/spaces
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:200] or "audio"


def seconds_to_hms(seconds: float) -> str:
    """Convert a number of seconds to HH:MM:SS string."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def human_size(nbytes: int) -> str:
    """Convert bytes to a human-readable string like '45.2 MB'."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def job_dir(job_id: str) -> Path:
    """Return the directory path for a job."""
    return UPLOAD_ROOT / job_id


def update_job(job_id: str, **kwargs):
    """Thread-safe update of a job's state dictionary."""
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kwargs)


# ---------------------------------------------------------------------------
# FFprobe – analyse audio file
# ---------------------------------------------------------------------------

def probe_file(filepath: Path) -> dict:
    """
    Run ffprobe on a file and return a dict with:
      duration, sample_rate, bitrate, channels, codec_name,
      format_name, file_size, has_audio, error
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(filepath),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return {"error": "ffprobe timed out analysing the file."}
    except FileNotFoundError:
        return {"error": "ffprobe not found."}

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        return {"error": f"ffprobe failed: {stderr[:300]}"}

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "Could not parse ffprobe output."}

    # Find the first audio stream
    audio_stream = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio":
            audio_stream = stream
            break

    if audio_stream is None:
        return {"error": "No audio stream found in this file."}

    fmt = info.get("format", {})
    duration = float(fmt.get("duration") or audio_stream.get("duration", 0))
    bitrate_raw = int(fmt.get("bit_rate") or audio_stream.get("bit_rate", 0))

    return {
        "duration":      duration,
        "duration_hms":  seconds_to_hms(duration),
        "sample_rate":   int(audio_stream.get("sample_rate", 0)),
        "bitrate":       bitrate_raw,
        "bitrate_kbps":  round(bitrate_raw / 1000),
        "channels":      int(audio_stream.get("channels", 0)),
        "codec_name":    audio_stream.get("codec_name", "unknown"),
        "format_name":   fmt.get("format_long_name", fmt.get("format_name", "unknown")),
        "file_size":     filepath.stat().st_size,
        "file_size_human": human_size(filepath.stat().st_size),
        "has_audio":     True,
        "error":         None,
    }


# ---------------------------------------------------------------------------
# Silence detection helper
# ---------------------------------------------------------------------------

def find_silence_points(filepath: Path, noise_db: float = -40.0,
                         min_silence: float = 0.5) -> list:
    """
    Use ffmpeg silencedetect filter to find all silence intervals.
    Returns a list of (start, end) tuples in seconds.
    """
    cmd = [
        "ffmpeg",
        "-i", str(filepath),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f", "null",
        "-",
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return []

    output = result.stderr.decode("utf-8", errors="replace")
    silence_start_times = [
        float(m) for m in re.findall(r"silence_start: ([\d.]+)", output)
    ]
    silence_end_times = [
        float(m) for m in re.findall(r"silence_end: ([\d.]+)", output)
    ]

    pairs = []
    for s, e in zip(silence_start_times, silence_end_times):
        pairs.append((s, e))
    return pairs


def snap_to_silence(target_time: float, silence_pairs: list,
                     window: float = 30.0) -> float:
    """
    Given a target cut time, find the nearest silence midpoint within ±window
    seconds.  Returns the adjusted time (or the original if no silence found).
    """
    best_time = target_time
    best_dist = float("inf")
    for (s, e) in silence_pairs:
        midpoint = (s + e) / 2.0
        dist = abs(midpoint - target_time)
        if dist <= window and dist < best_dist:
            best_dist = dist
            best_time = midpoint
    return best_time


# ---------------------------------------------------------------------------
# Core split logic (also usable as a standalone CLI function – Step 1)
# ---------------------------------------------------------------------------

def calculate_split_points(
    duration: float,
    mode: str,          # "chunk" | "parts"
    chunk_minutes: float = 20.0,
    num_parts: int = 4,
    smart_cut: bool = False,
    silence_pairs: list = None,
    last_part: str = "keep",   # "keep" | "merge"
) -> list:
    """
    Calculate the start/end times (in seconds) for each output part.

    Returns a list of (start, end) float tuples.
    """
    if silence_pairs is None:
        silence_pairs = []

    if mode == "chunk":
        chunk_secs = chunk_minutes * 60.0
        raw_boundaries = []
        t = chunk_secs
        while t < duration:
            raw_boundaries.append(t)
            t += chunk_secs
    else:  # "parts"
        chunk_secs = duration / num_parts
        raw_boundaries = [chunk_secs * i for i in range(1, num_parts)]

    # Optionally snap each boundary to nearest silence
    if smart_cut and silence_pairs:
        raw_boundaries = [
            snap_to_silence(b, silence_pairs) for b in raw_boundaries
        ]

    # Build (start, end) pairs
    boundaries = sorted(set(raw_boundaries))
    starts = [0.0] + boundaries
    ends   = boundaries + [duration]
    pairs  = list(zip(starts, ends))

    # Handle short last part: merge into previous
    if last_part == "merge" and len(pairs) >= 2:
        last_dur  = pairs[-1][1] - pairs[-1][0]
        total_dur = ends[-1]
        # If the last part is less than 50 % of a normal chunk, merge it
        typical = total_dur / len(pairs)
        if last_dur < typical * 0.5:
            pairs = pairs[:-2] + [(pairs[-2][0], pairs[-1][1])]

    return pairs


def split_audio(
    job_id: str,
    input_path: Path,
    output_dir: Path,
    split_pairs: list,
    base_name: str,
    output_format: str = "original",
    original_ext: str = "mp3",
    cutting_method: str = "fast",  # "fast" | "precise"
    bitrate: str = "192k",
    sample_rate: str = "",
    probe_info: dict = None,
):
    """
    Split *input_path* according to *split_pairs* using FFmpeg.

    This function runs in a background thread.  It updates JOBS[job_id]
    with progress information so the frontend can poll it.
    """
    total_parts = len(split_pairs)
    results = []
    start_time = time.time()

    # Determine output extension
    if output_format == "original":
        ext = original_ext
        codec_args = ["-c:a", "copy"] if cutting_method == "fast" else []
    else:
        info = OUTPUT_CODECS.get(output_format, {"codec": "copy", "ext": output_format})
        ext  = info["ext"]
        codec_args = ["-c:a", info["codec"]]

    for idx, (t_start, t_end) in enumerate(split_pairs):
        # Check if the job was cancelled
        with JOBS_LOCK:
            if JOBS.get(job_id, {}).get("status") == "cancelled":
                return

        part_num   = idx + 1
        part_name  = f"{base_name}_part_{part_num:02d}.{ext}"
        part_path  = output_dir / part_name
        duration_s = t_end - t_start

        update_job(job_id,
                   current_part=part_num,
                   total_parts=total_parts,
                   current_part_name=part_name,
                   status="processing")

        # Build FFmpeg command
        cmd = ["ffmpeg", "-y"]

        if cutting_method == "fast":
            # Seek BEFORE -i for faster seeking (keyframe-aligned)
            cmd += ["-ss", str(t_start), "-t", str(duration_s)]
            cmd += ["-i", str(input_path)]
            cmd += ["-c:a", "copy"]
        else:
            # Precise: decode the whole stream up to the cut point
            cmd += ["-i", str(input_path)]
            cmd += ["-ss", str(t_start), "-t", str(duration_s)]
            cmd += codec_args if codec_args else ["-c:a", "copy"]

        # Bitrate / sample rate for re-encoded formats
        if output_format != "original" and codec_args != ["-c:a", "copy"]:
            if bitrate and output_format in ("mp3", "m4a", "ogg"):
                cmd += ["-b:a", bitrate]
            if sample_rate:
                cmd += ["-ar", str(sample_rate)]

        # Copy metadata tags
        cmd += ["-map_metadata", "0"]
        # Audio only (drop video stream if input is a video file)
        cmd += ["-vn"]
        cmd += [str(part_path)]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Store process handle so we can cancel it
            update_job(job_id, _ffmpeg_proc=proc)
            stdout, stderr = proc.communicate()

            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="replace")[-400:]
                update_job(job_id,
                           status="error",
                           error=f"FFmpeg failed on part {part_num}: {err_msg}")
                return
        except Exception as exc:
            update_job(job_id,
                       status="error",
                       error=f"Unexpected error on part {part_num}: {exc}")
            return

        # Record result for this part
        size = part_path.stat().st_size if part_path.exists() else 0
        results.append({
            "name":          part_name,
            "path":          str(part_path),
            "duration":      duration_s,
            "duration_hms":  seconds_to_hms(duration_s),
            "size":          size,
            "size_human":    human_size(size),
        })

        # Update progress percentage and ETA
        elapsed  = time.time() - start_time
        progress = part_num / total_parts
        eta      = (elapsed / progress) * (1 - progress) if progress > 0 else 0
        update_job(job_id,
                   progress=round(progress * 100),
                   eta_seconds=round(eta))

    # All done!
    update_job(job_id, status="done", results=results, progress=100)


# ---------------------------------------------------------------------------
# Auto-cleanup: delete job folders older than 24 hours
# ---------------------------------------------------------------------------

def cleanup_old_jobs():
    """Delete any job folders under uploads/ that are older than 24 hours."""
    cutoff = datetime.now() - timedelta(hours=24)
    if not UPLOAD_ROOT.exists():
        return
    for d in UPLOAD_ROOT.iterdir():
        if d.is_dir():
            mtime = datetime.fromtimestamp(d.stat().st_mtime)
            if mtime < cutoff:
                try:
                    shutil.rmtree(d)
                    print(f"[cleanup] Deleted old job: {d.name}")
                except Exception as e:
                    print(f"[cleanup] Could not delete {d}: {e}")


# Run cleanup once at startup
cleanup_old_jobs()


# ---------------------------------------------------------------------------
# Flask routes – API
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the single-page frontend."""
    return render_template("index.html", ffmpeg_ok=FFMPEG_STATUS["ok"],
                           ffmpeg_msg=FFMPEG_STATUS["message"])


@app.route("/api/status")
def api_status():
    """Return whether ffmpeg is available (called by the frontend on load)."""
    return jsonify(FFMPEG_STATUS)


# ── Upload ──────────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Stream-upload a file to disk in chunks (never loads whole file in RAM).
    Returns a job_id that identifies this upload session.
    """
    if not FFMPEG_STATUS["ok"]:
        return jsonify({"error": FFMPEG_STATUS["message"]}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400

    # Check extension (we also verify with ffprobe after saving)
    original_name = f.filename
    ext = original_name.rsplit(".", 1)[-1].lower() if "." in original_name else ""
    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({
            "error": (
                f"Unsupported file type: .{ext}\n"
                "Supported: mp3, wav, m4a, aac, flac, ogg, opus, wma, "
                "aiff, mp4, mov, mkv, webm"
            )
        }), 415

    # Create a unique folder for this job
    job_id   = str(uuid.uuid4())
    dest_dir = job_dir(job_id)
    dest_dir.mkdir(parents=True)

    safe_name    = sanitize_filename(original_name)
    dest_path    = dest_dir / f"original.{ext}"

    # Stream to disk in 1 MB chunks
    chunk_size = 1024 * 1024
    try:
        with open(dest_path, "wb") as out:
            while True:
                chunk = f.stream.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
    except OSError as e:
        shutil.rmtree(dest_dir, ignore_errors=True)
        return jsonify({"error": f"Disk write failed: {e}"}), 500

    # Register job
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status":        "uploaded",
            "original_name": original_name,
            "safe_name":     safe_name,
            "ext":           ext,
            "input_path":    str(dest_path),
            "created_at":    time.time(),
        }

    return jsonify({"job_id": job_id, "original_name": original_name})


# ── Analyse ─────────────────────────────────────────────────────────────────

@app.route("/api/analyse/<job_id>")
def api_analyse(job_id):
    """Run ffprobe on the uploaded file and return metadata."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    input_path = Path(job["input_path"])
    info = probe_file(input_path)

    if info.get("error"):
        return jsonify({"error": info["error"]}), 422

    # Store probe info in the job for later use
    update_job(job_id, probe_info=info, status="analysed")

    return jsonify({**info, "original_name": job["original_name"]})


# ── Preview split ────────────────────────────────────────────────────────────

@app.route("/api/preview", methods=["POST"])
def api_preview():
    """
    Calculate and return what the split will look like, without actually
    running FFmpeg.  Used to show 'This will create N parts: ...' preview.
    """
    data   = request.get_json(force=True)
    job_id = data.get("job_id")

    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or not job.get("probe_info"):
        return jsonify({"error": "Analyse the file first."}), 400

    probe_info  = job["probe_info"]
    duration    = probe_info["duration"]

    mode          = data.get("mode", "chunk")   # "chunk" | "parts"
    chunk_minutes = float(data.get("chunk_minutes", 20))
    num_parts     = int(data.get("num_parts", 4))
    last_part     = data.get("last_part", "keep")

    pairs = calculate_split_points(
        duration       = duration,
        mode           = mode,
        chunk_minutes  = chunk_minutes,
        num_parts      = num_parts,
        smart_cut      = False,   # preview without silence scan
        last_part      = last_part,
    )

    parts_preview = [
        {"index": i + 1, "duration_hms": seconds_to_hms(e - s),
         "duration_s": round(e - s, 2)}
        for i, (s, e) in enumerate(pairs)
    ]

    return jsonify({
        "num_parts":    len(pairs),
        "parts":        parts_preview,
        "total_duration_hms": seconds_to_hms(duration),
    })


# ── Start split ──────────────────────────────────────────────────────────────

@app.route("/api/split", methods=["POST"])
def api_split():
    """
    Kick off a background split job.
    Body (JSON): job_id, mode, chunk_minutes, num_parts, output_format,
                 cutting_method, bitrate, sample_rate, smart_cut, last_part
    """
    data   = request.get_json(force=True)
    job_id = data.get("job_id")

    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
    if job.get("status") == "processing":
        return jsonify({"error": "Job is already running."}), 409

    probe_info = job.get("probe_info")
    if not probe_info:
        return jsonify({"error": "Analyse the file first."}), 400

    duration      = probe_info["duration"]
    mode          = data.get("mode", "chunk")
    chunk_minutes = float(data.get("chunk_minutes", 20))
    num_parts     = int(data.get("num_parts", 4))
    output_format = data.get("output_format", "original")
    cutting_method= data.get("cutting_method", "fast")
    bitrate       = data.get("bitrate", "192k")
    sample_rate   = data.get("sample_rate", "")
    smart_cut     = bool(data.get("smart_cut", False))
    last_part     = data.get("last_part", "keep")

    # Silence detection (can take a while for long files)
    silence_pairs = []
    if smart_cut:
        input_path = Path(job["input_path"])
        silence_pairs = find_silence_points(input_path)

    split_pairs = calculate_split_points(
        duration      = duration,
        mode          = mode,
        chunk_minutes = chunk_minutes,
        num_parts     = num_parts,
        smart_cut     = smart_cut,
        silence_pairs = silence_pairs,
        last_part     = last_part,
    )

    if not split_pairs:
        return jsonify({"error": "Could not calculate any split points."}), 400

    # Create output sub-folder inside the job dir
    output_dir = job_dir(job_id) / "parts"
    output_dir.mkdir(exist_ok=True)

    safe_stem = Path(sanitize_filename(job["original_name"])).stem

    update_job(job_id,
               status="processing",
               progress=0,
               current_part=0,
               total_parts=len(split_pairs),
               eta_seconds=0,
               results=[],
               error=None)

    # Launch background thread
    t = threading.Thread(
        target=split_audio,
        kwargs=dict(
            job_id        = job_id,
            input_path    = Path(job["input_path"]),
            output_dir    = output_dir,
            split_pairs   = split_pairs,
            base_name     = safe_stem,
            output_format = output_format,
            original_ext  = job["ext"],
            cutting_method= cutting_method,
            bitrate       = bitrate,
            sample_rate   = sample_rate,
            probe_info    = probe_info,
        ),
        daemon=True,
    )
    t.start()

    return jsonify({"ok": True, "total_parts": len(split_pairs)})


# ── Poll status ──────────────────────────────────────────────────────────────

@app.route("/api/job/<job_id>")
def api_job_status(job_id):
    """Return the current state of a job (polled every second by the frontend)."""
    with JOBS_LOCK:
        job = dict(JOBS.get(job_id, {}))

    if not job:
        return jsonify({"error": "Job not found."}), 404

    # Don't expose the internal FFmpeg process object over JSON
    job.pop("_ffmpeg_proc", None)
    job.pop("probe_info",   None)   # large dict, not needed for polling

    return jsonify(job)


# ── Cancel ───────────────────────────────────────────────────────────────────

@app.route("/api/cancel/<job_id>", methods=["POST"])
def api_cancel(job_id):
    """Kill the running FFmpeg process and mark job as cancelled."""
    with JOBS_LOCK:
        job  = JOBS.get(job_id, {})
        proc = job.get("_ffmpeg_proc")

    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    update_job(job_id, status="cancelled")
    return jsonify({"ok": True})


# ── Download single part ─────────────────────────────────────────────────────

@app.route("/api/download/<job_id>/<part_name>")
def api_download(job_id, part_name):
    """Stream a single output file to the browser."""
    # Validate job
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    # Safety: never let path traversal escape the job dir
    safe_part  = secure_filename(part_name)
    parts_dir  = job_dir(job_id) / "parts"
    file_path  = parts_dir / safe_part

    if not file_path.exists():
        return jsonify({"error": "File not found."}), 404

    return send_file(file_path, as_attachment=True, download_name=safe_part)


# ── Download all as ZIP ──────────────────────────────────────────────────────

@app.route("/api/download_zip/<job_id>")
def api_download_zip(job_id):
    """Create a ZIP of all output parts and stream it."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    parts_dir  = job_dir(job_id) / "parts"
    zip_path   = job_dir(job_id) / "all_parts.zip"

    # (Re-)build the zip each time so it's always fresh
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(parts_dir.iterdir()):
            if f.is_file():
                zf.write(f, f.name)

    stem     = Path(sanitize_filename(job["original_name"])).stem
    zip_name = f"{stem}_parts.zip"

    return send_file(zip_path, as_attachment=True, download_name=zip_name)


# ── Stream audio for in-browser preview ─────────────────────────────────────

@app.route("/api/preview_audio/<job_id>/<part_name>")
def api_preview_audio(job_id, part_name):
    """Serve an audio part for the inline <audio> player (supports Range)."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    safe_part = secure_filename(part_name)
    file_path = job_dir(job_id) / "parts" / safe_part

    if not file_path.exists():
        return jsonify({"error": "File not found."}), 404

    return send_file(file_path)


# ── Delete / start over ──────────────────────────────────────────────────────

@app.route("/api/delete/<job_id>", methods=["POST"])
def api_delete(job_id):
    """Delete a job's files and remove it from the in-memory store."""
    with JOBS_LOCK:
        job  = JOBS.get(job_id, {})
        proc = job.get("_ffmpeg_proc")

    # Kill any running process first
    if proc and proc.poll() is None:
        proc.terminate()

    d = job_dir(job_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)

    with JOBS_LOCK:
        JOBS.pop(job_id, None)

    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  Audio Splitter – starting up")
    print("=" * 55)
    if not FFMPEG_STATUS["ok"]:
        print("\n[!] WARNING: FFmpeg not found!\n")
        print(FFMPEG_STATUS["message"])
    else:
        print("[OK] FFmpeg found - ready to split audio.")
    print("\n  Open your browser at:  http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
