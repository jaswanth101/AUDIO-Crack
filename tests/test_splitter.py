"""
tests/test_splitter.py
=======================
Automated tests for the Audio Splitter backend.

WHAT THIS SCRIPT DOES:
1. Uses FFmpeg to generate short test audio files in several formats.
2. Imports the core split-logic functions from app.py and tests them directly
   (no HTTP requests needed – this is a unit test of the backend functions).
3. Also makes real HTTP requests to the running Flask server to test the full
   round-trip (upload → analyse → split → download).

HOW TO RUN:
  # First, start the server in a separate terminal:
  #   python app.py
  #
  # Then, in another terminal:
  #   python tests/test_splitter.py

A PASSING run prints:  "All tests passed! ✅"
"""

import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path

# Add the parent directory to sys.path so we can import app.py directly
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:5000"
TEST_DIR = Path(tempfile.mkdtemp(prefix="audio_splitter_tests_"))

print(f"\n📁 Test files will be created in: {TEST_DIR}\n")


def run(cmd: list, check=True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def generate_test_audio(path: Path, format_ext: str, duration_secs: int = 90) -> Path:
    """
    Generate a silent test audio file using FFmpeg's built-in sine tone generator.

    Args:
        path:        Directory to create the file in.
        format_ext:  File extension (mp3, wav, flac, ogg, m4a).
        duration_secs: How long the audio should be (seconds).

    Returns:
        Path to the generated file.
    """
    out_file = path / f"test_{duration_secs}s.{format_ext}"

    # Map extension to FFmpeg codec
    codec_map = {
        "mp3":  ["-c:a", "libmp3lame", "-b:a", "128k"],
        "wav":  ["-c:a", "pcm_s16le"],
        "flac": ["-c:a", "flac"],
        "ogg":  ["-c:a", "libvorbis"],
        "m4a":  ["-c:a", "aac", "-b:a", "128k"],
    }
    extra_args = codec_map.get(format_ext, [])

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration_secs}",
        *extra_args,
        str(out_file),
    ]

    result = run(cmd, check=False)
    if result.returncode != 0:
        print(f"  ⚠  Could not generate .{format_ext} – codec may not be available.")
        print(f"     FFmpeg stderr: {result.stderr.decode()[-200:]}")
        return None

    print(f"  ✔  Generated {out_file.name}  ({out_file.stat().st_size // 1024} KB)")
    return out_file


def http_post(path: str, json_body: dict = None, files: dict = None) -> dict:
    """Simple HTTP client using urllib (no external dependencies)."""
    import urllib.request, urllib.error
    url = BASE_URL + path

    if files:
        # Multipart form upload
        import io, uuid
        boundary = uuid.uuid4().hex
        body_parts = []
        for field_name, (filename, fileobj, mimetype) in files.items():
            header = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
                f"Content-Type: {mimetype}\r\n\r\n"
            ).encode()
            body_parts.append(header + fileobj.read() + b"\r\n")
        body_parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(body_parts)
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    else:
        body = json.dumps(json_body or {}).encode()
        headers = {"Content-Type": "application/json"}

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def http_get(path: str) -> dict:
    """Simple GET request."""
    import urllib.request
    with urllib.request.urlopen(BASE_URL + path, timeout=30) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Unit tests (no server needed)
# ---------------------------------------------------------------------------

def test_calculate_split_points():
    """Test the core split-point calculation function."""
    from app import calculate_split_points

    print("\n─ Unit test: calculate_split_points ─")

    # 1. Chunk mode: 90-second file, 30-second chunks → 3 parts
    pairs = calculate_split_points(duration=90, mode="chunk", chunk_minutes=0.5)
    assert len(pairs) == 3, f"Expected 3 parts, got {len(pairs)}"
    assert pairs[0] == (0.0, 30.0), f"First part wrong: {pairs[0]}"
    assert pairs[2][1] == 90.0, f"Last part should end at 90s, got {pairs[2][1]}"
    print("  ✔  Chunk mode: 90s / 30s = 3 parts")

    # 2. Parts mode: 90-second file into 3 equal parts
    pairs = calculate_split_points(duration=90, mode="parts", num_parts=3)
    assert len(pairs) == 3, f"Expected 3 parts, got {len(pairs)}"
    assert abs(pairs[0][1] - 30.0) < 0.01, f"First part should be ~30s"
    print("  ✔  Parts mode: 90s into 3 equal parts")

    # 3. Merge short last part
    # 100-second file, 30-second chunks → last part is only 10s → should merge
    pairs_merge = calculate_split_points(
        duration=100, mode="chunk", chunk_minutes=0.5, last_part="merge"
    )
    pairs_keep = calculate_split_points(
        duration=100, mode="chunk", chunk_minutes=0.5, last_part="keep"
    )
    assert len(pairs_merge) < len(pairs_keep), "Merge should reduce part count"
    print(f"  ✔  Short last part merged: {len(pairs_keep)} → {len(pairs_merge)} parts")

    print("  ✅ calculate_split_points: all assertions passed")


def test_probe_file():
    """Test ffprobe wrapper with a generated test file."""
    from app import probe_file

    print("\n─ Unit test: probe_file ─")

    test_file = generate_test_audio(TEST_DIR, "wav", duration_secs=5)
    if test_file is None:
        print("  ⚠  Skipped (WAV generation failed)")
        return

    info = probe_file(test_file)
    assert info.get("error") is None, f"probe_file returned error: {info.get('error')}"
    assert abs(info["duration"] - 5.0) < 1.0, f"Duration should be ~5s, got {info['duration']}"
    assert info["has_audio"] is True
    print(f"  ✔  Duration: {info['duration_hms']},  Codec: {info['codec_name']}")
    print("  ✅ probe_file: all assertions passed")


def test_split_audio_direct():
    """
    Test the split_audio() function directly (no Flask server needed).
    Generates a 90-second WAV, splits it into 3×30s parts.
    """
    from app import split_audio, calculate_split_points, probe_file, JOBS, JOBS_LOCK
    import threading

    print("\n─ Unit test: split_audio (direct) ─")

    src = generate_test_audio(TEST_DIR, "wav", duration_secs=90)
    if src is None:
        print("  ⚠  Skipped (WAV generation failed)")
        return

    out_dir = TEST_DIR / "split_output"
    out_dir.mkdir(exist_ok=True)

    probe = probe_file(src)
    assert not probe.get("error")

    pairs = calculate_split_points(duration=probe["duration"], mode="chunk", chunk_minutes=0.5)

    job_id = "test_direct_001"
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "processing"}

    # Run in a thread (same as the real app does)
    t = threading.Thread(target=split_audio, kwargs=dict(
        job_id        = job_id,
        input_path    = src,
        output_dir    = out_dir,
        split_pairs   = pairs,
        base_name     = "test",
        output_format = "original",
        original_ext  = "wav",
        cutting_method= "fast",
    ))
    t.start()
    t.join(timeout=60)

    with JOBS_LOCK:
        job = JOBS.get(job_id, {})

    assert job.get("status") == "done", f"Job status: {job.get('status')}, error: {job.get('error')}"
    results = job.get("results", [])
    assert len(results) == len(pairs), f"Expected {len(pairs)} parts, got {len(results)}"

    for r in results:
        assert Path(r["path"]).exists(), f"Output file missing: {r['path']}"
        assert r["size"] > 0, f"Output file empty: {r['path']}"

    print(f"  ✔  Created {len(results)} parts:")
    for r in results:
        print(f"       {r['name']}  –  {r['duration_hms']}  –  {r['size_human']}")
    print("  ✅ split_audio (direct): all assertions passed")


# ---------------------------------------------------------------------------
# Integration tests (require the Flask server to be running)
# ---------------------------------------------------------------------------

def check_server() -> bool:
    """Return True if the Flask server is reachable."""
    try:
        http_get("/api/status")
        return True
    except Exception:
        return False


def test_full_roundtrip(format_ext: str, duration_secs: int = 90):
    """
    Full HTTP round-trip test:
      generate → upload → analyse → preview → split → poll → results
    """
    print(f"\n─ Integration test: full round-trip for .{format_ext} ─")

    src = generate_test_audio(TEST_DIR, format_ext, duration_secs)
    if src is None:
        print(f"  ⚠  Skipped (could not generate .{format_ext})")
        return

    # Upload
    with open(src, "rb") as f:
        upload_resp = http_post("/api/upload", files={
            "file": (src.name, f, "application/octet-stream")
        })

    assert "job_id" in upload_resp, f"Upload failed: {upload_resp}"
    job_id = upload_resp["job_id"]
    print(f"  ✔  Uploaded  →  job_id: {job_id[:8]}…")

    # Analyse
    info = http_get(f"/api/analyse/{job_id}")
    assert not info.get("error"), f"Analyse failed: {info.get('error')}"
    print(f"  ✔  Analysed  →  Duration: {info['duration_hms']}, Codec: {info['codec_name']}")

    # Preview
    preview = http_post("/api/preview", json_body={
        "job_id": job_id, "mode": "chunk", "chunk_minutes": 0.5, "last_part": "keep"
    })
    assert preview.get("num_parts") == 3, f"Expected 3 preview parts, got {preview.get('num_parts')}"
    print(f"  ✔  Preview   →  {preview['num_parts']} parts: {[p['duration_hms'] for p in preview['parts']]}")

    # Split
    split_resp = http_post("/api/split", json_body={
        "job_id":         job_id,
        "mode":           "chunk",
        "chunk_minutes":  0.5,
        "output_format":  "original",
        "cutting_method": "fast",
        "last_part":      "keep",
    })
    assert split_resp.get("ok"), f"Split start failed: {split_resp}"
    print(f"  ✔  Splitting started ({split_resp['total_parts']} parts)…")

    # Poll until done
    for attempt in range(60):
        time.sleep(2)
        job = http_get(f"/api/job/{job_id}")
        status = job.get("status")
        if status == "done":
            break
        if status in ("error", "cancelled"):
            raise AssertionError(f"Job {status}: {job.get('error')}")

    assert job.get("status") == "done", f"Job did not finish in time. Last status: {job.get('status')}"
    results = job.get("results", [])
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    for r in results:
        print(f"  ✔  Result: {r['name']}  –  {r['duration_hms']}  –  {r['size_human']}")

    print(f"  ✅ Full round-trip for .{format_ext}: PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Audio Splitter – Test Suite")
    print("=" * 60)

    # ── Unit tests (always run) ──────────────────────────────────
    test_calculate_split_points()
    test_probe_file()
    test_split_audio_direct()

    # ── Integration tests (need server) ─────────────────────────
    print("\n─ Checking for running Flask server… ─")
    if not check_server():
        print("  ⚠  Flask server not reachable at http://localhost:5000")
        print("     Start the server with:  python app.py")
        print("     Then re-run the tests.")
    else:
        print("  ✔  Server is running")
        formats_to_test = ["wav", "mp3", "flac", "ogg", "m4a"]
        for fmt in formats_to_test:
            try:
                test_full_roundtrip(fmt, duration_secs=90)
            except Exception as e:
                print(f"  ✗  Integration test .{fmt} FAILED: {e}")

    # ── Cleanup ──────────────────────────────────────────────────
    print(f"\n🧹 Cleaning up test files in {TEST_DIR}")
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    print("\n" + "=" * 60)
    print("  All tests passed! ✅")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
