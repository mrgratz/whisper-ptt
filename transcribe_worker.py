"""
Dictation transcription worker — one process per transcription.

Spawned by main.py (the hotkey stub). Loads faster-whisper + CUDA, prints a
ready line, waits for ONE job line on stdin:

    {"audio": "<path to .npy float32 mono 16 kHz>"}

transcribes it, prints {"ok": true, "text": "..."} on stdout and exits. EOF on
stdin before a job (the recording was discarded as too short) exits 0 without
loading anything further. Nothing survives the process: the model, the CUDA
context and CTranslate2's host buffers all go with it.

Why a process and not an in-daemon load/unload: the resident daemon held
~2.3 GB of commit while idle for 12 days and was the top-named consumer in both
commit-exhaustion events on 2026-09-01. A CUDA context never
fully unloads inside a live process; exiting is the only clean release.

Protocol notes for the stub side:
- stdout carries exactly two JSON lines (ready, result); stderr carries timing
  and diagnostics and is meant to be pointed at daemon.log.
- The job file is deleted by the worker after loading it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
DEFAULTS = {"model": "medium.en", "device": "cuda", "compute_type": "int8_float32"}


def _out(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    print(f"worker[{os.getpid()}]: {msg}", file=sys.stderr, flush=True)


# CTranslate2 on Windows expects cublas64_12.dll and cudnn*_9.dll on the DLL
# search path. The pip wheels nvidia-cublas-cu12 / nvidia-cudnn-cu12 install
# them into site-packages but do not register them. os.add_dll_directory alone
# is insufficient (CTranslate2 loads with LOAD_LIBRARY_SEARCH_DEFAULT_DIRS), so
# preload via ctypes and later LoadLibrary calls resolve to the loaded module.
def _preload_cuda_dlls() -> None:
    if sys.platform != "win32":
        return
    try:
        import nvidia  # type: ignore
    except ImportError:
        return
    import ctypes
    nvidia_roots = [Path(p) for p in nvidia.__path__]
    targets = [  # cudnn depends on cublas: load cublas first
        ("cublas/bin", "cublas64_12.dll"),
        ("cudnn/bin", "cudnn_ops64_9.dll"),
        ("cudnn/bin", "cudnn_cnn64_9.dll"),
        ("cudnn/bin", "cudnn_graph64_9.dll"),
        ("cudnn/bin", "cudnn64_9.dll"),
    ]
    for root in nvidia_roots:
        for sub in ("cublas/bin", "cudnn/bin", "cuda_nvrtc/bin", "cuda_runtime/bin"):
            p = root / sub
            if p.exists():
                os.add_dll_directory(str(p))
    for root in nvidia_roots:
        for sub, dll in targets:
            p = root / sub / dll
            if p.exists():
                try:
                    ctypes.CDLL(str(p))
                except OSError as e:
                    _log(f"preload skipped {dll}: {e}")


def _load_config(path: Path) -> dict:
    cfg = dict(DEFAULTS)
    if path.exists():
        try:
            cfg.update({k: v for k, v in json.loads(path.read_text()).items() if k in DEFAULTS})
        except Exception as e:
            _log(f"config unreadable ({e}); using defaults")
    return cfg


def main() -> int:
    if sys.stdout is not None:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--audio", help="run a single job on this .npy/.wav and exit (test mode; no stdin)")
    ap.add_argument("--idle-exit", type=float, default=0.0,
                    help="keep serving job lines until this many seconds pass with none (0 = one job, the default)")
    args = ap.parse_args()
    cfg = _load_config(Path(args.config))

    t0 = time.monotonic()
    _preload_cuda_dlls()
    # numpy is only used to load the audio array here; OpenBLAS would otherwise
    # commit ~31 MB per CPU thread of scratch at import.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import numpy as np
    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(cfg["model"], device=cfg["device"], compute_type=cfg["compute_type"])
        where = f"{cfg['device']}/{cfg['compute_type']}"
    except Exception as e:
        _log(f"failed on {cfg['device']} ({e}); falling back to cpu/int8")
        model = WhisperModel(cfg["model"], device="cpu", compute_type="int8")
        where = "cpu/int8"
    t_load = time.monotonic() - t0
    _log(f"model {cfg['model']} loaded on {where} in {t_load:.1f}s")
    _out({"ready": True, "load_sec": round(t_load, 2)})

    def run_job(audio_path: str, delete: bool) -> bool:
        try:
            if audio_path.lower().endswith(".wav"):
                import wave
                with wave.open(audio_path, "rb") as w:
                    raw = w.readframes(w.getnframes())
                    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    if w.getnchannels() > 1:
                        audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)
            else:
                audio = np.load(audio_path).astype(np.float32, copy=False).flatten()
            t1 = time.monotonic()
            segments, _info = model.transcribe(
                audio,
                language="en",
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            t_tx = time.monotonic() - t1
            _log(f"transcribed {len(audio) / 16000:.1f}s of audio in {t_tx:.1f}s")
            _out({"ok": True, "text": text, "load_sec": round(t_load, 2), "transcribe_sec": round(t_tx, 2)})
            return True
        except Exception as e:
            _out({"ok": False, "error": f"{type(e).__name__}: {e}"})
            return False
        finally:
            if delete:
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

    if args.audio:
        return 0 if run_job(args.audio, False) else 1

    # One job per process by default. EOF before a job = discarded recording; exit quietly.
    if not args.idle_exit:
        line = sys.stdin.readline() if sys.stdin is not None else ""
        if not line.strip():
            _log("no job (stdin EOF); exiting")
            return 0
        try:
            audio_path = json.loads(line)["audio"]
        except Exception as e:
            _out({"ok": False, "error": f"bad job line: {e}"})
            return 1
        return 0 if run_job(audio_path, True) else 1

    # --idle-exit N: keep the loaded model and serve job
    # lines until N seconds pass with none. The phone primes a worker on mic press, so the load
    # overlaps the talking and a second dictation inside the window is instant. Same exit-to-
    # release discipline as above: nothing stays resident past the window.
    import queue
    import threading
    jobs: "queue.Queue[str | None]" = queue.Queue()

    def reader() -> None:
        for ln in (sys.stdin or []):
            jobs.put(ln)
        jobs.put(None)

    threading.Thread(target=reader, daemon=True).start()
    while True:
        try:
            line = jobs.get(timeout=float(args.idle_exit))
        except queue.Empty:
            _log(f"idle for {args.idle_exit:.0f}s; exiting")
            return 0
        if line is None or not line.strip():
            _log("stdin EOF; exiting")
            return 0
        try:
            audio_path = json.loads(line)["audio"]
        except Exception as e:
            _out({"ok": False, "error": f"bad job line: {e}"})
            continue
        run_job(audio_path, True)


if __name__ == "__main__":
    sys.exit(main())
