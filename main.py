"""
Push-to-talk dictation daemon.

Hold the configured hotkey (default F13) → speak → release → transcript pastes
into the focused window. Tray icon for status + quit.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional


# CTranslate2 on Windows expects cublas64_12.dll and cudnn*_9.dll on the DLL
# search path. The pip wheels nvidia-cublas-cu12 / nvidia-cudnn-cu12 install
# them into site-packages but do not register them.
#
# os.add_dll_directory alone is insufficient: CTranslate2 loads DLLs with
# LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, which excludes user-added directories
# unless LOAD_LIBRARY_SEARCH_USER_DIRS is also set. We instead preload the
# DLLs via ctypes so they sit in the process module table, and any later
# LoadLibrary("cublas64_12.dll") resolves to the already-loaded module.
def _preload_cuda_dlls() -> None:
    if sys.platform != "win32":
        return
    try:
        import nvidia  # type: ignore
    except ImportError:
        return
    import ctypes
    nvidia_roots = [Path(p) for p in nvidia.__path__]
    # Order matters: cudnn depends on cublas, so load cublas first.
    targets = [
        ("cublas/bin", "cublas64_12.dll"),
        ("cudnn/bin", "cudnn_ops64_9.dll"),
        ("cudnn/bin", "cudnn_cnn64_9.dll"),
        ("cudnn/bin", "cudnn_graph64_9.dll"),
        ("cudnn/bin", "cudnn64_9.dll"),
    ]
    # Add directories first so transitively-loaded sibling DLLs resolve.
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
                    print(f"preload skipped {dll}: {e}", file=sys.stderr, flush=True)


_preload_cuda_dlls()


# Redirect to a log file when run under pythonw.exe (no console). With
# python.exe, sys.stdout is a real fd; with pythonw it's None.
def _setup_headless_logging() -> None:
    if sys.stdout is None:
        log_path = Path(__file__).parent / "daemon.log"
        f = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = f
        sys.stderr = f


_setup_headless_logging()

import keyboard
import numpy as np
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw
from pystray import Icon, Menu, MenuItem

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.json"
SAMPLE_RATE = 16000

DEFAULT_CONFIG = {
    "hotkey": "f13",
    "model": "medium.en",
    "device": "cuda",
    "compute_type": "int8_float32",
    "min_duration_sec": 0.3,
    "max_duration_sec": 60.0,
    "paste_delay_ms": 50,
    # Some hotkey remappers (e.g. SteelSeries GG) emit a sequence of taps while
    # held instead of a single continuous press. The release-debounce treats any
    # new keydown within this window as a continuation of the same hold.
    "release_debounce_ms": 250,
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        cfg = {**DEFAULT_CONFIG, **json.loads(CONFIG_PATH.read_text())}
    else:
        cfg = dict(DEFAULT_CONFIG)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    return cfg


# ── Audio recording ──────────────────────────────────────────────────────

class Recorder:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.stream: Optional[sd.InputStream] = None
        self.start_time: float = 0.0

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"audio status: {status}", file=sys.stderr)
        self.frames.append(indata.copy())

    def start(self) -> None:
        self.frames = []
        self.start_time = time.time()
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()

    def stop(self) -> tuple[np.ndarray, float]:
        if self.stream is None:
            return np.array([], dtype=np.float32), 0.0
        duration = time.time() - self.start_time
        self.stream.stop()
        self.stream.close()
        self.stream = None
        if not self.frames:
            return np.array([], dtype=np.float32), duration
        audio = np.concatenate(self.frames, axis=0).flatten()
        return audio, duration


# ── Transcription ────────────────────────────────────────────────────────

class Transcriber:
    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        print(f"loading whisper model: {model_name} ({device}/{compute_type})...", flush=True)
        try:
            self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as e:
            print(f"failed on {device}, falling back to cpu/int8: {e}", file=sys.stderr)
            self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        print("model loaded.", flush=True)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language="en",
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text


# ── Text injection ───────────────────────────────────────────────────────

def paste_text(text: str, paste_delay_ms: int) -> None:
    if not text:
        return
    prev_clip = ""
    try:
        prev_clip = pyperclip.paste()
    except Exception:
        pass
    pyperclip.copy(text)
    time.sleep(paste_delay_ms / 1000.0)
    keyboard.send("ctrl+v")
    # Restore clipboard after a beat so the paste lands first
    def _restore():
        time.sleep(0.5)
        try:
            pyperclip.copy(prev_clip)
        except Exception:
            pass
    threading.Thread(target=_restore, daemon=True).start()


# ── Tray icon ────────────────────────────────────────────────────────────

def make_icon(state: str) -> Image.Image:
    colors = {"idle": (110, 110, 110), "recording": (220, 60, 60), "working": (240, 180, 40)}
    color = colors.get(state, (110, 110, 110))
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((10, 10, 54, 54), fill=color)
    d.rectangle((28, 24, 36, 44), fill=(255, 255, 255))
    d.ellipse((24, 42, 40, 50), fill=(255, 255, 255))
    return img


# ── Main controller ──────────────────────────────────────────────────────

class Dictator:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.recorder = Recorder()
        self.transcriber = Transcriber(cfg["model"], cfg["device"], cfg["compute_type"])
        self.recording = False
        self.lock = threading.Lock()
        self.work_q: queue.Queue = queue.Queue()
        self.icon: Optional[Icon] = None
        self.release_timer: Optional[threading.Timer] = None

    def set_state(self, state: str) -> None:
        if self.icon is not None:
            self.icon.icon = make_icon(state)

    def on_press(self) -> None:
        # Cancel any pending release — this keydown is a continuation of the hold.
        with self.lock:
            if self.release_timer is not None:
                self.release_timer.cancel()
                self.release_timer = None
            if self.recording:
                return
            self.recording = True
        try:
            self.recorder.start()
            self.set_state("recording")
        except Exception as e:
            print(f"recorder failed to start: {e}", file=sys.stderr)
            with self.lock:
                self.recording = False

    def on_release(self) -> None:
        # Defer the actual stop. If a new keydown arrives within the debounce
        # window, on_press cancels this timer and the recording continues.
        with self.lock:
            if not self.recording:
                return
            if self.release_timer is not None:
                self.release_timer.cancel()
            delay = self.cfg["release_debounce_ms"] / 1000.0
            self.release_timer = threading.Timer(delay, self._finalize_release)
            self.release_timer.daemon = True
            self.release_timer.start()

    def _finalize_release(self) -> None:
        with self.lock:
            if not self.recording:
                return
            self.recording = False
            self.release_timer = None
        audio, duration = self.recorder.stop()
        if duration < self.cfg["min_duration_sec"]:
            self.set_state("idle")
            return
        if duration > self.cfg["max_duration_sec"]:
            audio = audio[: int(self.cfg["max_duration_sec"] * SAMPLE_RATE)]
        self.set_state("working")
        self.work_q.put(audio)

    def worker(self) -> None:
        while True:
            audio = self.work_q.get()
            if audio is None:
                return
            try:
                text = self.transcriber.transcribe(audio)
            except Exception as e:
                print(f"transcribe failed: {e}", file=sys.stderr, flush=True)
                self.set_state("idle")
                continue
            if text:
                try:
                    paste_text(text, self.cfg["paste_delay_ms"])
                except Exception as e:
                    print(f"paste failed: {e}", file=sys.stderr, flush=True)
                # Log without unicode chars — Windows console codec is cp1252.
                try:
                    safe = text.encode("ascii", "replace").decode("ascii")
                    print(f"-> {safe}", flush=True)
                except Exception:
                    pass
            self.set_state("idle")

    def quit(self) -> None:
        if self.icon is not None:
            self.icon.stop()
        self.work_q.put(None)

    def run(self) -> None:
        threading.Thread(target=self.worker, daemon=True).start()

        hotkey = self.cfg["hotkey"]
        # keyboard.on_press_key fires once per actual press; on_release_key once per release.
        # Using suppress=False so other apps don't receive the F13 key — but F13 is unused
        # by default Windows so suppression is unnecessary anyway.
        keyboard.on_press_key(hotkey, lambda _e: self.on_press(), suppress=False)
        keyboard.on_release_key(hotkey, lambda _e: self.on_release(), suppress=False)

        self.icon = Icon(
            "dictation",
            make_icon("idle"),
            "Dictation",
            menu=Menu(
                MenuItem(f"Hotkey: {hotkey}", None, enabled=False),
                MenuItem(f"Model: {self.cfg['model']}", None, enabled=False),
                MenuItem("Quit", lambda _i, _it: self.quit()),
            ),
        )
        print(f"ready. hold [{hotkey}] to dictate.", flush=True)
        self.icon.run()


def main() -> int:
    cfg = load_config()
    Dictator(cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
