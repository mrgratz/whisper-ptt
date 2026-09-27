"""
Push-to-talk + hands-free dictation daemon.

Two gestures on ONE configurable hotkey (default F13):
  - HOLD the key   -> push-to-talk: record while held, transcribe on release.
  - DOUBLE-TAP     -> hands-free: recording starts and the tray shows "Recording";
                      a SINGLE tap stops it and transcribes.

Settings live in a small tray-launched window (Settings...): hotkey (rebind +
"capture this key fully" to suppress its normal function), model, max recording
length, trailing space, and the gesture timings. Most settings apply live; model
changes apply on the next dictation (the worker reads config per spawn).

Process shape: this stub holds NO model. faster-whisper and
the CUDA context live in `transcribe_worker.py`. The worker stays WARM between
dictations for `worker_idle_sec` (default 90 s), serving each take over stdin
without reloading the model, then exits on its own once that idle window passes —
so a burst of dictation is near-instant (~2 s each) but idle memory falls back to
the stub alone (~27 MB) instead of ~2.3 GB parked in the pagefile (the 2026-09-01
commit-exhaustion crash named this process first). `worker_idle_sec: 0` reverts to
one process per job. With `worker_prespawn` on, the worker is spawned once a
gesture is confirmed a hold/hands-free (not on quick taps), so the FIRST load
after idle overlaps the operator's speech.

Single-instance: a named mutex refuses a second copy, so a careless
relaunch can't leave two daemons both pasting every transcription (the
double-daemon duplication). Verify a launch by PID / the daemon.log `ready.` line,
never by a WMI command-line scan (it can read null on a fresh pythonw).
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional


# Redirect to a log file when run under pythonw.exe (no console). With
# python.exe, sys.stdout is a real fd; with pythonw it's None.
def _setup_headless_logging() -> None:
    if sys.stdout is None:
        log_path = Path(__file__).parent / "daemon.log"
        f = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = f
        sys.stderr = f


_setup_headless_logging()

# DL-457: this tray app runs headless under pythonw (Startup-folder launch), so
# any console child (clipboard helper, the transcription worker) would flash a
# window. Install the no-window patch before anything below can spawn.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/
from win_console import suppress_child_windows  # noqa: E402
suppress_child_windows()

# numpy's bundled OpenBLAS commits ~31 MB of scratch PER CPU THREAD at import
# time (497 MB on a 16-core box, never touched; measured ). The
# stub only concatenates audio frames, so one BLAS thread is plenty. Inherited
# by the worker via Popen env.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import keyboard
import numpy as np
import pyperclip
import sounddevice as sd
from PIL import Image, ImageDraw
from pystray import Icon, Menu, MenuItem

import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.json"
WORKER_SCRIPT = ROOT / "transcribe_worker.py"
HISTORY_PATH = ROOT / "history.json"
HISTORY_MAX = 5
SAMPLE_RATE = 16000

MODELS = ["tiny.en", "base.en", "small.en", "medium.en", "large-v3"]

DEFAULT_CONFIG = {
    "hotkey": "f13",
    # Swallow the bound key so it can't also fire its normal function (e.g. bind
    # Caps Lock or Right Alt without it toggling caps / triggering alt). Off by
    # default: f13 needs no suppression, and suppression interacts with UP-event
    # delivery — enable it only when you rebind to a key that has a normal job.
    "suppress_key": False,
    "model": "medium.en",
    "device": "cuda",
    "compute_type": "int8_float32",
    "min_duration_sec": 0.3,
    "max_duration_sec": 300.0,
    "paste_delay_ms": 50,
    # Append a single space after the transcript so you can keep typing without
    # hitting space first.
    "trailing_space": True,
    # Spawn the transcription worker once a gesture is confirmed (hold/hands-free)
    # so the model loads while the operator speaks. Quick taps never spawn one.
    "worker_prespawn": True,
    # Hard cap on one worker run (model load + transcribe). Past it the worker
    # is killed and the recording is dropped.
    "worker_timeout_sec": 300,
    # Keep the loaded model warm between dictations for this many seconds of
    # inactivity, then let the worker exit and release its ~2 GB. The FIRST
    # dictation after idle pays the ~13 s model load (mostly hidden under speech
    # by prespawn); every dictation within the window reuses the resident model
    # and is near-instant (~2 s). 0 = cold every time (one process per job, the
    # one-process-per-job behaviour — lowest idle memory, slowest per dictation).
    "worker_idle_sec": 90,
    # ── Gesture model ────────────────────────────────────────────────────────
    "enable_hold": True,          # HOLD the key = push-to-talk
    "enable_double_tap": True,    # DOUBLE-TAP = hands-free; single tap stops it
    # A press held at least this long is a HOLD; shorter is a TAP.
    "hold_threshold_ms": 350,
    # Two taps within this window = a double-tap.
    "double_tap_ms": 400,
    # Some remappers (e.g. SteelSeries GG -> F13) emit a burst of taps WHILE the
    # key is physically held. A new press within this gap of the last release is
    # treated as a continuation of the same hold, not a new tap — so a burst
    # coalesces into one HOLD and doesn't read as a double-tap. Keep it below a
    # deliberate double-tap's inter-tap gap (~150-300 ms). 0 = off (default);
    # only turn it up if the debug log shows a held key emitting a tap-burst.
    "coalesce_ms": 0,
    # Log every key event + gesture decision to daemon.log (calibration aid).
    "debug": False,
}


def load_config() -> dict:
    file_cfg = {}
    if CONFIG_PATH.exists():
        try:
            file_cfg = json.loads(CONFIG_PATH.read_text())
        except Exception as e:
            print(f"config unreadable ({e}); using defaults", file=sys.stderr, flush=True)
    cfg = {**DEFAULT_CONFIG, **file_cfg}
    # Backfill any newly-added default keys into the file so they show up in the
    # settings window and config.json.
    if set(cfg) - set(file_cfg):
        save_config(cfg)
    return cfg


def save_config(cfg: dict) -> None:
    # Persist only recognised keys, in DEFAULT_CONFIG order for a stable file.
    ordered = {k: cfg.get(k, DEFAULT_CONFIG[k]) for k in DEFAULT_CONFIG}
    CONFIG_PATH.write_text(json.dumps(ordered, indent=2))


def load_history() -> list:
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [e for e in data if isinstance(e, dict) and e.get("text")][:HISTORY_MAX]
    except Exception:
        pass
    return []


def save_history(hist: list) -> None:
    try:
        HISTORY_PATH.write_text(
            json.dumps(hist[:HISTORY_MAX], ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"history save failed: {e}", file=sys.stderr, flush=True)


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


# ── Transcription (worker process per job) ───────────────────────────────

def _worker_python() -> str:
    # Under pythonw (Startup launch) sys.executable is pythonw.exe. The worker
    # needs a real stdout pipe, so run the console-subsystem sibling with
    # CREATE_NO_WINDOW (DL-457: no phantom console from a headless parent).
    exe = Path(sys.executable)
    cand = exe.with_name("python.exe")
    return str(cand) if cand.exists() else str(exe)


def _stderr_target():
    try:
        sys.stderr.fileno()
        return sys.stderr
    except Exception:
        return subprocess.DEVNULL


def _close_quietly(proc: subprocess.Popen, grace_sec: float = 5.0) -> None:
    """EOF on stdin tells an unused worker to exit; kill it if it lingers."""
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class _WorkerDied(RuntimeError):
    """The warm worker's pipe broke or it had already exited (idle-exit fired)."""


class WorkerTranscriber:
    """A warm transcribe_worker.py: load the model once, serve many jobs over
    stdin, and let the worker itself exit after `worker_idle_sec` of inactivity
    (releasing its ~2 GB). The first dictation after idle pays the model load;
    dictations within the window reuse the resident model and skip it.

    worker_idle_sec == 0 falls back to one process per job (the original shape):
    the worker reads a single line and exits, and each transcription respawns.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def _idle_sec(self) -> float:
        try:
            return max(0.0, float(self.cfg.get("worker_idle_sec", 90)))
        except (TypeError, ValueError):
            return 90.0

    def _spawn(self) -> subprocess.Popen:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        idle = self._idle_sec()
        proc = subprocess.Popen(
            [_worker_python(), str(WORKER_SCRIPT), "--config", str(CONFIG_PATH),
             "--idle-exit", str(idle)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=_stderr_target(),
            text=True, encoding="utf-8", creationflags=flags, cwd=str(ROOT),
        )
        print(f"worker spawned pid={proc.pid} (warm, idle-exit {idle:.0f}s)", flush=True)
        return proc

    def prespawn(self) -> None:
        """Start the model loading now (on gesture confirm) so it's warm — or
        already resident — by the time the operator releases the key."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return  # already warm
            try:
                self._proc = self._spawn()
            except Exception as e:
                print(f"worker prespawn failed: {e}", file=sys.stderr, flush=True)
                self._proc = None

    def _live_proc(self) -> subprocess.Popen:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                proc = self._spawn()
                self._proc = proc
            return proc

    def discard(self) -> None:
        """A too-short/cancelled capture no longer kills the worker — it stays
        warm to serve the next real dictation, and idle-exits on its own."""
        return

    def shutdown(self) -> None:
        """Daemon quitting: close the warm worker."""
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            _close_quietly(proc)

    def _run_on(self, proc: subprocess.Popen, path: str, timeout: float) -> str:
        if proc.stdin is None or proc.stdout is None:
            raise _WorkerDied("worker has no pipes")
        result: dict = {}
        err: dict = {}

        def pump() -> None:
            try:
                while True:
                    line = proc.stdout.readline()
                    if line == "":  # EOF -> worker gone
                        return
                    line = line.strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "ok" in msg:  # skip the one-time {"ready": ...} line
                        result["msg"] = msg
                        return
            except Exception as e:  # pragma: no cover - pipe teardown races
                err["e"] = e

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            proc.stdin.write(json.dumps({"audio": path}) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as e:
            raise _WorkerDied(f"stdin write failed: {e}")

        reader.join(timeout)
        if reader.is_alive():
            proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
            raise RuntimeError(f"worker timed out after {timeout:.0f}s")
        if "e" in err:
            raise _WorkerDied(str(err["e"]))
        msg = result.get("msg")
        if msg is None:
            raise _WorkerDied(f"worker exited rc={proc.returncode} without a result")
        if msg.get("ok"):
            print(f"worker done load={msg.get('load_sec')}s transcribe={msg.get('transcribe_sec')}s",
                  flush=True)
            return msg.get("text", "")
        raise RuntimeError(msg.get("error", "worker failed"))

    def transcribe(self, audio: np.ndarray) -> str:
        fd, path = tempfile.mkstemp(prefix="dictation-", suffix=".npy")
        os.close(fd)
        np.save(path, audio.astype(np.float32, copy=False))
        timeout = float(self.cfg["worker_timeout_sec"])
        try:
            proc = self._live_proc()
            try:
                return self._run_on(proc, path, timeout)
            except _WorkerDied as e:
                # The warm worker had exited (idle-exit fired between prespawn and
                # release, or the pipe broke). Spawn a fresh one and retry once —
                # this dictation pays the model load.
                print(f"warm worker gone ({e}); respawning", file=sys.stderr, flush=True)
                with self._lock:
                    self._proc = None
                proc = self._live_proc()
                return self._run_on(proc, path, timeout)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


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

    def _restore():
        time.sleep(0.5)
        try:
            pyperclip.copy(prev_clip)
        except Exception:
            pass
    threading.Thread(target=_restore, daemon=True).start()


# ── Tray icon ────────────────────────────────────────────────────────────

_REC = (220, 60, 60)      # recording (red)
_WORK = (240, 180, 40)    # transcribing (amber)
_IDLE = (110, 110, 110)   # nothing happening (gray)


def make_icon(state: str) -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if state == "recording_working":
        # Recording a new take WHILE a prior transcription is still running:
        # left half red (recording), right half amber (a job cooking). Never gray.
        d.pieslice((10, 10, 54, 54), start=90, end=270, fill=_REC)
        d.pieslice((10, 10, 54, 54), start=-90, end=90, fill=_WORK)
    else:
        color = {"recording": _REC, "working": _WORK}.get(state, _IDLE)
        d.ellipse((10, 10, 54, 54), fill=color)
    d.rectangle((28, 24, 36, 44), fill=(255, 255, 255))
    d.ellipse((24, 42, 40, 50), fill=(255, 255, 255))
    return img


# ── Main controller ──────────────────────────────────────────────────────

class Dictator:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.recorder = Recorder()
        self.transcriber = WorkerTranscriber(cfg)
        self.work_q: queue.Queue = queue.Queue()
        self.icon: Optional[Icon] = None
        self.root: Optional[tk.Tk] = None
        self.lock = threading.RLock()
        self.history: list = load_history()

        # Tray-icon state, tracked as two INDEPENDENT facts so a finishing
        # transcription can't paint the icon gray while a new take is recording.
        self._recording = False        # a capture is live right now
        self._jobs_in_flight = 0       # transcriptions queued or running

        # Gesture state
        self._press_hook = None
        self._release_hook = None
        self.gesture_active = False       # a press-gesture is recording (hold candidate)
        self.gesture_pending_end = False  # release seen; awaiting coalesce window
        self.gesture_start = 0.0
        self.last_release = 0.0
        self.coalesce_timer: Optional[threading.Timer] = None
        self.prespawn_timer: Optional[threading.Timer] = None
        self.tap_count = 0
        self.last_tap = 0.0               # time of the previous completed tap
        self.hands_free = False
        self.consuming = False            # a stop-gesture key-hold is in progress

    # ---- tray state ----
    # The icon color is DERIVED from two independent facts (recording + jobs in
    # flight), never set directly by whichever event fired last — so finishing a
    # transcription can't gray out the icon while a new take is being recorded.
    def _refresh_icon(self) -> None:
        with self.lock:
            rec = self._recording
            working = self._jobs_in_flight > 0
        if rec and working:
            state = "recording_working"
        elif rec:
            state = "recording"
        elif working:
            state = "working"
        else:
            state = "idle"
        if self.icon is not None:
            try:
                self.icon.icon = make_icon(state)
            except Exception:
                pass

    def _set_recording(self, on: bool) -> None:
        with self.lock:
            self._recording = on
        self._refresh_icon()

    def _job_started(self) -> None:
        with self.lock:
            self._jobs_in_flight += 1
        self._refresh_icon()

    def _job_finished(self) -> None:
        with self.lock:
            if self._jobs_in_flight > 0:
                self._jobs_in_flight -= 1
        self._refresh_icon()

    def notify(self, msg: str) -> None:
        if self.icon is not None:
            try:
                self.icon.notify(msg, "Dictation")
            except Exception:
                pass

    # ---- key event dispatch ----
    def _dbg(self, msg: str) -> None:
        if self.cfg.get("debug", False):
            print(f"[dbg {time.time():.3f}] {msg}", flush=True)

    def _on_press_evt(self, _e=None) -> None:
        self.on_press()

    def _on_release_evt(self, _e=None) -> None:
        self.on_release()

    def on_press(self) -> None:
        now = time.time()
        with self.lock:
            gap = (now - self.last_release) * 1000 if self.last_release else 9999.0
            if self.consuming:
                return  # ignore the stop-gesture key's auto-repeat until it releases
            if self.hands_free:
                # Any press while hands-free is the STOP gesture.
                self._dbg("DOWN -> stop hands-free")
                self.consuming = True
                threading.Thread(target=self._stop_hands_free, daemon=True).start()
                return
            coal = self.cfg.get("coalesce_ms", 0)
            if coal and self.gesture_pending_end and gap < coal:
                # Burst continuation (a remapper emitting a tap-burst while held):
                # keep the same gesture instead of reading it as a new tap.
                self.gesture_pending_end = False
                if self.coalesce_timer is not None:
                    self.coalesce_timer.cancel()
                    self.coalesce_timer = None
                self._dbg("DOWN coalesced (burst continuation)")
                return
            if self.gesture_active:
                return  # auto-repeat during a hold (not logged — Windows key repeat)
            self.gesture_active = True
            self.gesture_pending_end = False
            self.gesture_start = now
            self._dbg(f"DOWN new gesture gap={gap:.0f}ms")
        try:
            self.recorder.start()
            self._set_recording(True)
        except Exception as e:
            print(f"recorder failed to start: {e}", file=sys.stderr, flush=True)
            with self.lock:
                self.gesture_active = False
            return
        # Prespawn the worker only once this gesture is confirmed a hold — a quick
        # tap never spawns (and discards) a worker.
        if self.cfg.get("worker_prespawn", True):
            with self.lock:
                if self.prespawn_timer is not None:
                    self.prespawn_timer.cancel()
                self.prespawn_timer = threading.Timer(
                    self.cfg["hold_threshold_ms"] / 1000.0,
                    lambda: threading.Thread(target=self.transcriber.prespawn, daemon=True).start(),
                )
                self.prespawn_timer.daemon = True
                self.prespawn_timer.start()

    def on_release(self) -> None:
        now = time.time()
        with self.lock:
            if self.consuming:
                self.consuming = False  # stop-gesture key released; settle
                self._dbg("UP (consumed stop-gesture)")
                return
            if not self.gesture_active:
                self._dbg("UP (no active gesture)")
                return
            self.last_release = now
            self._dbg(f"UP dur={(now - self.gesture_start) * 1000:.0f}ms")
            coal = self.cfg.get("coalesce_ms", 0)
            if coal:
                # Defer the decision so a burst re-press can continue the gesture.
                self.gesture_pending_end = True
                if self.coalesce_timer is not None:
                    self.coalesce_timer.cancel()
                self.coalesce_timer = threading.Timer(coal / 1000.0, self._end_gesture)
                self.coalesce_timer.daemon = True
                self.coalesce_timer.start()
                return
        self._end_gesture()

    def _end_gesture(self) -> None:
        with self.lock:
            if not self.gesture_active:
                return
            self.gesture_active = False
            self.gesture_pending_end = False
            self.coalesce_timer = None
            if self.prespawn_timer is not None:
                self.prespawn_timer.cancel()
                self.prespawn_timer = None
            dur = self.last_release - self.gesture_start
            hold_ok = self.cfg.get("enable_hold", True)
            dt_ok = self.cfg.get("enable_double_tap", True)
            hold_thresh = self.cfg["hold_threshold_ms"] / 1000.0
            dt_window = self.cfg["double_tap_ms"] / 1000.0
        audio, rec_dur = self.recorder.stop()
        self._set_recording(False)

        if dur >= hold_thresh:
            # HOLD -> push-to-talk
            with self.lock:
                self.tap_count = 0
            self._dbg(f"  HOLD dur={dur * 1000:.0f}ms rec={rec_dur:.2f}s")
            if hold_ok and rec_dur >= self.cfg["min_duration_sec"]:
                self._enqueue(audio)
            # else: too short — recording flag already cleared; the warm worker
            # stays resident for the next dictation (idle-exits on its own).
            return

        # TAP -> nothing captured; resolve tap / double-tap. The prespawned worker
        # (if any) stays warm for the next real dictation.
        if not dt_ok:
            with self.lock:
                self.tap_count = 0
            return
        now = time.time()
        with self.lock:
            if self.last_tap and (now - self.last_tap) <= dt_window:
                self.tap_count += 1
            else:
                self.tap_count = 1
            self.last_tap = now
            n = self.tap_count
            trigger = self.tap_count >= 2
            if trigger:
                self.tap_count = 0
        self._dbg(f"  TAP count={n} -> hands_free={trigger}")
        if trigger:
            self._start_hands_free()

    def _start_hands_free(self) -> None:
        with self.lock:
            self.hands_free = True
        try:
            self.recorder.start()
        except Exception as e:
            print(f"hands-free start failed: {e}", file=sys.stderr, flush=True)
            with self.lock:
                self.hands_free = False
            return
        self._set_recording(True)
        self.notify("Recording... single-tap to stop")
        print("hands-free ON", flush=True)
        if self.cfg.get("worker_prespawn", True):
            threading.Thread(target=self.transcriber.prespawn, daemon=True).start()

    def _stop_hands_free(self) -> None:
        with self.lock:
            if not self.hands_free:
                return
            self.hands_free = False
        audio, dur = self.recorder.stop()
        self._set_recording(False)
        print("hands-free OFF", flush=True)
        if dur >= self.cfg["min_duration_sec"]:
            self._enqueue(audio)
        # else: too short — recording flag cleared; warm worker stays for next use.

    def _enqueue(self, audio: np.ndarray) -> None:
        max_dur = float(self.cfg["max_duration_sec"])
        if len(audio) > int(max_dur * SAMPLE_RATE):
            audio = audio[: int(max_dur * SAMPLE_RATE)]
        self._job_started()
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
                self._job_finished()
                continue
            if text:
                self._add_history(text)
                if self.cfg.get("trailing_space", True):
                    text = text + " "
                try:
                    paste_text(text, self.cfg["paste_delay_ms"])
                except Exception as e:
                    print(f"paste failed: {e}", file=sys.stderr, flush=True)
                try:
                    safe = text.encode("ascii", "replace").decode("ascii")
                    print(f"-> {safe}", flush=True)
                except Exception:
                    pass
            self._job_finished()

    # ---- hotkey (re)binding ----
    def bind_hotkey(self) -> None:
        self.unbind_hotkey()
        key = self.cfg["hotkey"]
        suppress = bool(self.cfg.get("suppress_key", False))
        try:
            # Separate press/release hooks. hook_key(suppress=True) was observed
            # to swallow the UP event for f13, wedging the state machine
            # after the first press. on_release_key delivers UP reliably.
            self._press_hook = keyboard.on_press_key(key, self._on_press_evt, suppress=suppress)
            self._release_hook = keyboard.on_release_key(key, self._on_release_evt)
            print(f"bound [{key}] suppress={suppress}", flush=True)
        except Exception as e:
            print(f"hotkey bind failed for '{key}': {e}", file=sys.stderr, flush=True)

    def unbind_hotkey(self) -> None:
        for h in (getattr(self, "_press_hook", None), getattr(self, "_release_hook", None)):
            if h is not None:
                try:
                    keyboard.unhook(h)
                except Exception:
                    pass
        self._press_hook = None
        self._release_hook = None

    def apply_config(self, new_cfg: dict) -> None:
        """Called from the settings window on Save. Applies live."""
        rebind = (new_cfg.get("hotkey") != self.cfg.get("hotkey")
                  or new_cfg.get("suppress_key") != self.cfg.get("suppress_key"))
        self.cfg.clear()
        self.cfg.update(new_cfg)
        self.transcriber.cfg = self.cfg
        save_config(self.cfg)
        if rebind:
            self.bind_hotkey()
        print("settings applied", flush=True)

    # ---- recent-transcription history ----
    def _hist_label(self, text: str) -> str:
        label = " ".join(text.split())
        return (label[:45].rstrip() + "…") if len(label) > 45 else label

    def _copy_history(self, text: str) -> None:
        try:
            pyperclip.copy(text)
            self.notify("Copied to clipboard")
        except Exception as e:
            print(f"history copy failed: {e}", file=sys.stderr, flush=True)

    def _make_recent_menu(self) -> Menu:
        with self.lock:
            hist = list(self.history)
        if not hist:
            return Menu(MenuItem("(nothing yet)", None, enabled=False))
        items = [
            MenuItem(self._hist_label(e.get("text", "")),
                     (lambda t: (lambda _i, _it: self._copy_history(t)))(e.get("text", "")))
            for e in hist
        ]
        return Menu(*items)

    def _build_menu(self) -> Menu:
        return Menu(
            MenuItem(f"Hotkey: {self.cfg['hotkey']}", None, enabled=False),
            MenuItem(f"Model: {self.cfg['model']}", None, enabled=False),
            Menu.SEPARATOR,
            MenuItem("Recent (click to copy)", self._make_recent_menu()),
            Menu.SEPARATOR,
            MenuItem("Settings...", lambda _i, _it: self.request_settings(), default=True),
            MenuItem("Quit", lambda _i, _it: self.quit()),
        )

    def _refresh_menu(self) -> None:
        if self.icon is not None:
            try:
                self.icon.menu = self._build_menu()
                self.icon.update_menu()
            except Exception:
                pass

    def _add_history(self, text: str) -> None:
        text = " ".join(text.split()).strip()
        if not text:
            return
        with self.lock:
            self.history.insert(0, {"text": text, "ts": time.time()})
            self.history = self.history[:HISTORY_MAX]
            hist = list(self.history)
        save_history(hist)
        self._refresh_menu()

    # ---- settings window (tkinter, on the main thread) ----
    def open_settings(self) -> None:
        if self.root is None:
            return
        SettingsWindow(self.root, self)

    def request_settings(self) -> None:
        if self.root is not None:
            self.root.after(0, self.open_settings)

    # ---- lifecycle ----
    def quit(self) -> None:
        self.unbind_hotkey()
        self.transcriber.shutdown()
        self.work_q.put(None)
        if self.icon is not None:
            self.icon.stop()
        if self.root is not None:
            self.root.after(0, self.root.quit)

    def run(self) -> None:
        threading.Thread(target=self.worker, daemon=True).start()
        self.bind_hotkey()

        # tkinter owns the main thread (hidden root); the tray runs detached.
        self.root = tk.Tk()
        self.root.withdraw()

        mode_bits = []
        if self.cfg.get("enable_hold", True):
            mode_bits.append("hold=push-to-talk")
        if self.cfg.get("enable_double_tap", True):
            mode_bits.append("double-tap=hands-free")
        mode = ", ".join(mode_bits) or "no gestures enabled"

        self.icon = Icon("dictation", make_icon("idle"), "Dictation", menu=self._build_menu())
        self.icon.run_detached()
        print(f"ready. hold [{self.cfg['hotkey']}] to dictate ({mode}).", flush=True)
        self.root.mainloop()


# ── Settings window ──────────────────────────────────────────────────────

class SettingsWindow:
    def __init__(self, root: tk.Tk, dictator: Dictator) -> None:
        self.d = dictator
        cfg = dictator.cfg
        self.win = tk.Toplevel(root)
        self.win.title("Dictation Settings")
        self.win.resizable(False, False)
        self.win.attributes("-topmost", True)
        pad = {"padx": 10, "pady": 4}

        frm = ttk.Frame(self.win, padding=14)
        frm.grid(row=0, column=0, sticky="nsew")
        r = 0

        ttk.Label(frm, text="Dictation", font=("Segoe UI", 12, "bold")).grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(0, 8))
        r += 1

        # Hotkey
        ttk.Label(frm, text="Hotkey").grid(row=r, column=0, sticky="w", **pad)
        self.hotkey_var = tk.StringVar(value=cfg["hotkey"])
        self.hotkey_lbl = ttk.Label(frm, textvariable=self.hotkey_var,
                                    relief="solid", width=18, anchor="center")
        self.hotkey_lbl.grid(row=r, column=1, sticky="w", **pad)
        self.capture_btn = ttk.Button(frm, text="Change", command=self._capture_hotkey)
        self.capture_btn.grid(row=r, column=2, sticky="w", **pad)
        r += 1

        # Suppress
        self.suppress_var = tk.BooleanVar(value=bool(cfg.get("suppress_key", True)))
        ttk.Checkbutton(frm, text="Capture this key fully (block its normal function)",
                        variable=self.suppress_var).grid(
            row=r, column=0, columnspan=3, sticky="w", **pad)
        r += 1

        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=8)
        r += 1

        # Gestures
        self.hold_var = tk.BooleanVar(value=bool(cfg.get("enable_hold", True)))
        ttk.Checkbutton(frm, text="Hold to talk (push-to-talk)",
                        variable=self.hold_var).grid(
            row=r, column=0, columnspan=3, sticky="w", **pad)
        r += 1
        self.dt_var = tk.BooleanVar(value=bool(cfg.get("enable_double_tap", True)))
        ttk.Checkbutton(frm, text="Double-tap for hands-free (single tap to stop)",
                        variable=self.dt_var).grid(
            row=r, column=0, columnspan=3, sticky="w", **pad)
        r += 1

        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=8)
        r += 1

        # Model
        ttk.Label(frm, text="Model").grid(row=r, column=0, sticky="w", **pad)
        self.model_var = tk.StringVar(value=cfg["model"])
        model_vals = MODELS if cfg["model"] in MODELS else MODELS + [cfg["model"]]
        ttk.Combobox(frm, textvariable=self.model_var, values=model_vals,
                     state="readonly", width=16).grid(row=r, column=1, columnspan=2, sticky="w", **pad)
        r += 1

        # Max length
        ttk.Label(frm, text="Max length (sec)").grid(row=r, column=0, sticky="w", **pad)
        self.maxlen_var = tk.StringVar(value=str(int(float(cfg["max_duration_sec"]))))
        ttk.Spinbox(frm, from_=10, to=3600, increment=30, textvariable=self.maxlen_var,
                    width=8).grid(row=r, column=1, sticky="w", **pad)
        r += 1

        # Keep model warm — reuse the loaded model between dictations for this
        # many idle seconds (faster back-to-back; 0 = reload every time).
        ttk.Label(frm, text="Keep model warm (sec)").grid(row=r, column=0, sticky="w", **pad)
        self.warm_var = tk.StringVar(value=str(int(float(cfg.get("worker_idle_sec", 90)))))
        ttk.Spinbox(frm, from_=0, to=600, increment=30, textvariable=self.warm_var,
                    width=8).grid(row=r, column=1, sticky="w", **pad)
        r += 1

        # Trailing space
        self.space_var = tk.BooleanVar(value=bool(cfg.get("trailing_space", True)))
        ttk.Checkbutton(frm, text="Add a space after each dictation",
                        variable=self.space_var).grid(
            row=r, column=0, columnspan=3, sticky="w", **pad)
        r += 1

        # Advanced timings
        adv = ttk.LabelFrame(frm, text="Gesture timing (ms)", padding=8)
        adv.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        self.hold_ms_var = tk.StringVar(value=str(int(cfg["hold_threshold_ms"])))
        self.dt_ms_var = tk.StringVar(value=str(int(cfg["double_tap_ms"])))
        self.coal_ms_var = tk.StringVar(value=str(int(cfg["coalesce_ms"])))
        ttk.Label(adv, text="Hold threshold").grid(row=0, column=0, sticky="w", padx=6, pady=2)
        ttk.Spinbox(adv, from_=100, to=1500, increment=50, textvariable=self.hold_ms_var,
                    width=7).grid(row=0, column=1, padx=6, pady=2)
        ttk.Label(adv, text="Double-tap window").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        ttk.Spinbox(adv, from_=150, to=1000, increment=50, textvariable=self.dt_ms_var,
                    width=7).grid(row=1, column=1, padx=6, pady=2)
        ttk.Label(adv, text="Burst coalesce").grid(row=2, column=0, sticky="w", padx=6, pady=2)
        ttk.Spinbox(adv, from_=0, to=400, increment=20, textvariable=self.coal_ms_var,
                    width=7).grid(row=2, column=1, padx=6, pady=2)
        r += 1

        self.status = ttk.Label(frm, text="", foreground="#0a7d18")
        self.status.grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        r += 1

        btns = ttk.Frame(frm)
        btns.grid(row=r, column=0, columnspan=3, sticky="e", pady=(8, 0))
        ttk.Button(btns, text="Save", command=self._save).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="Close", command=self.win.destroy).grid(row=0, column=1, padx=4)

        self.win.update_idletasks()
        self.win.lift()
        self.win.focus_force()

    def _capture_hotkey(self) -> None:
        self.capture_btn.config(state="disabled")
        self.hotkey_var.set("press a key...")

        def listen():
            try:
                ev = keyboard.read_event(suppress=False)
                while ev.event_type != "down":
                    ev = keyboard.read_event(suppress=False)
                name = ev.name or self.d.cfg["hotkey"]
            except Exception:
                name = self.d.cfg["hotkey"]

            def done():
                self.hotkey_var.set(name)
                self.capture_btn.config(state="normal")
            try:
                self.win.after(0, done)
            except Exception:
                pass

        threading.Thread(target=listen, daemon=True).start()

    def _save(self) -> None:
        cfg = dict(self.d.cfg)
        cfg["hotkey"] = self.hotkey_var.get().strip() or cfg["hotkey"]
        cfg["suppress_key"] = self.suppress_var.get()
        cfg["enable_hold"] = self.hold_var.get()
        cfg["enable_double_tap"] = self.dt_var.get()
        cfg["model"] = self.model_var.get()
        cfg["trailing_space"] = self.space_var.get()

        def _num(s, default, cast):
            try:
                return cast(s)
            except Exception:
                return default
        cfg["max_duration_sec"] = float(_num(self.maxlen_var.get(), cfg["max_duration_sec"], float))
        cfg["worker_timeout_sec"] = max(int(cfg["max_duration_sec"]), int(cfg.get("worker_timeout_sec", 300)))
        cfg["worker_idle_sec"] = _num(self.warm_var.get(), cfg.get("worker_idle_sec", 90), int)
        cfg["hold_threshold_ms"] = _num(self.hold_ms_var.get(), cfg["hold_threshold_ms"], int)
        cfg["double_tap_ms"] = _num(self.dt_ms_var.get(), cfg["double_tap_ms"], int)
        cfg["coalesce_ms"] = _num(self.coal_ms_var.get(), cfg["coalesce_ms"], int)

        self.d.apply_config(cfg)
        self.status.config(text="Saved. Hotkey applies now; model on next dictation.")


# ── Single instance ──────────────────────────────────────────────────────

def _acquire_single_instance() -> bool:
    """Named mutex so a second launch refuses instead of double-pasting."""
    if sys.platform != "win32":
        return True
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    handle = kernel32.CreateMutexW(None, False, "Global\\MatzekDictationDaemon")
    ERROR_ALREADY_EXISTS = 183
    if kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
        return False
    # Leak the handle intentionally: it lives for the process lifetime.
    _SINGLE_INSTANCE_HANDLES.append(handle)
    return True


_SINGLE_INSTANCE_HANDLES: list = []


def main() -> int:
    if not _acquire_single_instance():
        print("another dictation instance is already running; exiting.", flush=True)
        return 0
    cfg = load_config()
    Dictator(cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
