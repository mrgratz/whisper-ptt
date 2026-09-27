# whisper-ptt

Local push-to-talk dictation for Windows. Hold a hotkey, speak, release: the transcript pastes wherever your cursor is. Runs [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on your own GPU (or CPU); nothing leaves the machine.

- System tray app with a settings window (hotkey, model, gesture timings)
- Hold for push-to-talk, double-tap for hands-free
- The model loads in a separate worker that exits after idle, so the resident tray app stays around 30 MB

## Install

```pwsh
pip install -r requirements.txt
# NVIDIA GPU: also install the CUDA runtime wheels (see "CUDA DLL preload" below)
pythonw main.py
```

First run writes `config.json` next to `main.py` from the defaults. `config.json`, `history.json` and `daemon.log` are local runtime files and are gitignored.

## Notes

## Usage

- **Hold** the hotkey (default F13) for push-to-talk: speak while held, transcript pastes on release.
- **Double-tap** to start hands-free recording (tray goes red + a "Recording…" notification); a **single tap** stops it and pastes.
- **Tray → Settings…** — rebind the hotkey, "Capture this key fully" (suppress its normal function, needed for keys like Caps Lock / Right Alt), model, max length, trailing space, and the gesture timings. Most settings apply live; model applies on the next dictation.
- **Tray → Recent (click to copy)** — the last 5 transcriptions, truncated; click one to copy its full text. Persisted in `history.json`.
- **Tray icon color** — gray = idle, red = recording, amber = transcribing, red/amber split = recording a new take while a previous one is still transcribing. The color is derived from two independent facts (am I recording / how many jobs are in flight), so a finishing transcription can never gray out the icon mid-recording.
- Config is `config.json` (read at startup; the settings window rewrites it). Set `"debug": true` to log every gesture to `daemon.log` for calibrating the tap timings.

## GPU compute type

Pascal cards (GTX 10xx, CC 6.x) **don't** support `float16` or `int8_float16` on CTranslate2. Use `int8_float32`. Probe what's available:

```pwsh
python -c "import ctranslate2; print(ctranslate2.get_supported_compute_types('cuda'))"
```

| GPU | Pick |
|---|---|
| Pascal (GTX 10xx) | `int8_float32` |
| Turing (RTX 20xx, GTX 16xx) | `int8_float16` |
| Ampere+ (RTX 30xx, 40xx) | `int8_bfloat16` |

## CUDA DLL preload

CTranslate2 4.x needs `cublas64_12.dll` and `cudnn*_9.dll`. Pip wheels supply them:

```pwsh
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

`os.add_dll_directory` alone doesn't reach CTranslate2's loader — it uses `LOAD_LIBRARY_SEARCH_DEFAULT_DIRS` which excludes user-added dirs. `main.py` preloads via `ctypes.CDLL` instead.

## SteelSeries key

The Apex Pro logo key has no Windows-visible scancode — it goes through SteelSeries GG's vendor HID interface. Capturing it below GG is a multi-day USB RE project.

Workaround: in **SteelSeries GG → Apex Pro → key bindings**, remap the SS key to **F13**. GG must be running for the remap to be active.

## Auto-launch

Shortcut at `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Dictation.lnk` runs `pythonw.exe main.py`. Output → `daemon.log` (because `pythonw` has no console).

To disable: drag shortcut out of the Startup folder.

## Recording length cap is a silent audio truncation

`max_duration_sec` clips the recorded audio to that many seconds **before** it reaches Whisper (`main.py` `_finalize_release`), with no warning — a long dictation just loses everything past the cap. It was 60s, which cut off any take over ~1 minute halfway through. Raised to 300s and exposed in `config.json`. Its partner `worker_timeout_sec` (the model-load + transcribe budget) must track it: if a recording is longer than the worker can transcribe in time, the whole take is dropped, not truncated. Both are read only at daemon startup — restart after editing.

## Restart carefully — verify by PID, not by WMI command line

When restarting the daemon, a freshly-launched `pythonw.exe` can show a **null `CommandLine`** in WMI/CIM for a second or two, so a "did it start?" check that filters `Get-CimInstance Win32_Process` by a command-line substring gives a false negative on the new process. Trusting that false negative and launching again leaves **two daemons** both bound to the hotkey — every transcription gets pasted twice. Verify a launch by the PID from `Start-Process -PassThru` (`$proc.HasExited`) or by the fresh `ready.` line in `daemon.log`, never by a command-line-substring process scan. Kill an accidental duplicate with `Stop-Process -Id <pid> -Force`.

## Warm worker: fast back-to-back dictation, memory released on idle

The transcription worker now stays **warm** between dictations instead of one process per job. It loads the model once, serves each take over stdin, and exits on its own after `worker_idle_sec` (default 90s) of inactivity — so idle memory still falls back to the ~27MB stub (the "no idle ballast" property is preserved), but a burst of dictation only pays the model load once.

Measured (medium.en / cuda / int8_float32, this box): **cold job = ~14.5s load + ~1.8s transcribe; warm job = ~0.0–1.8s (no reload)** — a ~1000× drop in the load component for every dictation within the window. The first dictation after idle still pays the cold load, but prespawn (worker spawns on gesture-confirm) overlaps most of it with the operator's speech.

Knobs:
- `worker_idle_sec` (settings: "Keep model warm (sec)") — how long the model stays resident after the last dictation. `0` = one process per job (the old  shape: lowest idle memory, slowest per dictation).
- `worker_prespawn` — spawn the worker the moment a hold/hands-free gesture is confirmed, so the first cold load runs while you talk.

Not yet done: pre-converting `medium.en` to an int8 CT2 format on disk would shrink the *cold* load (~14.5s) further; the warm worker already removes it from every back-to-back dictation, which is where the "sometimes takes a long time" latency came from.
