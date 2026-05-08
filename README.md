# Dictation — gotchas

Working state and the things that aren't obvious from `main.py`.

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
