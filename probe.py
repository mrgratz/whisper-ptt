"""Probe: log every keyboard event for 20s. Press F13 (the remapped SS key) once."""
import keyboard
import time
import sys

print("=" * 50)
print("PRESS F13 NOW (the remapped SteelSeries key).")
print("Press it once, hold for ~1 second, release.")
print("Listening for 20 seconds...")
print("=" * 50, flush=True)

events = []
def on_event(e):
    events.append(e)
    et = "DOWN" if e.event_type == "down" else "UP  "
    print(f"  {et} name={e.name!r:15} scan={e.scan_code:5} time={e.time:.2f}", flush=True)

keyboard.hook(on_event)
time.sleep(20)
keyboard.unhook_all()

print("=" * 50)
print(f"TOTAL EVENTS: {len(events)}")
f13_like = [e for e in events if (e.name and "f13" in e.name.lower()) or e.scan_code == 100 or e.scan_code == 64]
print(f"F13-like events: {len(f13_like)}")
if events and not f13_like:
    print("Got keyboard events but no F13. The SS remap may not be active, or it's mapping to a different key.")
    print("Names seen:", sorted(set(e.name for e in events if e.name)))
elif not events:
    print("Got ZERO events. Possible causes:")
    print("  - You didn't press anything during the window")
    print("  - The keyboard library's hook didn't install (rare; would normally raise)")
    print("  - Some other process is suppressing the hook")
