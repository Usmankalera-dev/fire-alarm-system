# 🔥 AI-Based Fire Alarm System

A computer-vision fire alarm that watches a camera feed, detects flames,
and raises an alarm — no smoke sensor hardware needed.

## How it works

1. **Colour segmentation** — each frame is converted to HSV; pixels in the
   flame range (red → orange → yellow, bright + saturated) are extracted.
2. **Noise cleanup** — morphological open/close removes speckles; only
   blobs above a minimum area count.
3. **Flicker check** — real flames dance, so the flame area must jitter
   frame-to-frame. This rejects static orange objects (lamps, posters).
4. **Temporal alarm** — fire must be confirmed in 5 of the last 10 frames
   before the alarm turns ON; 15 quiet frames turn it back OFF.

No training data, no GPU — pure OpenCV, runs on CPU.

## Files

| File | Purpose |
|---|---|
| `fire_detector.py` | `FireDetector` class + overlay drawing |
| `main.py` | CLI: process video file or webcam |
| `make_synth_fire.py` | generates synthetic fire test video |
| `tests/test_fire_alarm.py` | 6 unit tests (`pytest tests/ -v`) |
| `tests/synth_fire.mp4` | synthetic fire (8 s) + no-fire (4 s) |
| `tests/real_fire.mp4` | real fireplace clip (positive test) |
| `tests/no_fire.mp4` | calm room with orange lamp (negative test) |
| `output/demo_fire_beep.mp4` | annotated demo with siren audio |

## Usage

```bash
pip install opencv-python numpy pytest

# run on a video file (saves annotated output + siren beep)
python main.py --input tests/real_fire.mp4 --output output/demo.mp4 --beep

# live webcam
python main.py --input 0 --show

# unit tests
python -m pytest tests/ -v
```

## Test results

| Test video | Expected | Result |
|---|---|---|
| Real fireplace clip (246 frames) | ALARM | ✅ alarm at frame 10 |
| Synthetic fire (120 fire frames) | ALARM | ✅ alarm at frame 10, clears after |
| Calm room + orange lamp (90 frames) | SILENT | ✅ no alarm (flicker gate) |

Unit tests: **6/6 passed**.

## Honest limitations

- Needs a clear view of the flames; heavy smoke without visible flame
  is not detected.
- Very small/distant fires may fall below the minimum-area threshold.
- Strong orange stage lighting with movement could still false-alarm —
  the flicker gate reduces but cannot fully eliminate this.
