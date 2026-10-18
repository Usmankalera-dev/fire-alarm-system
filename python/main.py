"""
AI-Based Fire Alarm System — main entry point.

Usage:
    python main.py --input <video.mp4> --output output/result.mp4
    python main.py --input 0            # webcam (0 = default camera)
    python main.py --input tests/real_fire.mp4 --output output/demo.mp4 --beep

Processes a video file or camera feed, draws flame boxes + status banner,
and (with --beep) mixes an alarm siren into the output video's audio track.
"""

import argparse
import cv2
import numpy as np
import subprocess
import os
from fire_detector import FireDetector, draw_overlay


def add_alarm_beep(video_path, alarm_ranges, fps, out_path):
    """Mix a siren beep into the video for the frames where alarm was ON.

    alarm_ranges: list of (start_frame, end_frame) with alarm active.
    """
    import wave, struct, math, tempfile
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    sr = 22050
    dur = total / fps
    n = int(sr * dur)
    audio = np.zeros(n, dtype=np.float32)
    # siren: alternating 660/880 Hz square-ish beeps
    for (s, e) in alarm_ranges:
        a, b = int(s / fps * sr), int(e / fps * sr)
        t = np.arange(b - a) / sr
        freq = np.where((t * 2).astype(int) % 2 == 0, 660, 880)
        audio[a:b] += 0.5 * np.sign(np.sin(2 * np.pi * freq * t))
    audio = np.clip(audio, -1, 1)
    pcm = (audio * 32767).astype(np.int16)
    tmp = tempfile.mktemp(suffix=".wav")
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", tmp,
           "-c:v", "copy", "-c:a", "aac", "-shortest", out_path]
    subprocess.run(cmd, capture_output=True)
    os.remove(tmp)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="AI-Based Fire Alarm System")
    ap.add_argument("--input", required=True, help="video file or camera index")
    ap.add_argument("--output", default="output/result.mp4")
    ap.add_argument("--beep", action="store_true", help="add siren audio on alarm")
    ap.add_argument("--show", action="store_true", help="show live window")
    args = ap.parse_args()

    src = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit("Could not open input: %s" % args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    w, h = int(cap.get(3)), int(cap.get(4))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    vw = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (w, h))
    det = FireDetector()
    alarm_ranges, start, n = [], None, 0
    print("Monitoring... press Q in the preview window to stop.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        res = det.update(frame)
        if res["alarm"] and start is None:
            start = n
            print("[ALARM] Fire detected at frame %d (%.1fs)" % (n, n / fps))
        if start is not None and not res["alarm"]:
            alarm_ranges.append((start, n)); start = None
            print("[OK] Alarm cleared at frame %d" % n)
        vw.write(draw_overlay(frame, res))
        if args.show:
            cv2.imshow("AI Fire Alarm", draw_overlay(frame, res))
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    if start is not None:
        alarm_ranges.append((start, n))
    cap.release(); vw.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass  # headless OpenCV build has no GUI

    out = args.output
    if args.beep and alarm_ranges:
        beeped = args.output.replace(".mp4", "_beep.mp4")
        add_alarm_beep(args.output, alarm_ranges, fps, beeped)
        out = beeped
    print("Saved:", out)
    print("Alarm was ON for %d frame-range(s)." % len(alarm_ranges))


if __name__ == "__main__":
    main()
