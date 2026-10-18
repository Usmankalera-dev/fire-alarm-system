"""
AI-Based Fire Alarm System — flame detection with a regular camera.

How it works
------------
1. Each frame is converted to HSV colour space.
2. Pixels in the "flame" colour range (red -> orange -> yellow, bright and
   saturated) are segmented out.
3. Noise is removed with morphological open/close; remaining blobs are
   measured. A frame counts as "fire" only if the flame area is big enough.
4. A temporal alarm filter requires fire in K of the last N frames before the
   alarm turns ON (kills single-frame false alarms), and requires a quiet
   period before it turns back OFF.

No training data or GPU needed — pure computer vision, runs on CPU.
"""

import cv2
import numpy as np
from collections import deque


class FireDetector:
    """Detects flames in video frames and raises a temporal alarm."""

    def __init__(self,
                 min_area_ratio=0.0015,
                 alarm_need=5,
                 alarm_window=10,
                 clear_need=15,
                 flicker_window=12,
                 flicker_min=0.03):
        # HSV ranges for flame colours (OpenCV H: 0-179)
        # reds/oranges/yellows, must be bright and saturated
        self.lower_a = np.array([0, 110, 130])
        self.upper_a = np.array([30, 255, 255])
        # wrap-around deep reds (H 165-179)
        self.lower_b = np.array([165, 110, 130])
        self.upper_b = np.array([179, 255, 255])

        self.min_area_ratio = min_area_ratio  # flame px / frame px
        self.alarm_need = alarm_need          # fire frames needed...
        self.alarm_window = alarm_window      # ...out of this many...
        self.clear_need = clear_need          # quiet frames to reset alarm
        self.flicker_min = flicker_min        # flames must dance (kills static orange objects)
        self.history = deque(maxlen=alarm_window)
        self.ratio_history = deque(maxlen=flicker_window)
        self.quiet = 0
        self.alarm = False
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def segment(self, frame):
        """Return the cleaned flame mask for one BGR frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, self.lower_a, self.upper_a)
        m2 = cv2.inRange(hsv, self.lower_b, self.upper_b)
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def analyse(self, frame):
        """Detect flames in one frame.

        Returns dict with: fire (bool), ratio (float), boxes (list),
        flicker (float, 0-1).
        """
        h, w = frame.shape[:2]
        mask = self.segment(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        min_px = self.min_area_ratio * h * w
        boxes = []
        fire_px = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_px:
                x, y, bw, bh = cv2.boundingRect(c)
                boxes.append((x, y, bw, bh))
                fire_px += area
        ratio = fire_px / (h * w)
        fire = len(boxes) > 0

        # flicker score: flames dance, so the flame area jitters frame to frame
        self.ratio_history.append(ratio)
        flicker = 0.0
        if len(self.ratio_history) >= 6:
            vals = np.array(self.ratio_history)
            mean = vals.mean()
            if mean > 1e-6:
                flicker = float(np.clip(vals.std() / mean, 0.0, 1.0))

        return {"fire": fire, "ratio": ratio, "boxes": boxes,
                "flicker": flicker, "mask": mask}

    def update(self, frame):
        """Feed one frame; returns (result_dict, alarm_on_bool).

        A frame only counts toward the alarm if it has BOTH flame colours
        AND flame-like flicker — this rejects static orange objects
        (lamps, sunset posters, orange clothes).
        """
        res = self.analyse(frame)
        confirmed = res["fire"] and res["flicker"] >= self.flicker_min
        res["confirmed"] = confirmed
        self.history.append(confirmed)
        if confirmed:
            self.quiet = 0
        else:
            self.quiet += 1
        if not self.alarm:
            if sum(self.history) >= self.alarm_need:
                self.alarm = True
        else:
            if self.quiet >= self.clear_need:
                self.alarm = False
        res["alarm"] = self.alarm
        return res


def draw_overlay(frame, res):
    """Draw boxes + status banner on a frame (returns new image)."""
    out = frame.copy()
    for (x, y, bw, bh) in res["boxes"]:
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
        cv2.putText(out, "FLAME", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
    h, w = out.shape[:2]
    if res["alarm"]:
        cv2.rectangle(out, (0, 0), (w, 70), (0, 0, 255), -1)
        cv2.putText(out, "FIRE DETECTED - ALARM!", (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    else:
        cv2.rectangle(out, (0, 0), (w, 70), (0, 120, 0), -1)
        cv2.putText(out, "NO FIRE - monitoring...", (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    info = "flame: %.2f%%  flicker: %.2f" % (res["ratio"] * 100,
                                             res["flicker"])
    cv2.putText(out, info, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    return out
