
from __future__ import annotations

import argparse
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
LOW_LIMIT = 5.0      # p <  5.0  -> LOW
HIGH_LIMIT = 7.0     # p >  7.0  -> HIGH       (5.0 <= p <= 7.0 -> NORMAL)

LCD_W, LCD_H = 258, 198          # canonical rectified LCD size (aspect ~ 1.3)
MIN_LCD_WIDTH_PX = 60            # below this the digits are < ~10px tall and decoding is a guess
MIN_CONF = 0.20                  # single-frame reads below this are rejected (UNKNOWN)
MIN_SHARPNESS = 0.0              # motion-blur gate (optional; calibrate on real robot footage)

# Blue LCD back-light in OpenCV HSV (H 0..179). The floor on S/V rejects the dim
# bezel buttons and dark-blue hoses.
HSV_LO = (88, 150, 175)
HSV_HI = (118, 255, 255)

# Top-line digit cells inside the rectified LCD: (x0, width) for each of X . X X
# and the shared row (y0, height).  Measured on the real display.
DIGIT_Y0, DIGIT_H = 38, 82
DIGIT_W = 48
DIGIT_X0 = (73, 137, 191)
MASK_FRAC = 0.30                 # default stroke threshold (fraction of the brightest red strokes)
MASK_FRACS = (0.30, 0.40, 0.50)  # thresholds tried per frame; most confident decode wins
X_SEARCH = 6                     # +/- px per-cell alignment search
Y_SEARCH = 5                     # +/- px shared vertical alignment search

#   a
# f   b
#   g
# e   c
#   d
SEG_PATTERNS = {
    0: "abcdef", 1: "bc", 2: "abdeg", 3: "abcdg", 4: "bcfg",
    5: "acdfg", 6: "acdefg", 7: "abc", 8: "abcdefg", 9: "abcdfg",
}


# --------------------------------------------------------------------------- #
# 1. Display detection
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    quad: np.ndarray            # 4x2 float32 corners (tl, tr, br, bl) in frame coords
    score: float

    @property
    def bbox(self):
        x, y, w, h = cv2.boundingRect(self.quad.astype(np.int32))
        return x, y, w, h


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _candidates(frame: np.ndarray, offset=(0, 0), min_area: float = 250):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bright = frame[..., 1] > 200            # digit pixels (white / cyan) are bright in G
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        hull = cv2.convexHull(c)
        rect = cv2.minAreaRect(hull)
        (w, h) = rect[1]
        if min(w, h) < 12:
            continue
        aspect = max(w, h) / max(min(w, h), 1e-3)
        rect_fill = area / max(w * h, 1e-3)          # a real LCD is a solid rectangle
        hull_fill = area / max(cv2.contourArea(hull), 1e-3)
        if aspect > 2.0 or rect_fill < 0.70 or hull_fill < 0.80:
            continue                                  # hoses / tubes: long & thin
        m = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(m, [hull], -1, 255, -1)
        inside = m > 0