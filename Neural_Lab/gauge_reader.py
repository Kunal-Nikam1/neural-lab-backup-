
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
        digit_frac = float((bright & inside).sum()) / max(int(inside.sum()), 1)
        if not (0.04 < digit_frac < 0.60):            # must contain bright digits
            continue
        score = rect_fill * 2.0 + digit_frac * 2.0 - abs(aspect - 1.3) + min(area / 4000.0, 1.0)
        box = cv2.boxPoints(rect)
        box[:, 0] += offset[0]
        box[:, 1] += offset[1]
        out.append(Detection(_order_corners(box), float(score)))
    return out


def find_display(frame: np.ndarray, prev: Optional[Detection] = None,
                 pad: float = 1.5) -> Optional[Detection]:
    """Locate the LCD. If `prev` is given, search a window around it first (fast tracking)."""
    H, W = frame.shape[:2]
    if prev is not None:
        x, y, w, h = prev.bbox
        px, py = int(w * pad), int(h * pad)
        x0, y0 = max(x - px, 0), max(y - py, 0)
        x1, y1 = min(x + w + px, W), min(y + h + py, H)
        cands = _candidates(frame[y0:y1, x0:x1], (x0, y0))
        if cands:
            return max(cands, key=lambda d: d.score)
    cands = _candidates(frame)
    return max(cands, key=lambda d: d.score) if cands else None


# --------------------------------------------------------------------------- #
# 2. Rectification
# --------------------------------------------------------------------------- #
def rectify(frame: np.ndarray, det: Detection) -> np.ndarray:
    """Warp the detected quad to a canonical LCD_W x LCD_H upright image."""
    q = det.quad
    wv = np.linalg.norm(q[1] - q[0])
    hv = np.linalg.norm(q[3] - q[0])
    if hv > wv:                               # minAreaRect came out portrait: rotate corner order
        q = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    dst = np.array([[0, 0], [LCD_W - 1, 0], [LCD_W - 1, LCD_H - 1], [0, LCD_H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(frame, M, (LCD_W, LCD_H), flags=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------- #
# 3. Seven-segment decoding
# --------------------------------------------------------------------------- #
def digit_mask(lcd: np.ndarray, frac: Optional[float] = None):
    """Binary mask of lit strokes (white or cyan-white) vs the blue back-light.

    The red channel is the discriminator: the blue/cyan back-light and glow have R~0-30,
    while every lit stroke (white or tinted) has R >= ~100.  Threshold is relative to the
    brightest strokes so it adapts to exposure changes as the robot moves.
    """
    frac = MASK_FRAC if frac is None else frac
    r = cv2.GaussianBlur(lcd[..., 2], (3, 3), 0)
    zone = r[20:135, 55:250]
    hi = float(np.percentile(zone, 99))
    thr = max(40.0, frac * hi)
    return (r > thr).astype(np.float32), hi


def _seg_boxes(w: int, h: int):
    t = max(int(round(0.17 * h)), 6)           # horizontal bar thickness
    s = max(int(round(0.27 * w)), 6)           # vertical bar thickness
    m = int(round(0.15 * w))                   # horizontal bars skip the corners
    mid = h // 2
    return [  # order must match "abcdefg"
        (m, 0, w - m, t),                                   # a
        (w - s, t, w, mid - t // 2),                        # b
        (w - s, mid + t // 2 + 1, w, h - t),                # c
        (m, h - t, w - m, h),                               # d
        (0, mid + t // 2 + 1, s, h - t),                    # e
        (0, t, s, mid - t // 2),                            # f
        (m, mid - t // 2, w - m, mid + t // 2 + 1),         # g
    ]


_SEG_BOXES = _seg_boxes(DIGIT_W, DIGIT_H)
_SEG_AREA = np.array([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in _SEG_BOXES], np.float32)
_PAT = np.array([[1.0 if n in SEG_PATTERNS[d] else 0.0 for n in "abcdefg"] for d in range(10)],
                np.float32)                                  # (10, 7)


def _cell_scores(ii: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """Soft match score (0..1) of the cell at (x0, y0) against digits 0-9.
    `ii` is cv2.integral() of the stroke mask, so each segment's lit fraction is O(1)."""
    Hh, Ww = ii.shape[0] - 1, ii.shape[1] - 1
    if x0 < 0 or y0 < 0 or x0 + DIGIT_W > Ww or y0 + DIGIT_H > Hh:
        return np.zeros(10, np.float32)
    f = np.empty(7, np.float32)
    for k, (sx0, sy0, sx1, sy1) in enumerate(_SEG_BOXES):
        xa, xb, ya, yb = x0 + sx0, x0 + sx1, y0 + sy0, y0 + sy1
        f[k] = (ii[yb, xb] - ii[ya, xb] - ii[yb, xa] + ii[ya, xa]) / _SEG_AREA[k]
    return (_PAT @ f + (1.0 - _PAT) @ (1.0 - f)) / 7.0


@dataclass
class Reading:
    ok: bool
    digits: str = ""
    value: Optional[float] = None
    conf: float = 0.0               # 0..1, combines digit fit and best-vs-runner-up margin
    cell_conf: tuple = ()
    sharpness: float = 0.0
    reason: str = ""


def _decode_mask(mask: np.ndarray):
    ii = cv2.integral(mask).astype(np.float32)
    best = None
    for dy in range(-Y_SEARCH, Y_SEARCH + 1):
        y0 = DIGIT_Y0 + dy
        cells, total = [], 0.0
        for x0 in DIGIT_X0:
            cell_best = None
            for dx in range(-X_SEARCH, X_SEARCH + 1):
                sc = _cell_scores(ii, x0 + dx, y0)
                order = np.argsort(sc)[::-1]
                cand = (float(sc[order[0]]), int(order[0]), float(sc[order[0]] - sc[order[1]]))
                if cell_best is None or cand[0] > cell_best[0]:
                    cell_best = cand
            cells.append(cell_best)
            total += cell_best[0]
        if best is None or total > best[0]:
            best = (total, cells)
    cells = best[1]
    fit = float(np.mean([c[0] for c in cells]))          # how well strokes match a digit
    margin = min(c[2] for c in cells)                    # how unambiguous each digit is
    conf = float(np.clip(min(margin / 0.20, 1.0) * np.clip((fit - 0.70) / 0.20, 0.0, 1.0), 0, 1))
    return [c[1] for c in cells], conf, tuple(round(c[2], 2) for c in cells)


def read_digits(lcd: np.ndarray) -> Reading:
    """Decode the top line 'X.XX' from a rectified LCD image.
    Decodes at several stroke thresholds (glare / exposure change the best one) and keeps
    the most confident result."""
    best = None
    for frac in MASK_FRACS:
        mask, _ = digit_mask(lcd, frac)
        digits, conf, cc = _decode_mask(mask)
        if best is None or conf > best[1]:
            best = (digits, conf, cc)
    digits, conf, cc = best
    txt = f"{digits[0]}.{digits[1]}{digits[2]}"
    sharp = float(cv2.Laplacian(lcd[..., 2], cv2.CV_32F).var())
    return Reading(True, txt, float(txt), conf, cc, sharp)


# --------------------------------------------------------------------------- #
# 4. Classification
# --------------------------------------------------------------------------- #
def classify(value: Optional[float]) -> str:
    if value is None:
        return "UNKNOWN"
    if value < LOW_LIMIT:
        return "LOW PRESSURE"
    if value > HIGH_LIMIT:
        return "HIGH PRESSURE"
    return "NORMAL PRESSURE"


# --------------------------------------------------------------------------- #
# Single-frame pipeline
# --------------------------------------------------------------------------- #
@dataclass
class FrameResult:
    reading: Reading
    status: str
    det: Optional[Detection]
    lcd: Optional[np.ndarray] = None


def normalise_exposure(frame: np.ndarray) -> np.ndarray:
    """Brighten under-exposed frames (robot walks into shadow). Only ever brightens: a normal
    frame always contains near-white areas, so its 99.5th-percentile brightness is ~255."""
    p = float(np.percentile(frame[::8, ::8].max(axis=2), 99.5))
    if p < 215:
        return cv2.convertScaleAbs(frame, alpha=min(235.0 / max(p, 1.0), 2.2))
    return frame


def process_frame(frame: np.ndarray, prev: Optional[Detection] = None) -> FrameResult:
    frame = normalise_exposure(frame)
    det = find_display(frame, prev)
    if det is None:
        return FrameResult(Reading(False, reason="display not found"), "UNKNOWN", None)
    _, _, w, h = det.bbox
    if w < MIN_LCD_WIDTH_PX:
        return FrameResult(Reading(False, reason=f"display too small/far ({w}px)"), "UNKNOWN", det)
    lcd = rectify(frame, det)
    rd = read_digits(lcd)
    if rd.sharpness < MIN_SHARPNESS:
        rd.ok, rd.reason = False, f"blurry ({rd.sharpness:.0f})"
    elif rd.conf < MIN_CONF:
        rd.ok, rd.reason = False, f"low confidence ({rd.conf:.2f})"
    return FrameResult(rd, classify(rd.value) if rd.ok else "UNKNOWN", det, lcd)


# --------------------------------------------------------------------------- #
# 5. Live-stream layer
# --------------------------------------------------------------------------- #
class StreamReader:
    """Feed frames one by one; get a stable value + status.

    * tracks the display (searches near the last position first -> fast)
    * drops frames that are blurry / tiny / low-confidence (robot motion)
    * confidence-weighted majority vote per digit over the last `window` good frames
    * status hysteresis: class only flips after `confirm` consecutive agreeing frames
    * reports UNKNOWN if nothing good was seen for `timeout` seconds
    """

    def __init__(self, window: int = 9, min_conf: float = MIN_CONF, confirm: int = 3,
                 timeout: float = 2.0):
        self.window, self.min_conf, self.confirm, self.timeout = window, min_conf, confirm, timeout
        self.hist: deque = deque(maxlen=window)
        self.prev_det: Optional[Detection] = None
        self.last_good_t = 0.0
        self.status = "UNKNOWN"
        self._cand, self._cand_n = None, 0

    def update(self, frame: np.ndarray, now: Optional[float] = None):
        now = time.time() if now is None else now
        res = process_frame(frame, self.prev_det)
        self.prev_det = res.det                    # None -> next frame does a full search
        rd = res.reading
        if rd.ok and rd.conf >= self.min_conf:
            self.hist.append((rd.digits.replace(".", ""), rd.conf))
            self.last_good_t = now
        fused, fconf = self._fuse()
        if fused is None or now - self.last_good_t > self.timeout:
            self.status, self._cand, self._cand_n = "UNKNOWN", None, 0
            return None, 0.0, self.status, res
        new = classify(fused)
        if new == self.status:
            self._cand, self._cand_n = None, 0
        elif new == self._cand:
            self._cand_n += 1
            if self._cand_n >= self.confirm:
                self.status, self._cand, self._cand_n = new, None, 0
        else:
            self._cand, self._cand_n = new, 1
            if self.status == "UNKNOWN":           # nothing to be sticky about yet
                self.status, self._cand, self._cand_n = new, None, 0
        return fused, fconf, self.status, res

    def _fuse(self):
        if not self.hist:
            return None, 0.0
        digits, share = [], []
        for pos in (0, 1, 2):
            votes = {}
            for txt, c in self.hist:
                votes[txt[pos]] = votes.get(txt[pos], 0.0) + c
            d, v = max(votes.items(), key=lambda kv: kv[1])
            digits.append(d)
            share.append(v / sum(votes.values()))
        return float(f"{digits[0]}.{digits[1]}{digits[2]}"), float(min(share))


# --------------------------------------------------------------------------- #
# Debug drawing + CLI
# --------------------------------------------------------------------------- #
_STATUS_COLORS = {
    "NORMAL PRESSURE": (0, 200, 0),
    "LOW PRESSURE":    (0, 165, 255),
    "HIGH PRESSURE":   (0, 0, 255),
    "UNKNOWN":         (180, 180, 180),
}


def draw_overlay(frame, res: FrameResult, value=None, status=None, conf=None):
    """Draw a rich HUD overlay on *frame* (mutates a copy).

    Includes:
    * green quad around the detected display
    * translucent top banner with reading, status, and confidence bar
    * picture-in-picture inset of the rectified LCD (top-right corner)
    """
    out = frame.copy()
    h_frame, w_frame = out.shape[:2]
    stat = status or res.status or "UNKNOWN"
    color = _STATUS_COLORS.get(stat, (180, 180, 180))

    # --- green quad around display ---
    if res.det is not None:
        cv2.polylines(out, [res.det.quad.astype(np.int32)], True, (0, 255, 0), 2)

    # --- top banner ---
    banner_h = 60
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w_frame, banner_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.70, out, 0.30, 0, out)

    # reading text
    if value is not None:
        txt = f"{value:.2f} bar"
    elif res.reading.ok:
        txt = f"{res.reading.value:.2f} bar"
    else:
        txt = "-- bar"
    cv2.putText(out, txt, (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    # status pill
    status_txt = stat
    (sw, sh), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    pill_x = 300
    cv2.rectangle(out, (pill_x, 12), (pill_x + sw + 20, 12 + sh + 16), color, -1)
    cv2.putText(out, status_txt, (pill_x + 10, 12 + sh + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # confidence bar
    c = conf if conf is not None else (res.reading.conf if res.reading.ok else 0.0)
    bar_x, bar_w = pill_x + sw + 40, 140
    cv2.rectangle(out, (bar_x, 20), (bar_x + bar_w, 44), (80, 80, 80), -1)
    fill = int(bar_w * min(c, 1.0))
    cv2.rectangle(out, (bar_x, 20), (bar_x + fill, 44), color, -1)
    cv2.putText(out, f"{c:.0%}", (bar_x + bar_w + 8, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # --- LCD inset (picture-in-picture, top-right) ---
    if res.lcd is not None:
        inset_h = 80
        aspect = res.lcd.shape[1] / max(res.lcd.shape[0], 1)
        inset_w = int(inset_h * aspect)
        inset = cv2.resize(res.lcd, (inset_w, inset_h), interpolation=cv2.INTER_LINEAR)
        # bordered
        inset = cv2.copyMakeBorder(inset, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=color)
        iw, ih = inset.shape[1], inset.shape[0]
        x0 = w_frame - iw - 10
        y0 = banner_h + 8
        if x0 >= 0 and y0 + ih <= h_frame:
            out[y0:y0 + ih, x0:x0 + iw] = inset

    # --- help hint at bottom ---
    cv2.putText(out, "Q: quit   F: fullscreen", (10, h_frame - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    return out


def _debug_panel(res: FrameResult):
    if res.lcd is None:
        return None
    mask, _ = digit_mask(res.lcd)
    vis = res.lcd.copy()
    for x0 in DIGIT_X0:
        cv2.rectangle(vis, (x0, DIGIT_Y0), (x0 + DIGIT_W, DIGIT_Y0 + DIGIT_H), (0, 255, 255), 1)
    m3 = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return np.hstack([vis, m3])


def run_images(paths, debug=False):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in paths:
        if not os.path.exists(p) and os.path.exists(os.path.join(here, p)):
            p = os.path.join(here, p)            # allow running from any working directory
        frame = cv2.imread(p)
        if frame is None:
            print(f"{p}: cannot read image (file not found: {os.path.abspath(p)})")
            continue
        t0 = time.time()
        res = process_frame(frame)
        ms = (time.time() - t0) * 1000
        rd = res.reading
        if rd.ok:
            print(f"{os.path.basename(p)}: {rd.value:.2f} bar -> {res.status}   "
                  f"(conf {rd.conf:.2f}, sharp {rd.sharpness:.0f}, {ms:.0f} ms)")
        else:
            print(f"{os.path.basename(p)}: no reading ({rd.reason})")
        if debug:
            base = os.path.splitext(os.path.basename(p))[0]
            cv2.imwrite(f"debug_{base}.png", draw_overlay(frame, res))
            panel = _debug_panel(res)
            if panel is not None:
                cv2.imwrite(f"debug_{base}_lcd.png", cv2.resize(panel, None, fx=2, fy=2))


def run_stream(source, show=True):
    """Read a live stream (camera, RTSP, video file) and display the feed.

    The preview window opens by default.  Pass ``show=False`` or ``--no-show``
    on the CLI to suppress it.
    """
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise SystemExit(f"cannot open stream {source}")
    sr = StreamReader()
    n, t_start = 0, time.time()
    fullscreen = False
    win_name = "Gauge Reader – press Q to quit"
    if show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 960, 540)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        value, conf, status, res = sr.update(frame)
        n += 1
        if value is None:
            print(f"[{n:05d}] --  UNKNOWN ({res.reading.reason or 'low confidence'})")
        else:
            print(f"[{n:05d}] {value:.2f} bar  {status}  (agree {conf:.2f})")
        if show:
            vis = draw_overlay(frame, res, value, status, conf)
            cv2.imshow(win_name, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    win_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
    cap.release()
    if show:
        cv2.destroyAllWindows()
    elapsed = max(time.time() - t_start, 1e-6)
    print(f"{n} frames, {n / elapsed:.1f} fps")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*")
    ap.add_argument("--stream", help="camera index, RTSP/HTTP url or video file")
    ap.add_argument("--no-show", dest="no_show", action="store_true",
                    help="suppress the live preview window (stream mode)")
    ap.add_argument("--debug", action="store_true", help="save overlay + LCD debug images")
    a = ap.parse_args()
    if a.stream is not None:
        run_stream(a.stream, show=not a.no_show)
    elif a.images:
        run_images(a.images, a.debug)
    else:
        ap.print_help()
