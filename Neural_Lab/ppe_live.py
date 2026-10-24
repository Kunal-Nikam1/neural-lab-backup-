"""
PPE Compliance — Live Webcam Application
==========================================
Real-time hi-vis vest compliance detection using a two-stage
MediaPipe + CLIP pipeline. No dataset or training required.

Controls:
  q  — Quit
  p  — Pause / Resume
  s  — Screenshot (saves to current directory)

Author: Neural_Lab
"""

import sys
import time

import cv2
import numpy as np

from ppe_detector import PPECompliancePipeline


# ── Colour palette ────────────────────────────────────────────────────────
GREEN       = (0, 210, 80)
RED         = (0, 40, 230)
YELLOW      = (0, 220, 255)
WHITE       = (255, 255, 255)
LIGHT_GREY  = (200, 200, 200)
DARK_BG     = (30, 30, 30)
ACCENT_BLUE = (220, 160, 40)


def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=12):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

    # Corners
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Edges
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)


def draw_label(frame, text, x, y, color, font_scale=0.55, thickness=2):
    """Draw a text label with a filled background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 6

    # Background pill
    cv2.rectangle(
        frame,
        (x, y - th - pad * 2),
        (x + tw + pad * 2, y),
        color,
        -1,
    )
    # Text
    cv2.putText(
        frame, text,
        (x + pad, y - pad),
        font, font_scale, WHITE, thickness, cv2.LINE_AA,
    )


def draw_dashboard(frame, fps, total, compliant, violations, paused):
    """Draw a semi-transparent status dashboard in the top-left corner."""
    h, w = frame.shape[:2]
    dash_w, dash_h = 310, 140

    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (dash_w, dash_h), DARK_BG, -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # Title bar accent
    cv2.rectangle(frame, (8, 8), (dash_w, 38), ACCENT_BLUE, -1)
    cv2.putText(
        frame, "PPE COMPLIANCE MONITOR",
        (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, WHITE, 1, cv2.LINE_AA,
    )

    # Metrics
    y0 = 58
    gap = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.50

    cv2.putText(frame, f"FPS:        {fps:.1f}", (20, y0), font, fs, LIGHT_GREY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Persons:    {total}",   (20, y0 + gap), font, fs, LIGHT_GREY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Compliant:  {compliant}", (20, y0 + gap * 2), font, fs, GREEN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Violations: {violations}", (20, y0 + gap * 3), font, fs, RED, 1, cv2.LINE_AA)

    if paused:
        cv2.putText(
            frame, "|| PAUSED",
            (200, y0), font, 0.55, YELLOW, 2, cv2.LINE_AA,
        )


def draw_person_box(frame, r):
    """Draw bounding box + compliance label for one person result."""
    x, y, bw, bh = r["x"], r["y"], r["w"], r["h"]
    compliant = r["is_compliant"]
    conf = r["confidence"]

    if compliant:
        color = GREEN
        label = f"COMPLIANT {conf:.0%}"
        icon = "[OK]"
    else:
        color = RED
        label = f"NO VEST {conf:.0%}"
        icon = "[!!]"

    # Bounding box (rounded)
    draw_rounded_rect(frame, (x, y), (x + bw, y + bh), color, 2, radius=10)

    # Corner accents — small L-shaped brackets
    L = 18
    cv2.line(frame, (x, y), (x + L, y), color, 3)
    cv2.line(frame, (x, y), (x, y + L), color, 3)
    cv2.line(frame, (x + bw, y), (x + bw - L, y), color, 3)
    cv2.line(frame, (x + bw, y), (x + bw, y + L), color, 3)
    cv2.line(frame, (x, y + bh), (x + L, y + bh), color, 3)
    cv2.line(frame, (x, y + bh), (x, y + bh - L), color, 3)
    cv2.line(frame, (x + bw, y + bh), (x + bw - L, y + bh), color, 3)
    cv2.line(frame, (x + bw, y + bh), (x + bw, y + bh - L), color, 3)

    # Label
    draw_label(frame, f"{icon} {label}", x, y - 4, color)


def main():
    print("=" * 60)
    print("  PPE COMPLIANCE DETECTION — Hi-Vis Vest")
    print("  MediaPipe + CLIP  |  Zero-Shot  |  No Training")
    print("=" * 60)
    print()

    # ── Initialise pipeline ──────────────────────────────────────────────
    pipeline = PPECompliancePipeline(
        person_threshold=0.45,
        crop_padding=15,
    )

    # ── Open webcam ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam (index 0).")
        print("  Try changing cv2.VideoCapture(0) to 1 or 2.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nWebcam opened at {actual_w}×{actual_h}")
    print("Controls:  q = quit  |  p = pause  |  s = screenshot\n")

    # ── State ────────────────────────────────────────────────────────────
    frame_count = 0
    skip_frames = 3              # Only run inference every Nth frame
    cached_results = []
    fps = 0.0
    prev_time = time.time()
    paused = False
    screenshot_idx = 0

    WINDOW = "PPE Compliance Detection"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    # ── Main loop ────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam feed ended.")
            break

        frame_count += 1

        # ── Run pipeline (skipping frames for performance) ───────────
        if not paused and frame_count % skip_frames == 0:
            cached_results = pipeline.process(frame)

        # ── FPS calculation ──────────────────────────────────────────
        now = time.time()
        dt = max(now - prev_time, 1e-6)
        fps = 0.85 * fps + 0.15 * (1.0 / dt)
        prev_time = now

        # ── Draw results ─────────────────────────────────────────────
        compliant_count = 0
        violation_count = 0

        for r in cached_results:
            draw_person_box(frame, r)
            if r["is_compliant"]:
                compliant_count += 1
            else:
                violation_count += 1

        draw_dashboard(
            frame, fps,
            total=len(cached_results),
            compliant=compliant_count,
            violations=violation_count,
            paused=paused,
        )

        # ── Help text (bottom) ───────────────────────────────────────
        cv2.putText(
            frame,
            "q=quit  p=pause  s=screenshot",
            (10, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, LIGHT_GREY, 1, cv2.LINE_AA,
        )

        # ── Display ─────────────────────────────────────────────────
        cv2.imshow(WINDOW, frame)

        # ── Keyboard ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        elif key == ord("s"):
            fname = f"ppe_screenshot_{screenshot_idx:03d}.png"
            cv2.imwrite(fname, frame)
            print(f"Screenshot saved → {fname}")
            screenshot_idx += 1

    # ── Cleanup ──────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
