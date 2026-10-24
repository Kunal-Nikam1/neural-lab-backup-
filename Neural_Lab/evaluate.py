"""
evaluate.py - measure accuracy of gauge_reader on the sample images, plus a
"robot-dog" stress test that simulates what a live moving camera does to the frames
(rotation, distance/scale change, motion blur, exposure change, JPEG, defocus).

  python evaluate.py              # clean images + stress test
  python evaluate.py --sweep      # sweep MASK_FRAC to see sensitivity
"""
import argparse
import os
import sys

import cv2
import numpy as np

import gauge_reader as g

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH = {"img1.png": 6.66, "img2.png": 6.68, "img3.png": 6.43, "img4.png": 5.00, "img5.png": 5.78}


def load():
    return {n: cv2.imread(os.path.join(HERE, n)) for n in TRUTH}


# ---------------- augmentations that mimic a moving camera ---------------- #
def rotate(im, deg):
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def scale(im, s):
    h, w = im.shape[:2]
    small = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR) if s < 1 else small


def zoom_out(im, s):
    """Camera further away: shrink and paste on canvas (display gets smaller in frame)."""
    h, w = im.shape[:2]
    small = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    canvas = cv2.copyMakeBorder(small, 0, h - small.shape[0], 0, w - small.shape[1],
                                cv2.BORDER_REFLECT)
    return canvas


def shift(im, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(im, M, im.shape[1::-1], borderMode=cv2.BORDER_REFLECT)


def motion_blur(im, k, angle=0):
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1)
    ker = cv2.warpAffine(ker, M, (k, k))
    ker /= max(ker.sum(), 1e-6)
    return cv2.filter2D(im, -1, ker)


def gamma(im, gm):
    return (255 * (im / 255.0) ** gm).astype(np.uint8)


def bright(im, a, b):
    return np.clip(im.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


def jpeg(im, q):
    ok, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, 1)


def defocus(im, s):
    return cv2.GaussianBlur(im, (0, 0), s)


def noise(im, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(im + rng.normal(0, sigma, im.shape), 0, 255).astype(np.uint8)


AUGS = {
    "rot -8deg": lambda im: rotate(im, -8),
    "rot +8deg": lambda im: rotate(im, 8),
    "rot +15deg": lambda im: rotate(im, 15),
    "far 0.75x": lambda im: zoom_out(im, 0.75),
    "far 0.6x": lambda im: zoom_out(im, 0.6),
    "shift (+300,+120)": lambda im: shift(im, 300, 120),
    "motion blur 5px": lambda im: motion_blur(im, 5),
    "motion blur 9px diag": lambda im: motion_blur(im, 9, 30),
    "defocus s=1.5": lambda im: defocus(im, 1.5),
    "darker (0.6x)": lambda im: bright(im, 0.6, 0),
    "brighter (+40)": lambda im: bright(im, 1.0, 40),
    "gamma 1.6": lambda im: gamma(im, 1.6),
    "jpeg q25": lambda im: jpeg(im, 25),
    "noise s=8": lambda im: noise(im, 8),
    "rot+far+blur": lambda im: motion_blur(zoom_out(rotate(im, 6), 0.7), 5),
}


DETAIL = False


def read(im):
    res = g.process_frame(im)
    return res.reading if res.reading.ok else None, res.status


def run_clean(verbose=True):
    ims = load()
    ok = 0
    for n, im in ims.items():
        rd, st = read(im)
        good = rd is not None and abs(rd.value - TRUTH[n]) < 1e-6
        ok += good
        if verbose:
            got = f"{rd.value:.2f}" if rd else "none"
            print(f"  {n}: truth {TRUTH[n]:.2f}  got {got}  conf {rd.conf if rd else 0:.2f}  "
                  f"{st:16s} {'OK' if good else 'WRONG'}")
    return ok, len(ims)


def run_stress():
    ims = load()
    tot_ok = tot = 0
    print(f"\n{'condition':24s} correct   no-read   wrong   (wrong = a confident-looking bad value)")
    for name, fn in AUGS.items():
        ok = nr = wr = 0
        for n, im in ims.items():
            rd, st = read(fn(im))
            if rd is None:
                nr += 1
            elif abs(rd.value - TRUTH[n]) < 1e-6:
                ok += 1
            else:
                wr += 1
                if DETAIL:
                    print(f"      wrong: {name} {n} truth {TRUTH[n]:.2f} got {rd.value:.2f} conf {rd.conf:.2f}")
        tot_ok += ok
        tot += len(ims)
        print(f"{name:24s} {ok}/5       {nr}/5       {wr}/5")
    print(f"\nstress total: {tot_ok}/{tot} exactly correct")


def run_stream_sim(n_frames=60, seed=1):
    """Simulate a robot-dog feed: every frame gets a random degradation (some unreadable).
    Checks the *fused* StreamReader output, not single frames."""
    rng = np.random.default_rng(seed)
    ims = load()
    names = list(AUGS)
    print("\nStream simulation (random degradation per frame, 10 fps timeline):")
    tot_ok = tot_wrong_status = tot_unknown = tot = 0
    for n, im in ims.items():
        sr = g.StreamReader()
        ok = unk = bad_status = cnt = 0
        for i in range(n_frames):
            aug = None if rng.random() < 0.25 else AUGS[names[rng.integers(len(names))]]
            frame = aug(im) if aug else im
            val, conf, status, _ = sr.update(frame, now=i * 0.1)
            if i < 8:                       # warm-up
                continue
            cnt += 1
            if val is None:
                unk += 1
            elif abs(val - TRUTH[n]) < 1e-6:
                ok += 1
            if val is not None and status != g.classify(TRUTH[n]):
                bad_status += 1
        tot_ok += ok
        tot_unknown += unk
        tot_wrong_status += bad_status
        tot += cnt
        print(f"  {n}: exact value {ok}/{cnt}   unknown {unk}/{cnt}   wrong status {bad_status}/{cnt}")
    print(f"  -> exact {tot_ok}/{tot}, unknown {tot_unknown}/{tot}, wrong LOW/NORMAL/HIGH {tot_wrong_status}/{tot}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    DETAIL = a.detail
    if a.sweep:
        for f in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            g.MASK_FRAC = f
            ok, n = run_clean(verbose=False)
            ims = load()
            confs = [read(im)[0].conf if read(im)[0] else 0 for im in ims.values()]
            print(f"MASK_FRAC {f:.2f}: {ok}/{n} correct   conf = {[round(c, 2) for c in confs]}")
    else:
        print("Clean images:")
        ok, n = run_clean()
        print(f"  -> {ok}/{n} correct")
        run_stress()
        run_stream_sim()
