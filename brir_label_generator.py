# ============================================================
# BRIR LABEL GENERATOR  (Decision B2: better-ear U/D, reverberation-only)
# Produces ONE continuous scalar per BRIR = the regression LABEL.
#
# This is NOT the full Leclere model: it implements the better-ear U/D branch
# (onset detection + linear early/late windows + per-band U/D ratio, SII-weighted,
# summed to a broadband dB scalar). It intentionally OMITS the binaural-unmasking
# (BMLD) term, which in reverberation-only contributes only ~1 dB and requires the
# Lavandier et al. 2012 Eq.1.2 not available here. Higher value = more intelligible.
#
# Role: this file LABELS the BRIRs. The blind predictor (separate) will see only
# reverberant speech (BRIR * dry speech) and regress this scalar. Never mix them.
# ============================================================

import os, glob, re, csv
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, resample_poly

# ---------------- CONFIG ----------------
TARGET_SR = 48000            # all BRIRs resampled to this (data has 44.1/48/96 kHz)
ELL_MS    = 30.0             # early/late limit (flat part), Leclere RI
DD_MS     = 25.0             # decay duration (linear ramp), Leclere RI
# Octave-band centers (Hz). NOTE: Leclere uses a gammatone bank; octave bands are a
# runnable approximation for the better-ear U/D. Swap to gammatone if you want closer fidelity.
OCT_CENTERS = [125, 250, 500, 1000, 2000, 4000]
# SII-style octave-band importance weights (APPROXIMATE, normalized, peak at 1-2 kHz).
# VERIFY against ANSI S3.5-1997 if you need exact values; they are swappable here.
SII_W = np.array([0.06, 0.13, 0.19, 0.24, 0.24, 0.14]); SII_W = SII_W / SII_W.sum()
EPS = 1e-12

# ---------------- 1) load + resample ----------------
def load_brir(path, target_sr=TARGET_SR):
    y, sr = sf.read(path, always_2d=True)          # (n, ch)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)                 # mono -> duplicate (no binaural info)
    y = y.astype(np.float64)
    if sr != target_sr:
        from math import gcd
        g = gcd(int(sr), int(target_sr))
        y = resample_poly(y, target_sr // g, sr // g, axis=0)
    return y, target_sr, sr

# ---------------- 2) direct-sound onset (Leclere 25% rule) ----------------
def find_direct_index(x, rel_db=-20.0):
    """Robust direct-sound onset = first sample within rel_db of the global peak.
    Tolerant to leading silence/noise/propagation delay (which sit far below -20 dB of
    the direct peak). Replaces Leclere's 25%-running-max rule, which fires on the first
    noise blip when a BRIR has a leading delay, corrupting the early/late split."""
    a = np.abs(x)
    pk = a.max()
    if pk <= 0:
        return 0
    thr = pk * (10.0 ** (rel_db / 20.0))
    return int(np.argmax(a > thr))

def direct_index_stereo(y):
    return int(min(find_direct_index(y[:, 0]), find_direct_index(y[:, 1])))

# ---------------- 3) noise-floor truncation (Lundeby-lite) ----------------
def noise_floor_end(h, sr):
    e = h ** 2
    win = max(1, int(0.005 * sr))
    sm = np.convolve(e, np.ones(win) / win, "same")
    nf = np.mean(sm[int(0.9 * len(sm)):]) + EPS      # noise power from last 10%
    above = np.where(sm > nf * 10 ** 0.5)[0]         # +5 dB over noise floor
    return int(above[-1]) if len(above) else len(h)

# ---------------- 4) linear complementary early/late windows ----------------
def early_late_windows(n, sr, d0, ell_ms=ELL_MS, dd_ms=DD_MS):
    ell = int(round(ell_ms / 1000.0 * sr))
    dd  = int(round(dd_ms  / 1000.0 * sr))
    flat_end  = min(n, d0 + ell)
    decay_end = min(n, flat_end + dd)
    w_e = np.zeros(n)
    w_e[d0:flat_end] = 1.0                            # before direct = 0, flat part = 1
    if decay_end > flat_end and dd > 0:
        w_e[flat_end:decay_end] = np.linspace(1.0, 0.0, decay_end - flat_end, endpoint=False)
    return w_e, 1.0 - w_e

# ---------------- 5) band filtering ----------------
def _sos(fc, sr, order=4):
    lo, hi = fc / np.sqrt(2), min(fc * np.sqrt(2), 0.999 * sr / 2)
    if hi <= lo: return None
    return butter(order, [lo, hi], btype="band", fs=sr, output="sos")

def band_energy(x, sos):
    xf = sosfiltfilt(sos, x) if sos is not None else x
    return float(np.sum(xf * xf) + EPS)

# ---------------- 6) better-ear U/D label ----------------
def better_ear_ud(path, target_sr=TARGET_SR, rt_s=None):
    y, sr, orig_sr = load_brir(path, target_sr)
    n0 = y.shape[0]
    # truncate both channels at the (max of L/R) noise-floor knee
    end = max(noise_floor_end(y[:, 0], sr), noise_floor_end(y[:, 1], sr))
    y = y[:end]
    n = y.shape[0]
    d0 = direct_index_stereo(y)
    w_e, w_l = early_late_windows(n, sr, d0)
    yE = y * w_e[:, None]
    yL = y * w_l[:, None]

    ud_be_bands, ud_mono_bands = [], []
    for fc in OCT_CENTERS:
        sos = _sos(fc, sr)
        eE_L, eL_L = band_energy(yE[:, 0], sos), band_energy(yL[:, 0], sos)
        eE_R, eL_R = band_energy(yE[:, 1], sos), band_energy(yL[:, 1], sos)
        ud_L = 10 * np.log10(eE_L / eL_L)
        ud_R = 10 * np.log10(eE_R / eL_R)
        ud_be_bands.append(max(ud_L, ud_R))                      # better ear
        # diotic/mono reference for the binaural-advantage diagnostic
        m = 0.5 * (y[:, 0] + y[:, 1])
        eE_m = band_energy((m * w_e), sos); eL_m = band_energy((m * w_l), sos)
        ud_mono_bands.append(10 * np.log10(eE_m / eL_m))

    label   = float(np.dot(SII_W, ud_be_bands))                  # broadband better-ear U/D (dB) = LABEL
    ud_mono = float(np.dot(SII_W, ud_mono_bands))
    dur = n0 / sr
    gate = "OK" if (rt_s is None or dur >= 1.5 * rt_s) else ("SHORT" if dur >= rt_s else "TOO_SHORT")
    return {
        "label_better_ear_ud_dB": round(label, 3),
        "ud_diotic_dB": round(ud_mono, 3),
        "binaural_gain_dB": round(label - ud_mono, 3),   # better-ear advantage (diagnostic)
        "orig_sr": orig_sr, "duration_s": round(dur, 3),
        "onset_ms": round(1000 * d0 / sr, 1), "ir_length_gate": gate,
    }

# ---------------- 7) batch runner ----------------
def parse_rt_from_folder(folder_name):
    m = re.search(r"_([\d.]+)$", folder_name)
    return float(m.group(1)) if m else None

def run(data_root, out_csv):
    rows = []
    for folder in sorted(glob.glob(os.path.join(data_root, "*"))):
        if not os.path.isdir(folder): continue
        room = os.path.basename(folder)
        rt = parse_rt_from_folder(room)
        for wav in sorted(glob.glob(os.path.join(folder, "*.wav")) +
                          glob.glob(os.path.join(folder, "*.WAV"))):
            try:
                d = better_ear_ud(wav, rt_s=rt)
            except Exception as e:
                d = {"error": str(e)}
            d.update({"room": room, "rt_s": rt, "file": os.path.basename(wav)})
            rows.append(d)
    keys = ["room", "rt_s", "file", "label_better_ear_ud_dB", "ud_diotic_dB",
            "binaural_gain_dB", "duration_s", "onset_ms", "ir_length_gate", "orig_sr", "error"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow(r)
    return rows

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    run(root, "brir_labels.csv")
    print("wrote brir_labels.csv")
