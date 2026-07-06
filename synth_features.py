#!/usr/bin/env python3
# ============================================================
# REVERBERANT-SPEECH SYNTHESIS + FEATURE EXTRACTION
# Blind side of the pipeline: the model sees only reverberant SPEECH and predicts
# the per-BRIR label (better-ear U/D) produced by brir_label_generator.py.
#
# Per (BRIR, dry clip):
#   dry mono s(t)  ->  reverberant stereo  [ s*h_L , s*h_R ]  ->  RMS-normalize
#   features:  log-mel per channel  +  interaural ILD/IC vector
#   modes:     'binaural' (2-ch mel + interaural) | 'mono' (1-ch downmix mel)  -> for ablation
#
# Inputs it consumes (on YOUR machine, where the real files live):
#   - brir_root      : folder that contains the room subfolders (as fed to the label generator)
#   - brir_labels.csv: output of brir_label_generator.py (room, file, label_better_ear_ud_dB, ...)
#   - vctk_selected_clips.csv: the dry-speech pool (role=POOL used for training clips)
#
# Output: out_dir/feat_XXXXXX.npz per sample + index.csv (room, split_group, label, mode, path)
# NOTE: mechanics validated here on sample BRIRs + a synthetic clip; run on your data for real.
# ============================================================
import os, csv, glob, argparse
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, stft, coherence, resample_poly

SR = 48000
N_FFT, HOP, N_MELS, FMIN, FMAX = 1024, 512, 64, 50, 16000
FIXED_DUR = 4.0          # seconds of reverberant speech analysed (pad/truncate)
MIN_DRY   = 2.5          # if a dry clip is shorter, it is looped to this length
EPS = 1e-10

# ---------- mel filterbank (Slaney-style triangular) ----------
def _hz2mel(f): return 2595.0*np.log10(1.0+f/700.0)
def _mel2hz(m): return 700.0*(10.0**(m/2595.0)-1.0)
def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    mpts = np.linspace(_hz2mel(fmin), _hz2mel(fmax), n_mels+2)
    fpts = _mel2hz(mpts)
    bins = np.floor((n_fft+1)*fpts/sr).astype(int)
    fb = np.zeros((n_mels, n_fft//2+1))
    for m in range(1, n_mels+1):
        l, c, r = bins[m-1], bins[m], bins[m+1]
        if c>l: fb[m-1, l:c] = (np.arange(l, c)-l)/(c-l)
        if r>c: fb[m-1, c:r] = (r-np.arange(c, r))/(r-c)
    return fb
_MEL = mel_filterbank()

# ---------- dry clip loading ----------
def load_dry(path, sr=SR):
    x, s = sf.read(path, always_2d=False)
    if x.ndim > 1: x = x.mean(axis=1)          # force mono
    x = x.astype(np.float64)
    if s != sr:
        from math import gcd; g = gcd(int(s), int(sr)); x = resample_poly(x, sr//g, s//g)
    # trim leading/trailing silence (energy threshold at -40 dB of peak)
    e = np.abs(x); thr = 0.01*e.max() if e.max()>0 else 0
    nz = np.where(e > thr)[0]
    if len(nz): x = x[nz[0]:nz[-1]+1]
    if len(x) < int(MIN_DRY*sr) and len(x) > 0:  # loop-pad short clips
        x = np.tile(x, int(np.ceil(MIN_DRY*sr/len(x))))[:int(MIN_DRY*sr)]
    x = x/(np.sqrt(np.mean(x**2))+EPS)           # rms normalize dry
    return x

# ---------- synthesis ----------
def synth(dry_mono, brir_stereo, sr=SR, dur=FIXED_DUR):
    L = fftconvolve(dry_mono, brir_stereo[:,0].astype(np.float64))
    R = fftconvolve(dry_mono, brir_stereo[:,1].astype(np.float64))
    n = int(dur*sr)
    def fit(v): return np.pad(v,(0,n-len(v)))[:n] if len(v)<n else v[:n]
    y = np.stack([fit(L), fit(R)], axis=1)
    y = y/(np.sqrt(np.mean(y**2))+EPS)           # rms normalize stereo (label is level-free)
    return y

# ---------- features ----------
def _logmel(x, sr=SR):
    _,_,Z = stft(x, fs=sr, nperseg=N_FFT, noverlap=N_FFT-HOP, boundary=None)
    P = np.abs(Z)**2                              # power [freq, frames]
    M = _MEL @ P                                  # [n_mels, frames]
    return np.log(M+EPS), M

def interaural(L, R, sr=SR):
    _, mL = _logmel(L, sr); _, mR = _logmel(R, sr)
    ild = 10.0*np.log10((mL.mean(1)+EPS)/(mR.mean(1)+EPS))      # per-mel-band ILD (dB), len n_mels
    with np.errstate(divide='ignore', invalid='ignore'):
        f, Cxy = coherence(L, R, fs=sr, nperseg=N_FFT)         # magnitude-squared coherence
    Cxy = np.nan_to_num(Cxy, nan=0.0, posinf=0.0, neginf=0.0)  # zero-power bins: coherence undefined -> 0
    ic = (_MEL @ Cxy)/(_MEL.sum(1)+EPS)                        # map coherence into mel bands, len n_mels
    return np.concatenate([ild, ic]).astype(np.float32)        # [2*n_mels]

def extract_features(rev_stereo, sr=SR, mode='binaural'):
    L, R = rev_stereo[:,0], rev_stereo[:,1]
    if mode == 'binaural':
        mel = np.stack([_logmel(L,sr)[0], _logmel(R,sr)[0]], axis=0)   # [2, n_mels, T]
        ia  = interaural(L, R, sr)                                     # [2*n_mels]
    elif mode == 'mono':
        m = 0.5*(L+R)
        mel = _logmel(m,sr)[0][None]                                   # [1, n_mels, T]
        ia  = np.zeros(2*N_MELS, np.float32)                           # no interaural info
    else:
        raise ValueError(mode)
    return mel.astype(np.float32), ia

# ---------- batch runner (memory-safe: writes each sample to disk) ----------
def run(brir_root, labels_csv, clips_csv, out_dir, K=6, mode='binaural', seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    pool = [r['filepath'] for r in csv.DictReader(open(clips_csv)) if r['role']=='POOL']
    brirs = list(csv.DictReader(open(labels_csv)))
    idx = []
    for i, b in enumerate(brirs):
        if b.get('error'): continue
        bpath = os.path.join(brir_root, b['room'], b['file'])
        try:
            h, s = sf.read(bpath, always_2d=True)
            if s != SR:
                from math import gcd; g=gcd(int(s),SR); h=resample_poly(h,SR//g,s//g,axis=0)
            if h.shape[1]==1: h=np.repeat(h,2,1)
        except Exception as e:
            print(f"[skip BRIR] {bpath}: {e}"); continue
        label = float(b['label_better_ear_ud_dB'])
        for k, cp in enumerate(rng.choice(pool, size=K, replace=False)):
            try:
                y = synth(load_dry(cp), h)
                mel, ia = extract_features(y, mode=mode)
            except Exception as e:
                print(f"[skip clip] {cp}: {e}"); continue
            fp = os.path.join(out_dir, f"feat_{i:05d}_{k}.npz")
            np.savez_compressed(fp, mel=mel, ia=ia, label=label)
            idx.append({'room':b['room'],'label':label,'mode':mode,'path':fp})
        if i % 100 == 0: print(f"  {i}/{len(brirs)} BRIRs...")
    with open(os.path.join(out_dir,'index.csv'),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['room','label','mode','path']); w.writeheader()
        for r in idx: w.writerow(r)
    print(f"done: {len(idx)} samples -> {out_dir}/index.csv")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('brir_root'); ap.add_argument('labels_csv'); ap.add_argument('clips_csv')
    ap.add_argument('out_dir'); ap.add_argument('--K', type=int, default=6)
    ap.add_argument('--mode', choices=['binaural','mono'], default='binaural')
    a = ap.parse_args()
    run(a.brir_root, a.labels_csv, a.clips_csv, a.out_dir, K=a.K, mode=a.mode)
