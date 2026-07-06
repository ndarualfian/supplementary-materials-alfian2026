#!/usr/bin/env python3
# ============================================================
# BLIND-REVERBERATION BASELINE
# The decisive control for the "this is just blind-RT rebranded" critique.
# Same input as the deep model (the saved log-mel), same room-grouped CV, but the CNN is
# replaced by ~8 hand-crafted blind-reverberation descriptors + Ridge regression.
#
# READ THE RESULT LIKE THIS:
#   if this baseline's MACRO PCC ~ the CNN's (0.56) -> the deep model adds little (a problem).
#   if it is clearly worse                          -> the CNN's learned processing adds value.
#
# Uses the SAME folds as train_eval_colab.py (imports assign_folds) -> apples-to-apples.
# Run on the MONO index (reverberation is monaural): the honest "coarse reverberation" baseline.
#
# Usage:  python baseline_reverb.py index_mono_colab.csv --folds 5
# ============================================================
import argparse, numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
# these do NOT require torch (torch is imported only inside train_eval_colab.run_cv):
from train_eval_colab import load_index, parse_rt, regime, assign_folds, report, GOLD, room_num

FRAME_RATE = 48000/512        # mel hop=512 @ 48 kHz -> temporal-envelope sample rate (~93.75 Hz)
EPS = 1e-8

def reverb_features(mel):
    """~8 blind reverberation descriptors from a saved log-mel [C,64,T] (downmixed to mono)."""
    if mel.ndim == 3: mel = mel.mean(0)          # [C,64,T] -> mono [64,T]
    mel = mel.astype(np.float32)
    B, T = mel.shape
    # temporal envelope statistics (reverberation smears modulation -> lower dynamics)
    band_std = mel.std(1)                                   # per-band temporal std
    dyn = np.percentile(mel,95,axis=1) - np.percentile(mel,5,axis=1)   # per-band dynamic range
    flux = np.mean(np.abs(np.diff(mel, axis=1)))           # spectral/temporal flux (reverb -> smoother)
    # modulation spectrum per band (reverb -> energy shifts to LOW modulation freqs)
    env = mel - mel.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(env, axis=1))**2                # [64, F]
    fmod = np.fft.rfftfreq(T, d=1.0/FRAME_RATE)
    lo = (fmod>=1)&(fmod<4); hi=(fmod>=4)&(fmod<16); vhi=(fmod>=8)&(fmod<16)
    lo_e = P[:,lo].sum(1); hi_e = P[:,hi].sum(1); vhi_e = P[:,vhi].sum(1); tot=P.sum(1)+EPS
    lohi = np.mean(lo_e/(hi_e+EPS))                         # low/high modulation ratio (reverb up)
    vhi_frac = np.mean(vhi_e/tot)                           # fast-modulation fraction (reverb down)
    # spectral shape
    lin = np.exp(mel)                                       # back to (relative) linear energy
    centroid = np.mean((np.arange(B)[:,None]*lin).sum(0)/(lin.sum(0)+EPS))   # temporal-mean spectral centroid
    tilt = lin[B//2:].mean()/(lin[:B//2].mean()+EPS)        # HF/LF energy ratio
    # temporal skew of overall energy envelope (reverb fills valleys -> shifts/asymmetry)
    tenv = lin.mean(0); tenv = (tenv-tenv.mean())/(tenv.std()+EPS)
    skew = float(np.mean(tenv**3))
    return np.array([band_std.mean(), dyn.mean(), flux, np.log(lohi+EPS),
                     vhi_frac, centroid, tilt, skew], np.float32)

FEAT_NAMES = ['env_std','dyn_range','flux','log_lo/hi_mod','fast_mod_frac','spec_centroid','hf/lf_tilt','env_skew']

def load_features(index_path):
    rows = load_index(index_path)
    X = np.empty((len(rows), 8), np.float32); y=np.empty(len(rows),np.float32); rooms=np.empty(len(rows),object)
    for i,r in enumerate(rows):
        z = np.load(r['path']); X[i]=reverb_features(z['mel']); y[i]=r['label']; rooms[i]=r['room']
        if i%3000==0: print(f"  features {i}/{len(rows)}")
    return X, y, rooms

def run_baseline(index_path, fold_of, alpha=1.0):
    X, y, rooms = load_features(index_path)
    fid = np.array([fold_of[r] for r in rooms]); per_room={}
    for f in sorted(set(fid)):
        tr=np.where(fid!=f)[0]; te=np.where(fid==f)[0]
        if len(te)==0: continue
        sc=StandardScaler().fit(X[tr])                     # leakage-free: fit on train only
        m=Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr])
        pred=m.predict(sc.transform(X[te]))
        for j,gi in enumerate(te):
            per_room.setdefault(rooms[gi],{'y_true':[],'y_pred':[],'rt':parse_rt(rooms[gi])})
            per_room[rooms[gi]]['y_true'].append(float(y[gi])); per_room[rooms[gi]]['y_pred'].append(float(pred[j]))
    for r in per_room:
        per_room[r]['y_true']=np.array(per_room[r]['y_true']); per_room[r]['y_pred']=np.array(per_room[r]['y_pred'])
    # which single feature correlates most with the label? (sharpens the "one number" critique)
    from scipy.stats import pearsonr
    print("\nsingle-feature |PCC| with label (blind reverberation power in ONE number):")
    for k in np.argsort([-abs(pearsonr(X[:,i],y)[0]) for i in range(8)])[:4]:
        print(f"   {FEAT_NAMES[k]:16} |PCC| {abs(pearsonr(X[:,k],y)[0]):.3f}")
    return per_room

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('index'); ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--seed',type=int,default=0)
    a=ap.parse_args()
    print(f"########## BASELINE | SEED {a.seed} | FOLDS {a.folds} ##########")
    rr={r['room']:r['rt'] for r in load_index(a.index)}
    fo=assign_folds(rr, a.folds, seed=a.seed)              # MUST match the deep run's --folds and --seed
    pr=run_baseline(a.index, fo)
    report(pr, "BLIND-REVERBERATION BASELINE (Ridge on 8 reverberation descriptors)")
    print("\n>>> COMPARE to your deep model (macro): binaural MAE 1.671 PCC 0.562 | mono MAE 1.805 PCC 0.406")
    print(">>> If this baseline's macro PCC ~0.56, the CNN is redundant. If clearly lower, the CNN earns its place.")

if __name__=='__main__': main()
