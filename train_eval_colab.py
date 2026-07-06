#!/usr/bin/env python3
# ============================================================
# TRAIN + EVALUATE (Google Colab / GPU)  -- blind binaural speech-intelligibility regressor
#
# WHY THIS IS FAST NOW: features are loaded into RAM ONCE per mode (the old version re-read
# every .npz from disk every epoch -> that, not compute, caused ~1 h/fold). GPU is used if present.
#
# TRAINING here is offline and GPU is fine -- it does NOT affect any "CPU/IoT" claim.
# The IoT claim is INFERENCE cost: use benchmark_cpu_inference() on the trained model (below).
#
# ---- COLAB SETUP ----
#   from google.colab import drive; drive.mount('/content/drive')
#   # Colab already has torch. Then:
#   !python train_eval_colab.py /content/drive/MyDrive/.../index_binaural.csv \
#                               /content/drive/MyDrive/.../index_mono.csv --folds 10
#   (paths inside the CSVs must be readable from Colab -- if they are Windows F:\ paths,
#    regenerate the indices on Colab, or rewrite the 'path' column to the Drive location.)
# ============================================================
import os, csv, re, argparse, time
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, wilcoxon

GOLD = {'6','7','8','13','67'}
def room_num(r): return r.split('_')[0]
def parse_rt(r):
    m=re.search(r'_([\d.]+)\s*$', r); return float(m.group(1)) if m else float('nan')
def regime(rt): return 'dry' if rt<0.7 else '0.7-1.0' if rt<1.0 else 'transition' if rt<2.5 else 'reverberant'

def load_index(p):
    rows=list(csv.DictReader(open(p)))
    for r in rows: r['label']=float(r['label']); r['rt']=parse_rt(r['room'])
    return rows

def assign_folds(rooms_rt, n_folds, seed=0):
    rng=np.random.default_rng(seed); fo={}; by=defaultdict(list)
    for room,rt in rooms_rt.items(): by[regime(rt)].append(room)
    for _,rooms in by.items():
        rooms=list(rooms); rng.shuffle(rooms)
        for i,room in enumerate(rooms): fo[room]=i%n_folds
    return fo

# ---- load ALL features into RAM: fast path (packed .npz) or slow path (per-file index.csv) ----
def load_all(path_or_index):
    if path_or_index.endswith('.npz'):                      # fast path: pack_features.py output
        print(f"  loading packed file {path_or_index} (single read, seconds not hours)...")
        z = np.load(path_or_index, allow_pickle=True)
        return z['mel'], z['ia'], z['label'], z['room'], z['mel'].shape[1]
    rows = load_index(path_or_index)                         # slow path: many small .npz (see warning below)
    n=len(rows); z0=np.load(rows[0]['path']); C,M,T=z0['mel'].shape; D=z0['ia'].shape[0]
    mel=np.empty((n,C,M,T),np.float16); ia=np.empty((n,D),np.float32); y=np.empty(n,np.float32)
    rooms=np.empty(n,object)
    print(f"  [SLOW PATH] reading {n} individual .npz files. If this is >>ms/file, your paths are likely")
    print(f"  hitting a Drive-mounted filesystem -- copy data to local disk first, or run pack_features.py once.")
    for i,r in enumerate(rows):
        z=np.load(r['path']); mel[i]=z['mel']; ia[i]=z['ia']; y[i]=r['label']; rooms[i]=r['room']
        if i%2000==0: print(f"    loaded {i}/{n}")
    return mel, ia, y, rooms, C

def run_cv(index_path, fold_of, in_ch, epochs=40, batch=128, lr=1e-3, patience=6, seed=0, device=None):
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  device: {device}")
    if device=='cpu':
        print("  [WARNING] on CPU despite a GPU runtime? torch does not see the GPU. Check Runtime>Change")
        print("  runtime type = GPU, and that torch.cuda.is_available() returns True. A bigger GPU won't help this.")
    torch.manual_seed(seed); np.random.seed(seed)
    mel, ia, y, rooms, C = load_all(index_path)            # RAM cache ONCE (packed .npz = fast path)
    assert C==in_ch, f"index has {C}-ch mel but in_ch={in_ch}"

    class Net(nn.Module):
        def __init__(self, ic, nia=128):
            super().__init__()
            self.pre = nn.AvgPool2d((1,2))   # halve time frames 374->187: ~2x cheaper early convs
            b=lambda i,o: nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),nn.MaxPool2d(2),nn.Dropout(0.1))
            self.cnn=nn.Sequential(b(ic,16),b(16,32),b(32,64),nn.AdaptiveAvgPool2d(1))
            self.mlp=nn.Sequential(nn.Linear(nia,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,32),nn.ReLU())
            self.head=nn.Sequential(nn.Linear(96,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,1))
        def forward(self,m,a): return self.head(torch.cat([self.cnn(self.pre(m)).flatten(1),self.mlp(a)],1)).squeeze(1)

    n_folds=max(fold_of.values())+1; per_room={}
    fid=np.array([fold_of[r] for r in rooms])
    for f in range(n_folds):
        tr=np.where(fid!=f)[0]; te=np.where(fid==f)[0]
        if len(te)==0: continue
        # leakage-free standardization from TRAIN (fast, in RAM)
        mtr=mel[tr].astype(np.float32); mmean,mstd=mtr.mean(),mtr.std()+1e-6
        iamean,iastd=ia[tr].mean(0),ia[tr].std(0)+1e-6
        def mk(idx):
            X=(mel[idx].astype(np.float32)-mmean)/mstd
            A=(ia[idx]-iamean)/iastd
            return TensorDataset(torch.from_numpy(X),torch.from_numpy(A),torch.from_numpy(y[idx]))
        # early-stopping val = 10% of train samples
        rng=np.random.default_rng(seed); perm=rng.permutation(len(tr)); nv=max(1,len(tr)//10)
        vi,ti=tr[perm[:nv]],tr[perm[nv:]]
        dtr=DataLoader(mk(ti),batch_size=batch,shuffle=True,pin_memory=(device=='cuda'))
        dva=DataLoader(mk(vi),batch_size=256,pin_memory=(device=='cuda'))
        dte=DataLoader(mk(te),batch_size=256)
        net=Net(in_ch).to(device); opt=torch.optim.Adam(net.parameters(),lr=lr); lf=nn.HuberLoss(delta=1.0)
        scaler=torch.cuda.amp.GradScaler(enabled=(device=='cuda'))
        best=1e9; bad=0; best_state=None
        for ep in range(epochs):
            t_ep=time.perf_counter(); net.train()
            for m,a,t in dtr:
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                    l=lf(net(m.to(device),a.to(device)),t.to(device))
                scaler.scale(l).backward(); scaler.step(opt); scaler.update()
            net.eval(); vs=[]
            with torch.no_grad():
                for m,a,t in dva:
                    with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                        vs.append(lf(net(m.to(device),a.to(device)),t.to(device)).item())
            v=np.mean(vs)
            if f==0 and ep<3:
                print(f"    [TIMING] fold1 epoch {ep+1}: {time.perf_counter()-t_ep:.1f}s "
                      f"({len(ti)} train samples, batch {batch}, device {device})")
            if v<best-1e-4: best=v; bad=0; best_state={k:vv.cpu().clone() for k,vv in net.state_dict().items()}
            else:
                bad+=1
                if bad>=patience: break
        if best_state: net.load_state_dict(best_state)
        net.eval(); preds=[]
        with torch.no_grad():
            for m,a,t in dte: preds.append(net(m.to(device),a.to(device)).cpu().numpy())
        preds=np.concatenate(preds)
        for j,gi in enumerate(te):
            per_room.setdefault(rooms[gi],{'y_true':[],'y_pred':[],'rt':parse_rt(rooms[gi])})
            per_room[rooms[gi]]['y_true'].append(float(y[gi])); per_room[rooms[gi]]['y_pred'].append(float(preds[j]))
        print(f"  fold {f+1}/{n_folds}: {len(te)} test, stopped ep {ep+1}")
    for r in per_room:
        per_room[r]['y_true']=np.array(per_room[r]['y_true']); per_room[r]['y_pred']=np.array(per_room[r]['y_pred'])
    # return the last-fold trained net too, for the CPU inference benchmark
    return per_room, net

def _corr(fn,a,b):
    a,b=np.asarray(a),np.asarray(b)
    return np.nan if len(a)<3 or np.std(a)<1e-9 or np.std(b)<1e-9 else fn(a,b)[0]
def rmet(yt,yp): return dict(mae=np.mean(np.abs(yt-yp)),rmse=np.sqrt(np.mean((yt-yp)**2)),
                             pcc=_corr(pearsonr,yt,yp),spr=_corr(spearmanr,yt,yp))
def report(pr,tag):
    rooms=sorted(pr); rm={r:rmet(pr[r]['y_true'],pr[r]['y_pred']) for r in rooms}
    mac=lambda k:np.nanmean([rm[r][k] for r in rooms])
    at=np.concatenate([pr[r]['y_true'] for r in rooms]); ap=np.concatenate([pr[r]['y_pred'] for r in rooms])
    print(f"\n===== {tag} =====")
    print(f"MACRO ({len(rooms)} rooms): MAE {mac('mae'):.3f} RMSE {mac('rmse'):.3f} PCC {mac('pcc'):.3f} Spearman {mac('spr'):.3f}")
    print(f"MICRO (pooled): MAE {np.mean(np.abs(at-ap)):.3f} PCC {_corr(pearsonr,at,ap):.3f}  [big rooms dominate - secondary]")
    for reg in ['dry','0.7-1.0','transition','reverberant']:
        rs=[r for r in rooms if regime(pr[r]['rt'])==reg]
        if rs: print(f"   {reg:12} rooms={len(rs):>3} MAE {np.nanmean([rm[r]['mae'] for r in rs]):.3f}")
    gr=[r for r in rooms if room_num(r) in GOLD]
    if gr: print(f"GOLD (n={len(gr)}): MAE {np.nanmean([rm[r]['mae'] for r in gr]):.3f}")
    return {r:rm[r]['mae'] for r in rooms}
def paired_ablation(mb,mm):
    rooms=sorted(set(mb)&set(mm)); d=np.array([mm[r]-mb[r] for r in rooms])
    gold=[r for r in rooms if room_num(r) in GOLD]; dg=np.array([mm[r]-mb[r] for r in gold])
    print("\n===== ABLATION binaural vs mono (paired per room) =====")
    print(f"ALL (n={len(rooms)}): mean MAE reduction {d.mean():+.3f} dB, binaural better {int((d>0).sum())}/{len(rooms)}")
    if len(rooms)>=6 and np.any(d!=0):
        try: print(f"   Wilcoxon p={wilcoxon(d).pvalue:.4f}")
        except Exception: pass
    if gold: print(f"GOLD (n={len(gold)}): mean MAE reduction {dg.mean():+.3f} dB, better {int((dg>0).sum())}/{len(gold)}")

# ---- IoT narrative: INFERENCE cost of the trained model on CPU ----
def benchmark_cpu_inference(net, in_ch=2, n=200):
    import torch
    net=net.to('cpu').eval()
    mel=torch.randn(1,in_ch,64,374); ia=torch.randn(1,128)
    with torch.no_grad():
        for _ in range(10): net(mel,ia)                 # warmup
        t0=time.perf_counter()
        for _ in range(n): net(mel,ia)
        dt=(time.perf_counter()-t0)/n
    params=sum(p.numel() for p in net.parameters())
    size_mb=sum(p.numel()*p.element_size() for p in net.parameters())/1e6
    print("\n===== CPU INFERENCE (the IoT claim; training was offline on GPU) =====")
    print(f"forward-pass latency (CPU, batch=1): {dt*1000:.2f} ms/sample")
    print(f"params: {params:,}  |  model size (fp32): {size_mb:.2f} MB")
    print("NOTE: end-to-end = feature extraction (synth_features timing) + this forward pass.")

def rooms_from(path_or_index):
    """Room list for fold assignment: works whether input is a packed .npz or a per-file index.csv."""
    if path_or_index.endswith('.npz'):
        z = np.load(path_or_index, allow_pickle=True)
        rt = {}
        for room in np.unique(z['room']): rt[str(room)] = parse_rt(str(room))
        return rt
    return {r['room']: r['rt'] for r in load_index(path_or_index)}

def main():
    import pickle
    ap=argparse.ArgumentParser()
    ap.add_argument('index_binaural'); ap.add_argument('index_mono')
    ap.add_argument('--folds',type=int,default=10); ap.add_argument('--epochs',type=int,default=40)
    ap.add_argument('--seed',type=int,default=0)
    a=ap.parse_args()
    print(f"########## SEED {a.seed} | FOLDS {a.folds} ##########")
    rr=rooms_from(a.index_binaural)
    fo=assign_folds(rr,a.folds,seed=a.seed)
    pb,net_b=run_cv(a.index_binaural,fo,in_ch=2,epochs=a.epochs,seed=a.seed); mb=report(pb,"BINAURAL")
    pickle.dump(pb, open(f'results_binaural_seed{a.seed}_folds{a.folds}.pkl','wb'))  # bank it (long-run safety)
    pm,_=run_cv(a.index_mono,fo,in_ch=1,epochs=a.epochs,seed=a.seed); mm=report(pm,"MONO")
    pickle.dump(pm, open(f'results_mono_seed{a.seed}_folds{a.folds}.pkl','wb'))
    paired_ablation(mb,mm)
    benchmark_cpu_inference(net_b, in_ch=2)

if __name__=='__main__': main()
