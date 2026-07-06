#!/usr/bin/env python3
# Clean NaN/Inf from already-generated features (run AFTER synth_features finishes).
# Usage: python clean_features.py <out_dir>
import numpy as np, glob, os, sys, csv
d=sys.argv[1]; fixed=0; total=0
for fp in glob.glob(os.path.join(d,"feat_*.npz")):
    total+=1
    z=dict(np.load(fp))
    bad = (not np.isfinite(z["ia"]).all()) or (not np.isfinite(z["mel"]).all())
    if bad:
        z["ia"]=np.nan_to_num(z["ia"],nan=0.0,posinf=0.0,neginf=0.0)
        z["mel"]=np.nan_to_num(z["mel"],nan=0.0,posinf=0.0,neginf=0.0)
        np.savez_compressed(fp, **z); fixed+=1
print(f"scanned {total} files, cleaned {fixed} with NaN/Inf")
