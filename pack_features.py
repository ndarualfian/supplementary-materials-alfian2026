#!/usr/bin/env python3
# ============================================================
# PACK FEATURES: consolidate thousands of small .npz into ONE file per mode.
# Run ONCE (locally or in Colab, reading from local/unzipped folder -- not Drive-mounted).
# After packing, train_eval_colab.py loads in seconds instead of hours on every run.
#
# Usage:
#   python pack_features.py index_binaural.csv packed_binaural.npz
#   python pack_features.py index_mono.csv     packed_mono.npz
# ============================================================
import sys, csv
import numpy as np

def main(index_csv, out_path):
    rows = list(csv.DictReader(open(index_csv)))
    n = len(rows)
    z0 = np.load(rows[0]['path']); C, M, T = z0['mel'].shape; D = z0['ia'].shape[0]
    mel = np.empty((n, C, M, T), np.float16)
    ia  = np.empty((n, D), np.float32)
    y   = np.empty(n, np.float32)
    room = np.empty(n, dtype=object)
    for i, r in enumerate(rows):
        z = np.load(r['path'])
        mel[i] = z['mel']; ia[i] = z['ia']; y[i] = float(r['label']); room[i] = r['room']
        if i % 2000 == 0: print(f"  packed {i}/{n}")
    np.savez(out_path, mel=mel, ia=ia, label=y, room=room)   # uncompressed = fast to re-read
    print(f"wrote {out_path}  ({n} samples, mel{mel.shape}, ia{ia.shape})")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("usage: python pack_features.py <index.csv> <out_packed.npz>"); sys.exit(1)
    main(sys.argv[1], sys.argv[2])
