#!/usr/bin/env python3
# ============================================================
# VCTK CLIP SELECTOR  (single big folder; variable, sparse per-speaker counts)
# Layout: ONE folder that directly contains p225 ... p376. Each holds a VARIABLE
# number of pXXX_<sentence>_micX.flac (often >366x2; sentence IDs are sparse,
# typically starting ~13, not 1).
#
# Rule: mic1 ONLY; DROP common sentences by ID (sentence_id <= SKIP_ID_MAX, the
# rainbow/elicitation set) -- filtered by ID, NOT by list position, so it is robust
# to variable counts and sparse numbering. Keep N_PER_SPEAKER sentences spread
# across each speaker's remaining (newspaper) range. 90 pool + 18 held-out.
#
# Usage:  python select_vctk_clips.py /path/to/BIG_FOLDER
# Output: vctk_selected_clips.csv  (speaker, role, sentence_id, filepath)
# ============================================================
import os, glob, re, csv, sys

SKIP_ID_MAX   = 24    # drop sentence IDs <= this (common rainbow/elicitation sentences)
N_PER_SPEAKER = 10

HELD_OUT = {'p226','p234','p245','p248','p253','p260','p266','p278','p282',
            'p294','p302','p311','p314','p326','p335','p345','p351','p376'}
EXCLUDED = {'p315','s5'}

def sentence_no(fp):
    base = re.sub(r'^[ps]\d+_', '', os.path.basename(fp))
    m = re.search(r'(\d+)', base)
    return int(m.group(1)) if m else 10**9

def select_mic1(flacs, spk):
    m1 = [f for f in flacs if 'mic1' in os.path.basename(f).lower()]
    if m1: return m1
    if any('mic2' in os.path.basename(f).lower() for f in flacs):
        print(f"  [WARN] {spk}: mic2 present but no mic1 -> skipped"); return []
    print(f"  [WARN] {spk}: no mic1/mic2 tag -> using ALL flac (verify)"); return flacs

def pick_even(items, n):
    if len(items) <= n: return items
    idx = [round(i*(len(items)-1)/(n-1)) for i in range(n)]
    return [items[j] for j in idx]

def main(big):
    if not os.path.isdir(big):
        print(f"[ERROR] not a folder: {big}"); sys.exit(1)
    dirs = sorted(d for d in glob.glob(os.path.join(big,'p*'))+glob.glob(os.path.join(big,'s*'))
                  if os.path.isdir(d))
    if not dirs:
        print(f"[ERROR] no p*/s* subfolders directly in {big}"); sys.exit(1)

    rows, counts, empty, fell_back = [], {'POOL':0,'HELD_OUT':0}, [], []
    for sd in dirs:
        spk = os.path.basename(sd)
        if spk in EXCLUDED: continue
        role = 'HELD_OUT' if spk in HELD_OUT else 'POOL'
        flacs = glob.glob(os.path.join(sd,'*.flac'))+glob.glob(os.path.join(sd,'*.FLAC'))
        m1 = sorted(select_mic1(flacs, spk), key=sentence_no)
        if not m1: empty.append(spk); continue
        kept = [f for f in m1 if sentence_no(f) > SKIP_ID_MAX]     # <-- drop common by ID
        if len(kept) < N_PER_SPEAKER:                              # fallback if too few remain
            kept = m1; fell_back.append(spk)
        for fp in pick_even(kept, N_PER_SPEAKER):
            rows.append({'speaker':spk,'role':role,'sentence_id':sentence_no(fp),'filepath':fp})
            counts[role]+=1

    out='vctk_selected_clips.csv'
    with open(out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['speaker','role','sentence_id','filepath']); w.writeheader()
        for r in rows: w.writerow(r)

    found={os.path.basename(d) for d in dirs}
    ids=[r['sentence_id'] for r in rows]
    print(f"\n=== VERIFY ===")
    print(f"speaker folders found: {len(found)}  (expect ~108)")
    print(f"selected clips: {len(rows)}  ->  POOL {counts['POOL']} | HELD_OUT {counts['HELD_OUT']}")
    print(f"sentence_id range: {min(ids)}..{max(ids)}   (min must be > {SKIP_ID_MAX})")
    print(f"clips with id<= {SKIP_ID_MAX} (should be 0 unless fallback): {sum(i<=SKIP_ID_MAX for i in ids)}")
    if fell_back: print(f"[WARN] fallback (<{N_PER_SPEAKER} clips after ID filter, used all): {fell_back}")
    if empty:     print(f"[WARN] 0 mic1 clips: {empty}")
    miss=HELD_OUT-found
    if miss:      print(f"[WARN] held-out not on disk: {sorted(miss)}")
    for r in rows[:3]: print("   ", r['role'], r['filepath'])
    print(f"\nwrote {out}")

if __name__=='__main__':
    if len(sys.argv)<2: print("usage: python select_vctk_clips.py /path/to/BIG_FOLDER"); sys.exit(1)
    main(sys.argv[1])
