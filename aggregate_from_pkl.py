#!/usr/bin/env python3
# Re-compute report + ablation from two banked result files (if a run finished but you
# want to recompare, or the process died AFTER both modes saved their .pkl).
# Usage: python aggregate_from_pkl.py results_binaural_seed0_folds115.pkl results_mono_seed0_folds115.pkl
import sys, pickle
from train_eval_colab import report, paired_ablation
pb = pickle.load(open(sys.argv[1],'rb')); pm = pickle.load(open(sys.argv[2],'rb'))
mb = report(pb, "BINAURAL (from banked results)")
mm = report(pm, "MONO (from banked results)")
paired_ablation(mb, mm)
