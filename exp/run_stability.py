"""Feature stability analysis: Jaccard similarity across seeds.

Usage:
  python exp/run_stability.py --results_dir exp/results/cls_digits
"""
from __future__ import annotations

import os
import json
import glob
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def load_features(csv_path):
    df = pd.read_csv(csv_path)
    features = set()
    for col in df.columns:
        for v in df[col].dropna().values:
            features.add(int(v))
    return features


def jaccard(s1, s2):
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    group_dir = os.path.join(args.results_dir, "per_group")
    feature_files = sorted(glob.glob(os.path.join(group_dir, "*_features.csv")))

    experiments = {}
    for fpath in feature_files:
        basename = os.path.basename(fpath).replace("_features.csv", "")
        meta_path = os.path.join(group_dir, f"{basename}_meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        model = meta["model"]
        dataset = meta["dataset"]
        k = meta.get("k", None)
        sw = meta.get("sparse_loss_weight", None)
        seed = meta.get("seed", 0)

        key = f"{model}_{dataset}"
        if k is not None:
            key += f"_k{k}"
        if sw is not None:
            key += f"_sw{sw}"

        features = load_features(fpath)
        if key not in experiments:
            experiments[key] = {"meta": meta, "seeds": {}}
        experiments[key]["seeds"][seed] = features

    results = []
    for key, exp_data in experiments.items():
        seeds_dict = exp_data["seeds"]
        seed_list = sorted(seeds_dict.keys())
        meta = exp_data["meta"]

        if len(seed_list) < 2:
            print(f"  SKIP {key}: only {len(seed_list)} seed(s)")
            continue

        pairs = list(itertools.combinations(seed_list, 2))
        jaccards = [jaccard(seeds_dict[s1], seeds_dict[s2]) for s1, s2 in pairs]
        avg_jaccard = np.mean(jaccards)

        all_features = [seeds_dict[s] for s in seed_list]
        union = set().union(*all_features)
        intersection = all_features[0]
        for s in all_features[1:]:
            intersection = intersection & s

        result = {
            "model": meta["model"],
            "dataset": meta["dataset"],
            "k": meta.get("k"),
            "sparse_weight": meta.get("sparse_loss_weight"),
            "n_seeds": len(seed_list),
            "jaccard_mean": round(avg_jaccard, 4),
            "jaccard_std": round(np.std(jaccards), 4),
            "union_features": len(union),
            "intersection_features": len(intersection),
            "avg_features_per_seed": round(np.mean([len(f) for f in all_features]), 1),
        }
        results.append(result)
        print(f"  {key}: jaccard={avg_jaccard:.4f}  union={len(union)}  intersect={len(intersection)}")

    if results:
        df = pd.DataFrame(results)
        out_path = os.path.join(args.results_dir, "stability_results.csv")
        df.to_csv(out_path, index=False)
        print(f"\nStability results saved to {out_path}")

        print(f"\n  Stability ranking (by Jaccard mean):")
        for _, row in df.sort_values("jaccard_mean", ascending=False).iterrows():
            print(f"    jaccard={row['jaccard_mean']:.4f}±{row['jaccard_std']:.4f}  "
                  f"union={row['union_features']}  intersect={row['intersection_features']}  "
                  f"{row['model']}")


if __name__ == "__main__":
    main()
