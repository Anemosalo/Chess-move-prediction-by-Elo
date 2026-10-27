"""Measures saved models on every rating band, and probes what they learned from the rating plane (written by Claude).

Usage: python3 eval_bands.py        writes results/bands_eval.csv (the models and bands below)
       python3 eval_bands.py --models joint=runs/joint_4bands/best.pt --sweep joint --bands 800-999 1000-1199 ... --out results/bands_eval_joint.csv

Part A, true ratings: each model on the test rows (split 2) and val rows (split 1) of shard 000 of every band.
Part B, rating sweep: each sweep model on the val rows of shard 000 of every band, with the rating plane
         overwritten by one assumed rating for every position (1000, 1100, ..., 2200). If a model uses the
         rating plane, a band is predicted best when the assumed rating is close to the band's real ratings.
"""
import argparse
import csv

import numpy as np
import torch

import train

BANDS = ["1000-1199", "1200-1399", "1400-1599", "1600-1799", "1800-1999", "2000-2199"]
MODELS = {                                    # name -> weights
    "base": "runs/base/best.pt",
    "ftall_1000_s0": "runs/ftall_1000_s0/best.pt",
    "ftall_10000_s0": "runs/ftall_10000_s0/best.pt",
    "ftall_all_s0": "runs/ftall_all_s0/best.pt",
    "scratch_all_s0": "runs/scratch_all_s0/best.pt",
}
SWEEP_MODELS = ["base", "ftall_all_s0", "scratch_all_s0"]
SWEEP_RATINGS = list(range(1000, 2201, 100))
OUT = "results/bands_eval.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Measure saved models on every band, with a rating sweep")
    p.add_argument("--models", nargs="+", default=None, help="name=path pairs, e.g. joint=runs/joint_4bands/best.pt")
    p.add_argument("--bands", nargs="+", default=BANDS, help="bands to measure on")
    p.add_argument("--sweep", nargs="*", default=None, help="model names to run the rating sweep on")
    p.add_argument("--ratings", nargs="+", type=int, default=SWEEP_RATINGS, help="ratings for the sweep")
    p.add_argument("--out", default=OUT, help="csv file to write")
    p.add_argument("--data-dir", default="data", help="folder holding games_encoded/ (default data)")
    return p.parse_args()


def main():
    args = parse_args()
    train.DATA_DIR = args.data_dir
    models = dict(m.split("=", 1) for m in args.models) if args.models else MODELS
    sweep = args.sweep if args.sweep is not None else (SWEEP_MODELS if not args.models else list(models))
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    ## only the val and test rows of shard 000 of each band are kept in memory (not whole shards)
    sets = {}
    for band in args.bands:
        data = train.load_shard(train.band_paths(band)[0])
        for split_name, split_value in (("test", 2), ("val", 1)):
            small = train.take_rows(data, np.flatnonzero(data[5] == split_value))
            sets[(band, split_name)] = small
        del data

    rows_out = []
    for name, path in models.items():
        model = train.Network().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        for band in args.bands:
            for split_name in ("test", "val"):
                planes, y, flat, off, rating, split, _ = sets[(band, split_name)]
                rows = np.arange(len(y))
                K = np.diff(off)
                common = {"model": name, "band": band, "split": split_name, "positions": len(rows),
                          "uniform_bits": float(np.log2(K).mean()), "random_top1": float((1 / K).mean()),
                          "mean_rating": float(rating.mean())}
                bits, top1 = train.evaluate(model, rows, planes, y, flat, off, rating, device)
                rows_out.append({**common, "rating_mode": "true", "assumed_rating": "", "bits": bits, "top1": top1})
                if split_name == "val" and name in sweep:
                    for r in args.ratings:
                        fake = np.full_like(rating, r)                    # every position gets rating r
                        bits, top1 = train.evaluate(model, rows, planes, y, flat, off, fake, device)
                        rows_out.append({**common, "rating_mode": "fixed", "assumed_rating": r, "bits": bits, "top1": top1})
            t = [r for r in rows_out if r["model"] == name and r["band"] == band and r["split"] == "test"][0]
            print(f"{name:15s} {band}: test {t['bits']:.3f} bits, top-1 {t['top1']:.1%} (uniform {t['uniform_bits']:.3f})", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {args.out} ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
