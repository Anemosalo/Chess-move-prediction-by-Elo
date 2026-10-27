"""Collects every run under runs/ into flat CSV files in results/ (written by Claude; regenerate after new runs).
Outputs (run from the repo root):
  results/all_runs.csv      one row per run: settings, best checkpoint, test results, timing
  results/curves_val.csv    every evaluation: run, step, band, bits, top1
  results/curves_train.csv  train loss points: run, step, train_bits (average of 50 steps)
  results/baselines.csv     uniform / random-move baselines for every row set used
"""
import csv, glob, json, os, re, sys, time
import numpy as np

sys.path.insert(0, ".")
import train  # the user's train.py: band_paths, train_plan

OUT = "results"
os.makedirs(OUT, exist_ok=True)
W_ALL, W_BN = 459145, 6409


def jload(p):
    with open(p) as f:
        return json.load(f)


def kind_of(name):
    if name.startswith("run1"):
        return "run1"
    if name == "base":
        return "base"
    if name == "base_test":
        return "base_test"
    if name == "n0_base":
        return "n0"
    if "_lr" in name:
        return "lr_check"
    return "experiment"


def option_of(name, cfg):
    if name.startswith(("scratch", "ftall", "ftbn")):
        return name.split("_")[0]
    return {"run1": "scratch", "base": "scratch"}.get(name.split("_")[0], name)


rows, val_rows, train_rows = [], [], []
for d in sorted(glob.glob("runs/*/")):
    name = d.rstrip("/").split("/")[-1]
    cfg = jload(d + "config.json") if os.path.exists(d + "config.json") else {}
    if name.startswith("run1"):  # run 1 predates config.json
        cfg = {"data": "1000-1199", "init": "scratch", "tune": "all", "games": 46802, "epochs": 2, "lr": 1e-3,
               "seed": None, "eval_every": None, "patience": None, "also": [], "test": False, "eval_only": False}
    test = jload(d + "test.json") if os.path.exists(d + "test.json") else {}
    hist = jload(d + "history.json") if os.path.exists(d + "history.json") else None
    if hist is not None and "val" not in hist:  # run 1's old format
        hist["val"] = {"1000-1199": {"step": hist["val_step"], "bits": hist["val_bits"], "top1": hist["val_top1"]}}
    log = f"logs/{name}.log"
    log_text = open(log).read() if os.path.exists(log) else ""
    m = re.search(r"trainable weights:\s*(\d+)", log_text)
    trainable = int(m.group(1)) if m else (W_BN if cfg.get("tune") == "bn" else W_ALL)

    band = cfg.get("data", "1000-1199")
    multi = isinstance(band, list) and len(band) > 1        # trained on several bands at once
    if isinstance(band, list) and len(band) == 1:           # new single-band runs store the band in a list
        band = band[0]
    r = {"run": name, "kind": "joint" if multi else kind_of(name), "option": option_of(name, cfg),
         "train_band": "+".join(band) if multi else band,
         "init": cfg.get("init"), "tune": cfg.get("tune"), "games": cfg.get("games") if cfg.get("games") else "all",
         "seed": cfg.get("seed"), "lr": cfg.get("lr"), "max_epochs": cfg.get("epochs"), "eval_every": cfg.get("eval_every"),
         "patience": cfg.get("patience"), "trainable_weights": trainable, "eval_only": cfg.get("eval_only", False)}

    # training positions actually used (re-creates the run's game selection with its seed)
    if r["kind"] in ("experiment", "lr_check"):
        np.random.seed(cfg["seed"])
        plan = train.train_plan(band, cfg.get("games"))
        r["train_positions"] = sum(len(rows_) for _, rows_ in plan)
        r["train_games_used"] = cfg["games"] if cfg.get("games") else None
    elif name == "base":                                    # all training games of 1000-1199
        r["train_positions"] = sum(len(rows_) for _, rows_ in train.train_plan(band, None))
    elif name.startswith("run1"):                           # run 1's v1 data is gone: the count from its log
        r["train_positions"] = 2258587
    elif multi:                                             # joint data lives on the desktop: the count from its log
        m = re.search(r"([\d,]+) training positions", log_text)
        r["train_positions"] = int(m.group(1).replace(",", "")) if m else None

    if hist is not None:
        v = hist["val_mean"] if multi else hist["val"][band]  # the curve that picked the best checkpoint
        i = int(np.argmin(v["bits"]))
        r.update({"evals": len(v["step"]), "last_step": v["step"][-1], "passes_completed": len(hist["epoch_end"]),
                  "best_step": v["step"][i], "best_is_last": i == len(v["step"]) - 1,
                  "early_stopped": "Early stopping" in log_text})
        for b, vb in hist["val"].items():
            tag = b[:4]
            r[f"val_bits_{tag}_start"] = vb["bits"][0]
            r[f"val_bits_{tag}_at_best"] = vb["bits"][i] if i < len(vb["bits"]) else None
            r[f"val_top1_{tag}_at_best"] = vb["top1"][i] if i < len(vb["top1"]) else None
            r[f"val_bits_{tag}_last"] = vb["bits"][-1]
            for s, bits, t1 in zip(vb["step"], vb["bits"], vb["top1"]):
                val_rows.append({"run": name, "step": s, "band": b, "bits": bits, "top1": t1})
        for s, tb in zip(hist["step"], hist["train_bits"]):
            train_rows.append({"run": name, "step": s, "train_bits": tb})
    for b, t in test.items():
        tag = b[:4]
        r[f"test_bits_{tag}"] = t["bits"]; r[f"test_top1_{tag}"] = t["top1"]; r[f"test_n_{tag}"] = t.get("positions")
    if test:
        r["test_scope"] = "all shards" if cfg.get("test_shards", 1) >= 10 or name == "base" else "shard 000"
    start = os.path.getctime(d + "config.json") if os.path.exists(d + "config.json") else None
    # training runs end at their last history write (runs/base/test.json was copied in later); eval-only runs at test.json
    ends = [os.path.getmtime(d + "history.json")] if os.path.exists(d + "history.json") else \
           [os.path.getmtime(d + f) for f in ("test.json",) if os.path.exists(d + f)]
    if start and ends:
        r["started"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(start))
        r["minutes"] = round((max(ends) - start) / 60, 1)
    rows.append(r)

cols = []
for r in rows:
    for k in r:
        if k not in cols:
            cols.append(k)
with open(f"{OUT}/all_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
with open(f"{OUT}/curves_val.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run", "step", "band", "bits", "top1"]); w.writeheader(); w.writerows(val_rows)
with open(f"{OUT}/curves_train.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run", "step", "train_bits"]); w.writeheader(); w.writerows(train_rows)

# baselines: uniform (mean log2 K) and random legal move (mean 1/K) for every row set used
base_rows = []
for b in ("1000-1199", "2000-2199"):
    paths = train.band_paths(b)
    for split, sname in ((1, "val"), (2, "test")):
        for scope, ps in (("shard 000", paths[:1]), ("all shards", paths)):
            if split == 1 and scope == "all shards":
                continue
            K = np.concatenate([np.diff(np.load(p)["offsets"])[np.load(p)["split"] == split] for p in ps])
            games = sum(len(np.unique(np.load(p)["game_id"][np.load(p)["split"] == split])) for p in ps)
            base_rows.append({"band": b, "split": sname, "scope": scope, "positions": len(K), "games": games,
                              "uniform_bits": float(np.log2(K).mean()), "random_top1": float((1 / K).mean()),
                              "mean_legal_moves": float(K.mean())})
with open(f"{OUT}/baselines.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(base_rows[0])); w.writeheader(); w.writerows(base_rows)

print(f"{len(rows)} runs, {len(val_rows)} val points, {len(train_rows)} train points")
for b in base_rows:
    print(b)
