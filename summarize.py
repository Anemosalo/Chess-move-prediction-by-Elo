"""Builds results/tables.md from results/all_runs.csv (written by Claude).

Usage: python3 summarize.py      (run collect_results.py first after new runs)
"""
import math

import pandas as pd

RUNS_CSV = "results/all_runs.csv"
BASELINES_CSV = "results/baselines.csv"
OUT = "results/tables.md"

OPTIONS = ["scratch", "ftall", "ftbn"]                      # column order in every table
NAMES = {"scratch": "scratch", "ftall": "full fine-tune", "ftbn": "BN + output layer"}   # ftbn: BN scale/shift (1,664) + output layer (4,745)
ALL = 40019                                                 # training games when --games is not set (2000-2199)


def load():
    """Returns (all runs, experiment runs with a numeric N column)."""
    df = pd.read_csv(RUNS_CSV)
    exp = df[df["kind"] == "experiment"].copy()
    exp["N"] = exp["games"].replace("all", ALL).astype(int)
    return df, exp


def run_value(df, run, column):
    """One number from one run's row."""
    return df.loc[df["run"] == run, column].iloc[0]


def baseline(band, split, scope):
    b = pd.read_csv(BASELINES_CSV)
    row = b[(b["band"] == band) & (b["split"] == split) & (b["scope"] == scope)].iloc[0]
    return row["uniform_bits"], row["random_top1"], int(row["positions"])


def games_in(band, split, scope):
    """Number of games behind a row set of baselines.csv."""
    b = pd.read_csv(BASELINES_CSV)
    return int(b[(b["band"] == band) & (b["split"] == split) & (b["scope"] == scope)].iloc[0]["games"])


def cell(values, fmt="{:.3f}"):
    """A list of seed values → '3.081 ± 0.006' (several seeds) or '2.834' (one seed)."""
    s = pd.Series(values)
    if len(s) == 1:
        return fmt.format(s.iloc[0])
    return f"{fmt.format(s.mean())} ± {fmt.format(s.std()).lstrip('+')}"


def grid(exp, column, fmt="{:.3f}"):
    """N × option table of one column, each cell formatted by cell()."""
    lists = exp.groupby(["N", "option"])[column].apply(list)
    table = lists.unstack("option")[OPTIONS]
    return table.map(lambda values: cell(values, fmt))


def n_label(n):
    return f"all ({ALL:,})" if n == ALL else f"{n:,}"


def md_table(header, rows):
    """header: list of column titles; rows: list of lists of text → Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def grid_md(table, bold_option=None):
    header = ["N strong games"] + [NAMES[o] for o in table.columns]
    rows = []
    for n, r in table.iterrows():
        rows.append([n_label(n)] + [f"**{r[o]}**" if o == bold_option else r[o] for o in table.columns])
    return md_table(header, rows)


def scratch_equivalent(exp, bits):
    """Games scratch would need to reach `bits`: linear interpolation in log10(N) between the measured scratch means."""
    s = exp[exp["option"] == "scratch"].groupby("N")["test_bits_2000"].mean().sort_index()
    ns, ys = list(s.index), list(s.values)
    for k in range(len(ns) - 1):
        if ys[k] >= bits >= ys[k + 1]:
            fr = (ys[k] - bits) / (ys[k] - ys[k + 1])
            return 10 ** (math.log10(ns[k]) + fr * (math.log10(ns[k + 1]) - math.log10(ns[k])))
    return None                                            # outside the measured scratch range


def rough(x):
    """Rounds an estimate to 2 significant figures, so it doesn't look more precise than it is."""
    return round(x, -int(math.floor(math.log10(abs(x)))) + 1)


def main():
    df, exp = load()
    base2 = run_value(df, "n0_base", "test_bits_2000")
    base1 = run_value(df, "n0_base", "test_bits_1000")
    base2_top1 = run_value(df, "n0_base", "test_top1_2000")
    u2, r2, n2 = baseline("2000-2199", "test", "shard 000")
    u1, r1, n1 = baseline("1000-1199", "test", "shard 000")
    exp["gain"] = base2 - exp["test_bits_2000"]
    exp["forget"] = exp["test_bits_1000"] - base1

    p = ["# Tables", "",
         f"Made by `summarize.py` from `{RUNS_CSV}`. Test positions (shard 000): {n2:,} for 2000-2199, {n1:,} for 1000-1199. "
         "± = standard deviation over 3 seeds; no ± = 1 seed (seed 0). Bold = best option per row.", ""]

    p += ["## 1. Strong players: test bits per move on 2000-2199 (lower is better)",
          f"Base model with 0 strong games: **{base2:.3f}**. Uniform over legal moves: {u2:.3f}.", "",
          grid_md(grid(exp, "test_bits_2000"), "ftall"), ""]
    p += ["## 2. Strong players: top-1 accuracy (higher is better)",
          f"Base model: {base2_top1:.1%}. Random legal move: {r2:.1%}.", "",
          grid_md(grid(exp, "test_top1_2000", "{:.1%}"), "ftall"), ""]
    p += ["## 3. Gain over the base model on strong players (bits per move, + = better)", "",
          grid_md(grid(exp, "gain", "{:+.3f}")), ""]
    p += ["## 4. Forgetting: change on weak players (1000-1199) after training on strong players (+ = worse)",
          f"Base model on weak players: {base1:.3f} bits.", "",
          grid_md(grid(exp, "forget", "{:+.3f}")), ""]

    # 5. data efficiency
    rows = []
    e0 = scratch_equivalent(exp, base2)
    rows.append(["base model, 0 games", f"{base2:.3f}", f"≈{rough(e0):,.0f}" if e0 else "n/a", f"≈{rough(e0):,.0f} games for free" if e0 else ""])
    means = exp.groupby(["option", "N"])["test_bits_2000"].mean()
    s_max = means["scratch"].max(); s_min = means["scratch"].min()
    for opt in ("ftall", "ftbn"):
        for n, bits in means[opt].items():
            e = scratch_equivalent(exp, bits)
            if e is not None:
                need, save = f"≈{rough(e):,.0f}", f"≈{rough(e / n):g}×"
            elif bits < s_min:
                need = f"more than {ALL:,} (scratch on all {ALL:,} games: {s_min:.3f}, {s_min - bits:.3f} worse)"
                save = f"> {ALL / n:,.1f}×" if n < ALL else "beats scratch on the same data"
            else:
                need, save = "less than 100", "n/a"
            rows.append([f"{NAMES[opt]}, {n_label(n)} games", f"{bits:.3f}", need, save])
    p += ["## 5. Strong-player games that pretraining saves",
          "Games that training from scratch would need for the same test bits, interpolated linearly in log(N) between the measured scratch points (rough).", "",
          md_table(["Result", "Bits", "Scratch would need", "Saving"], rows), "",
          f"Scratch was not tuned for small N (lr 1e-3, no regularisation). There is one seed at 10,000 and all games, and the test set has "
          f"{games_in('2000-2199', 'test', 'shard 000')} games, so a gap of a few thousandths of a bit is within test noise: \"more than "
          f"{ALL:,}\" for 10,000 games means about as good as scratch on all {ALL:,} games. With all games, both runs were still improving "
          "when they hit the pass limit.", ""]

    # 6. training cost
    rows = []
    for n in sorted(exp["N"].unique()):
        for opt in OPTIONS:
            r = exp[(exp["N"] == n) & (exp["option"] == opt)]
            stopped = r["early_stopped"].astype(str).eq("True").sum()
            still = r["best_is_last"].astype(str).eq("True").sum()
            rows.append([n_label(n), NAMES[opt], f"{r['train_positions'].mean():,.0f}", f"{r['best_step'].mean():,.0f}",
                         f"{r['last_step'].mean():,.0f}", f"{r['passes_completed'].mean():.1f}", f"{r['minutes'].mean():.1f}",
                         f"{stopped}/{len(r)}", f"{still}/{len(r)}"])
    p += ["## 6. Training cost and dynamics (means over seeds)", "",
          md_table(["N", "Option", "Training positions", "Best checkpoint (step)", "Last step", "Passes", "Minutes",
                    "Stopped early", "Best = last eval (still improving)"], rows), ""]

    # 7. base model
    rows = []
    for band, scope in (("1000-1199", "all shards"), ("2000-2199", "all shards")):
        u, r, n = baseline(band, "test", scope)
        tag = band[:4]
        rows.append([band + ("" if band == "1000-1199" else " (never trained on)"), f"{n:,}",
                     f"**{run_value(df, 'base', f'test_bits_{tag}'):.3f}**", f"{run_value(df, 'base', f'test_top1_{tag}'):.1%}",
                     f"{u:.3f}", f"{r:.1%}"])
    p += ["## 7. Base model (trained on 1000-1199): all test positions", "",
          md_table(["Band", "Positions", "Bits per move", "Top-1", "Uniform", "Random move"], rows), ""]

    # 8. learning-rate check
    rows = []
    for run in ("ftall_100_s0", "ftall_100_s0_lr1e-3", "ftbn_100_s0", "ftbn_100_s0_lr1e-4"):
        r = df[df["run"] == run].iloc[0]
        rows.append([run, NAMES[r["option"]], f"{r['lr']:g}", f"{r['test_bits_2000']:.3f}", f"{r['test_bits_1000']:.3f}",
                     f"{int(r['best_step'])}", f"{int(r['last_step'])}" + (" (early stop)" if str(r["early_stopped"]) == "True" else " (cap)")])
    p += ["## 8. Learning-rate check (N = 100, seed 0)", "",
          md_table(["Run", "Option", "Learning rate", "Test 2000-2199", "Test 1000-1199", "Best step", "Last step"], rows), ""]

    with open(OUT, "w") as f:
        f.write("\n".join(p))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
