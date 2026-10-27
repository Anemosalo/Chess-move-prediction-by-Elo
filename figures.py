"""Makes the summary figures F1-F10 into results/figures/ as PNG (200 dpi) and PDF (written by Claude).

Usage: python3 figures.py        (run collect_results.py first after new runs)
Every figure has a title that states the takeaway and a caption that says what is plotted and on which positions.
"""
import math
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from summarize import ALL, NAMES, OPTIONS, baseline, games_in, load, rough, run_value, scratch_equivalent

OUT_DIR = "results/figures"
OPT_COLORS = {"scratch": "tab:gray", "ftall": "tab:blue", "ftbn": "tab:red"}
BAND_COLORS = {"1000-1199": "tab:orange", "2000-2199": "tab:green"}
NS = [100, 1000, 10000, ALL]
N_LABELS = ["100", "1k", "10k", "40k\n(all)"]
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.grid": True, "grid.alpha": 0.25})


# ---------- shared helpers ----------

def log_n_axis(ax, label="strong-player (2000-2199) games used for training  (log scale)"):
    ax.set_xscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels(N_LABELS)
    ax.minorticks_off()
    ax.set_xlim(70, 60000)
    ax.set_xlabel(label)


def finish(fig, title, caption, bottom=0.2, top=0.88):
    """Bold takeaway title on top, a caption under the plots, then save PNG + PDF."""
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(bottom=bottom, top=top)
    width = int(fig.get_figwidth() * 14.5)              # characters per line at 8.5 pt
    fig.text(0.015, 0.012, textwrap.fill(caption, width), fontsize=8.5, ha="left", va="bottom", color="#333333")


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, name + ".png"), dpi=200)
    fig.savefig(os.path.join(OUT_DIR, name + ".pdf"))
    plt.close(fig)
    print("saved", os.path.join(OUT_DIR, name), "(.png, .pdf)")


def stats(exp, column):
    """Mean, std and number of seeds per (option, N)."""
    return exp.groupby(["option", "N"])[column].agg(["mean", "std", "count"])


def errorbar_lines(ax, st, scale=1.0, options=OPTIONS):
    for opt in options:
        s = st.loc[opt]
        ax.errorbar(s.index, s["mean"] * scale, yerr=s["std"].fillna(0) * scale, marker="o", ms=6, capsize=4,
                    lw=2, color=OPT_COLORS[opt], label=NAMES[opt])


# ---------- F1: main result ----------

def f1_main(df, exp):
    base2 = run_value(df, "n0_base", "test_bits_2000")
    u2, _, n2 = baseline("2000-2199", "test", "shard 000")
    _, _, nv = baseline("2000-2199", "val", "shard 000")
    gv = games_in("2000-2199", "val", "shard 000")
    st = stats(exp, "test_bits_2000")
    fig, ax = plt.subplots(figsize=(10, 7.4))
    errorbar_lines(ax, st)
    ax.axhline(base2, color="black", ls="--", lw=1.2, label=f"base model, 0 strong games ({base2:.3f})")
    ax.axhline(u2, color="gray", ls=":", lw=1.2, label=f"uniform over legal moves ({u2:.3f})")

    lines, savings = [], []
    for n in (100, 1000):
        bits = st.loc[("ftall", n), "mean"]
        e = scratch_equivalent(exp, bits)
        ax.annotate("", xy=(e, bits), xytext=(n * 1.08, bits),
                    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.3, ls=(0, (2, 2))))
        ax.plot(e, bits, "o", mfc="white", mec="tab:gray", ms=7)
        lines.append(f"full fine-tune on {n:,} games ({bits:.3f}) = scratch on ≈{rough(e):,.0f} games: ≈{rough(e / n):g}× fewer")
        savings.append(rough(e / n))
    e0 = scratch_equivalent(exp, base2)
    lines.append(f"base model with 0 strong games ({base2:.3f}) = scratch on ≈{rough(e0):,.0f} games")
    ax.text(0.98, 0.97, "Dotted arrows: games that training from scratch needs for the same score\n" + "\n".join(lines),
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="lightgray"))

    log_n_axis(ax)
    ax.set_ylabel("test bits per move on 2000-2199  (lower is better)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9, frameon=False)
    finish(fig, f"Fine-tuning needs ≈{min(savings):g}-{max(savings):g}× fewer strong games than untuned scratch for the same score",
           f"Test bits per move on {n2:,} held-out 2000-2199 positions (lower is better), against how many 2000-2199 games each model "
           "was trained on. Scratch = random start; fine-tunes start from the base model trained on 141,333 games of 1000-1199 players. "
           "Scratch was not tuned for small N (lr 1e-3, no regularisation), and every option picks its checkpoint on "
           f"{nv:,} val positions from {gv} strong games, more games than the N = 100 training set. "
           "Error bars: ±1 standard deviation over 3 seeds (N = 100 and 1k); one seed at 10k and 40k. Savings are interpolated "
           "linearly in log(N) between the measured scratch points, so they are rough.", bottom=0.3, top=0.91)
    save(fig, "F1_main_result")


# ---------- F2: top-1 ----------

def f2_top1(df, exp):
    base_t = run_value(df, "n0_base", "test_top1_2000") * 100
    _, r2, n2 = baseline("2000-2199", "test", "shard 000")
    st = stats(exp, "test_top1_2000")
    fig, ax = plt.subplots(figsize=(10, 6.2))
    errorbar_lines(ax, st, scale=100)
    ax.axhline(base_t, color="black", ls="--", lw=1.2, label=f"base model, 0 strong games ({base_t:.1f}%)")
    ax.text(0.98, 0.03, f"random legal move: {r2:.1%} (below the chart)", transform=ax.transAxes, ha="right", fontsize=8, color="dimgray")
    log_n_axis(ax)
    ax.set_ylabel("test top-1 accuracy on 2000-2199, %  (higher is better)")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.07), fontsize=9)
    best = st.loc[("ftall", 1000), "mean"]
    finish(fig, f"Fine-tuned on 1,000 strong games, the favourite move is right {best:.1%} of the time",
           f"Top-1 accuracy (how often the model's most likely move is the move played) on {n2:,} held-out 2000-2199 positions. "
           "Error bars: ±1 standard deviation over 3 seeds (N = 100 and 1k); one seed at 10k and 40k.", bottom=0.21, top=0.9)
    save(fig, "F2_top1")


# ---------- F3: forgetting ----------

def f3_forgetting(df, exp):
    base1 = run_value(df, "n0_base", "test_bits_1000")
    base2 = run_value(df, "n0_base", "test_bits_2000")
    _, _, n1 = baseline("1000-1199", "test", "shard 000")
    st = stats(exp, "test_bits_1000")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2))

    errorbar_lines(a1, st, options=["ftall", "ftbn"])
    a1.axhline(base1, color="tab:orange", ls="--", lw=1.5, label=f"base model before fine-tuning ({base1:.3f})")
    sc = st.loc["scratch", "mean"]
    a1.text(0.98, 0.97, f"scratch models: {sc.min():.2f}-{sc.max():.2f} bits\n(means; off the chart)", transform=a1.transAxes,
            ha="right", va="top", fontsize=8.5, color="dimgray")
    log_n_axis(a1)
    a1.set_ylabel("test bits per move on 1000-1199  (lower is better)")
    a1.set_title("Weak players after training on strong players")
    a1.legend(loc="lower left", fontsize=9)

    g = exp.assign(gain=base2 - exp["test_bits_2000"], forget=exp["test_bits_1000"] - base1)
    m = g.groupby(["option", "N"])[["gain", "forget"]].mean()
    for opt in ("ftall", "ftbn"):
        s = m.loc[opt]
        a2.plot(s["gain"], s["forget"], "-", color=OPT_COLORS[opt], lw=1, alpha=0.5)
        a2.scatter(s["gain"], s["forget"], s=[40 + 30 * math.log10(n) for n in s.index], color=OPT_COLORS[opt],
                   edgecolor="black", lw=0.5, zorder=3, label=NAMES[opt])
        for n, row in s.iterrows():
            a2.annotate(N_LABELS[NS.index(n)].replace("\n", " "), (row["gain"], row["forget"]), xytext=(7, 4),
                        textcoords="offset points", fontsize=8)
    a2.set_xlabel("gain on strong players: bits saved vs the base model  (→ better)")
    a2.set_ylabel("forgetting on weak players: bits lost vs the base model  (↓ better)")
    a2.set_title("The trade-off: best is bottom right")
    a2.legend(loc="upper left", fontsize=9)
    lo = m.loc[["ftall", "ftbn"], "forget"]
    finish(fig, f"Fine-tuning on strong players costs {lo.min():.2f}-{lo.max():.2f} bits on weak players",
           f"Left: test bits on {n1:,} held-out 1000-1199 positions after fine-tuning on N strong-player games (error bars ±1 sd over "
           "3 seeds at N = 100 and 1k). Right: mean gain on 2000-2199 against mean forgetting on 1000-1199 for each N (point size grows "
           "with N). Forgetting does not fall steadily with N (full fine-tune: "
           f"{', '.join(f'{v:+.2f}' for v in m.loc['ftall', 'forget'])} bits at 100, 1k, 10k and all games; one seed at 10k and all).",
           bottom=0.21)
    save(fig, "F3_forgetting")


# ---------- F4: learning dynamics ----------

def f4_dynamics(df, exp):
    val = pd.read_csv("results/curves_val.csv")
    tr = pd.read_csv("results/curves_train.csv")
    base_val = run_value(df, "base", "val_bits_2000_at_best")      # the base model's best checkpoint = the 0-games start
    _, _, nv = baseline("2000-2199", "val", "shard 000")
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.6), sharey=True)
    for ax, n, lab in zip(axes, NS, ["100 games", "1,000 games", "10,000 games", "all 40,019 games"]):
        for opt in OPTIONS:
            run = f"{opt}_{'all' if n == ALL else n}_s0"
            r = df[df["run"] == run].iloc[0]
            spp = math.ceil(r["train_positions"] / 1024)               # steps per pass
            v = val[(val["run"] == run) & (val["band"] == "2000-2199")]
            t = tr[tr["run"] == run]
            ax.plot(v["step"] / spp, v["bits"], "-", color=OPT_COLORS[opt], lw=2, label=f"{NAMES[opt]}: val")
            ax.plot(t["step"] / spp, t["train_bits"], ":", color=OPT_COLORS[opt], lw=1.5, label=f"{NAMES[opt]}: train")
            b = v.loc[v["bits"].idxmin()]
            ax.plot(b["step"] / spp, b["bits"], "*", ms=13, color=OPT_COLORS[opt], mec="black", mew=0.6)
        ax.axhline(base_val, color="black", ls="--", lw=1)
        ax.set_title(lab)
        ax.set_xlabel("passes over the training games")
    axes[0].set_ylabel("bits per move on 2000-2199  (lower is better)")
    axes[0].set_ylim(2.3, 4.75)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, 0.12))
    finish(fig, "With few games the best checkpoint comes after a few passes, then the model memorizes",
           f"Seed 0. Solid = val bits on {nv:,} held-out 2000-2199 positions; dotted = train loss (average of 50 steps); ★ = best "
           f"checkpoint (the one tested). Black dashed = base model with 0 strong games (val {base_val:.3f}). At 100 and 1k games the train "
           "loss keeps falling after the ★ while val rises (memorizing). BN + output layer: flat train loss at 100 and 1k games; at 10k and "
           "all games its train loss rises again after a few hundred steps, so the optimisation diverges (lr 1e-3), which says nothing about "
           "its capacity. With 10k and all games full fine-tuning and scratch were still improving when they hit the pass limit.",
           bottom=0.27, top=0.85)
    save(fig, "F4_learning_dynamics")


# ---------- F5: seed spread ----------

def f5_seeds(df, exp):
    base1 = run_value(df, "n0_base", "test_bits_1000")
    base2 = run_value(df, "n0_base", "test_bits_2000")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    sub = exp[exp["N"].isin([100, 1000]) & exp["option"].isin(["ftall", "ftbn"])]
    for ax, col, base, band in ((axes[0], "test_bits_2000", base2, "2000-2199 (strong, trained on)"),
                                (axes[1], "test_bits_1000", base1, "1000-1199 (weak, forgetting)")):
        x, ticks, labels = 0, [], []
        for n in (100, 1000):
            for opt in ("ftall", "ftbn"):
                vals = sub[(sub["N"] == n) & (sub["option"] == opt)][col]
                ax.scatter([x] * len(vals), vals, color=OPT_COLORS[opt], s=40, zorder=3)
                ax.hlines(vals.mean(), x - 0.25, x + 0.25, color=OPT_COLORS[opt], lw=2.5)
                ax.text(x + 0.3, vals.mean(), f"{vals.mean():.3f} ± {vals.std():.3f}", va="center", fontsize=8)
                ticks.append(x); labels.append(f"{'full' if opt == 'ftall' else 'BN'}\nN = {n:,}")
                x += 1
            x += 0.8
        ax.axhline(base, color="black", ls="--", lw=1, label=f"base model ({base:.3f})")
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_xlim(-0.6, x - 0.2)
        ax.set_ylabel(f"test bits per move on {band.split()[0]}  (lower is better)")
        ax.set_title(band)
        ax.legend(loc="upper right", fontsize=8.5)
    sc = exp[exp["option"] == "scratch"].groupby("N")["test_bits_2000"].agg(["mean", "std"])
    sw = exp[(exp["option"] == "scratch") & exp["N"].isin([100, 1000])]["test_bits_1000"]
    ft = sub.groupby(["option", "N"])["test_bits_2000"].agg(["std", "min", "max"])
    gt = games_in("2000-2199", "test", "shard 000")
    finish(fig, "Fine-tune results barely depend on which games are sampled",
           "Each dot is one seed (a different random choice of games and a different shuffling); the bar is the mean, with mean ± sd "
           f"written next to it. On strong players the fine-tunes have a standard deviation of at most {ft['std'].max():.3f} bits and a "
           f"range of at most {(ft['max'] - ft['min']).max():.3f} bits over 3 seeds. This spread leaves out test-set noise ({gt} test "
           f"games). Scratch varies more (sd up to {sc['std'].max():.3f} on strong players, {sw.min():.1f}-{sw.max():.1f} bits on weak "
           "players), not shown to keep the scale readable.", bottom=0.2)
    save(fig, "F5_seed_spread")


# ---------- F6: base model ----------

def f6_base(df, exp):
    val = pd.read_csv("results/curves_val.csv")
    tr = pd.read_csv("results/curves_train.csv")
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for band in ("1000-1199", "2000-2199"):
        v = val[(val["run"] == "base") & (val["band"] == band)]
        role = "val (trained on)" if band == "1000-1199" else "val (never trained on)"
        ax.plot(v["step"] * 1024 / 1e6, v["bits"], "o-", ms=3, color=BAND_COLORS[band], lw=2, label=f"{band}: {role}")
    t = tr[tr["run"] == "base"]
    ax.plot(t["step"] * 1024 / 1e6, t["train_bits"], color="tab:blue", alpha=0.5, lw=1, label="train loss (average of 50 steps)")
    for band in ("1000-1199", "2000-2199"):
        u, r, n = baseline(band, "val", "shard 000")
        ax.axhline(u, color=BAND_COLORS[band], ls="--", alpha=0.5, lw=1, label=f"uniform {band} ({u:.3f})")
    ax.set_xlabel("positions seen in training (millions)")
    ax.set_ylabel("bits per move  (lower is better)")
    ax.legend(fontsize=8.5)
    t1 = run_value(df, "base", "test_bits_1000"); t2 = run_value(df, "base", "test_bits_2000")
    _, _, v1 = baseline("1000-1199", "val", "shard 000"); _, _, v2 = baseline("2000-2199", "val", "shard 000")
    _, _, n1 = baseline("1000-1199", "test", "all shards"); _, _, n2 = baseline("2000-2199", "test", "all shards")
    still = str(run_value(df, "base", "best_is_last")) == "True"
    finish(fig, "The base model during its one pass over the 1000-1199 games",
           f"Val bits on {v1:,} held-out 1000-1199 positions and {v2:,} held-out 2000-2199 positions (shard 000) while training once over "
           f"{run_value(df, 'base', 'train_positions') / 1e6:.2f}M positions of 1000-1199 games"
           + (" (the best checkpoint is the last evaluation, so it was still improving)" if still else "") + ". "
           "The 2000-2199 line is noisy because those ratings were never in training. "
           f"Base model test (all test positions): {t1:.3f} bits on 1000-1199 ({n1:,} positions), {t2:.3f} on 2000-2199 ({n2:,}).",
           bottom=0.22)
    save(fig, "F6_base_model")


# ---------- F7: learning-rate check ----------

def f7_lr(df, exp):
    runs = [("ftall_100_s0", "full, lr 1e-4\n(experiment)"), ("ftall_100_s0_lr1e-3", "full, lr 1e-3\n(check)"),
            ("ftbn_100_s0", "BN + output\nlayer, lr 1e-3\n(experiment)"), ("ftbn_100_s0_lr1e-4", "BN + output\nlayer, lr 1e-4\n(check)")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, col, band, base in ((axes[0], "test_bits_2000", "2000-2199", run_value(df, "n0_base", "test_bits_2000")),
                                (axes[1], "test_bits_1000", "1000-1199", run_value(df, "n0_base", "test_bits_1000"))):
        for x, (run, lab) in enumerate(runs):
            r = df[df["run"] == run].iloc[0]
            color = OPT_COLORS[r["option"]]
            ax.plot(x, r[col], "o", ms=11, color=color, mfc=color if "check" not in lab else "white", mew=2)
            ax.text(x, r[col] + 0.006, f"{r[col]:.3f}", ha="center", va="bottom", fontsize=9)
        ax.axhline(base, color="black", ls="--", lw=1, label=f"base model ({base:.3f})")
        ax.set_xticks(range(len(runs))); ax.set_xticklabels([lab for _, lab in runs], fontsize=8.5)
        ax.set_xlim(-0.6, len(runs) - 0.4)
        ax.margins(y=0.15)
        ax.set_ylabel(f"test bits per move on {band}  (lower is better)")
        ax.set_title(band + (" (trained on)" if band == "2000-2199" else " (forgetting)"))
        ax.legend(fontsize=8.5)
    bn = df[df["run"].isin(["ftbn_100_s0", "ftbn_100_s0_lr1e-4"])].set_index("run")
    slow = bn.loc["ftbn_100_s0_lr1e-4", "best_step"] / bn.loc["ftbn_100_s0", "best_step"]
    still = str(bn.loc["ftbn_100_s0_lr1e-4", "best_is_last"]) == "True"
    fa = df[df["run"] == "ftall_100_s0_lr1e-3"].iloc[0]
    passes = fa["best_step"] / math.ceil(fa["train_positions"] / 1024)
    finish(fig, f"BN + output layer reaches ≈{bn['test_bits_2000'].mean():.2f} at either learning rate; full fine-tuning is best at 1e-4",
           "N = 100 games, seed 0 only (not checked at larger N). Filled = the learning rate used in the experiment; open = the other "
           f"option's learning rate. BN + output layer at 1e-4 reaches {bn.loc['ftbn_100_s0_lr1e-4', 'test_bits_2000']:.3f} against "
           f"{bn.loc['ftbn_100_s0', 'test_bits_2000']:.3f} at 1e-3, after {slow:g}× more steps"
           + (" (and still improving at the step limit)" if still else "") + f"; full fine-tuning at 1e-3 is best after "
           f"{passes:g} passes and then overfits. So at N = 100 the full-vs-BN gap is not caused by the learning rate.", bottom=0.22)
    save(fig, "F7_learning_rate_check")


# ---------- F8, F9: every rating band (from eval_bands.py) ----------

BANDS = ["1000-1199", "1200-1399", "1400-1599", "1600-1799", "1800-1999", "2000-2199"]
MODEL_STYLE = {   # name -> (label, colour, trained on)
    "base": ("base model (1000-1199 only)", "tab:orange", ["1000-1199"]),
    "ftall_1000_s0": ("base + full fine-tune on 1,000 games of 2000-2199", "#9ecae1", ["1000-1199", "2000-2199"]),
    "ftall_10000_s0": ("base + full fine-tune on 10,000 games", "#4292c6", ["1000-1199", "2000-2199"]),
    "ftall_all_s0": ("base + full fine-tune on all 40,019 games", "#08306b", ["1000-1199", "2000-2199"]),
    "scratch_all_s0": ("scratch on 2000-2199 only (40,019 games)", "tab:gray", ["2000-2199"]),
}


def f8_bands(be):
    t = be[(be["split"] == "test") & (be["rating_mode"] == "true")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.4))
    x = range(len(BANDS))
    for name, (lab, col, _) in MODEL_STYLE.items():
        m = t[t["model"] == name].set_index("band").loc[BANDS]
        a1.plot(x, m["bits"], "o-", color=col, lw=2, ms=6, label=lab)
        a2.plot(x, m["top1"] * 100, "o-", color=col, lw=2, ms=6, label=lab)
    u = t[t["model"] == "base"].set_index("band").loc[BANDS]
    a1.plot(x, u["uniform_bits"], ":", color="gray", lw=1.5, label="uniform over legal moves")
    for ax in (a1, a2):
        for i in (0, len(BANDS) - 1):
            ax.axvspan(i - 0.35, i + 0.35, color="khaki", alpha=0.35, lw=0)
        ax.set_xticks(list(x)); ax.set_xticklabels(BANDS)
        ax.set_xlabel("rating band of both players (shaded = bands used in some training)")
    a1.set_ylabel("test bits per move  (lower is better)")
    a1.set_title("Bits per move on every band")
    a2.set_ylabel("test top-1 accuracy, %  (higher is better)")
    a2.set_title("How often the favourite move was played")
    a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(1.1, -0.13), ncol=3, frameon=False)
    n = t[t["model"] == "base"].set_index("band").loc[BANDS, "positions"]
    title, note = f8_text(t)
    finish(fig, title,
           f"Test positions per band (shard 000 test rows): {', '.join(f'{b} {int(v):,}' for b, v in n.items())}. "
           "Each model uses the real rating of the player to move. The middle bands (1200-1999) were never used for training by any model. "
           + note, bottom=0.3)
    save(fig, "F8_every_band")


def f9_rating_sweep(be):
    v = be[(be["split"] == "val")]
    models = ["base", "ftall_all_s0", "scratch_all_s0"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.0), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, name in zip(axes, models):
        for i, band in enumerate(BANDS):
            s = v[(v["model"] == name) & (v["band"] == band) & (v["rating_mode"] == "fixed")].sort_values("assumed_rating")
            best = s["bits"].min()
            col = cmap(i / (len(BANDS) - 1))
            ax.plot(s["assumed_rating"], s["bits"] - best, "-", color=col, lw=2, label=band)
            r_best = s.loc[s["bits"].idxmin(), "assumed_rating"]
            ax.plot(r_best, 0, "v", color=col, ms=9, mec="black", mew=0.5)
            true = v[(v["model"] == name) & (v["band"] == band) & (v["rating_mode"] == "true")].iloc[0]
            ax.plot(true["mean_rating"], true["bits"] - best, "o", color=col, ms=6, mfc="white", mew=2)
        ax.set_title(MODEL_STYLE[name][0])
        ax.set_xlabel("rating written into the rating plane for every position")
        for r in MODEL_STYLE[name][2]:
            lo = int(r[:4]); ax.axvspan(lo, lo + 199, color="khaki", alpha=0.35, lw=0)
    axes[0].set_ylabel("val bits per move above the band's best  (0 = best rating to assume)")
    axes[0].set_ylim(-0.01, None)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, title="real band", loc="lower center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, 0.13))
    finish(fig, "Models trained on one band prefer the ratings they were trained on, whatever band they predict",
           "Each line is one band's val positions (shard 000), measured with the rating plane set to the same value for every position. "
           "▼ = the rating that predicts that band best; ○ = the band's real ratings (measured normally). Shaded = ratings the model "
           "saw in training. " + f9_note(v, models), bottom=0.3, top=0.86)
    save(fig, "F9_rating_sweep")


def span(bands):
    """["1400-1599", "1600-1799"] -> "1400-1799" (the bands must be neighbours)."""
    return bands[0].split("-")[0] + "-" + bands[-1].split("-")[1]


def f8_text(t):
    """F8 title and note, from the test rows of bands_eval.csv."""
    p = t.pivot(index="band", columns="model", values="bits").loc[BANDS]
    best, worst = p.idxmin(axis=1), p.idxmax(axis=1)
    mid = [b for b in BANDS[1:-1] if best[b] == "ftall_all_s0"]
    sw = [b for b in BANDS if worst[b] == "scratch_all_s0"]
    title = f"After full fine-tuning on all strong games, the model is the best of these models on {span(mid)}, where it never trained"
    note = (f"The base model goes from {p.loc[BANDS[0], 'base']:.3f} bits on {BANDS[0]} to {p.loc[BANDS[-1], 'base']:.3f} on {BANDS[-1]}. "
            f"Full fine-tuning on all strong games costs {p.loc[BANDS[0], 'ftall_all_s0'] - p.loc[BANDS[0], 'base']:+.3f} bits on "
            f"{BANDS[0]} (forgetting). The scratch model never saw weak players and is the worst model on {span(sw)}.")
    return title, note


def f9_note(v, models):
    """F9 note: which fixed ratings each model prefers, how flat the minimum is, and the cost of the real ratings."""
    fixed = v[v["rating_mode"] == "fixed"]
    pref, flat = {}, []
    for name in models:
        r = []
        for band in BANDS:
            s = fixed[(fixed["model"] == name) & (fixed["band"] == band)].sort_values("bits")
            r.append(int(s["assumed_rating"].iloc[0]))
            flat.append(s["bits"].iloc[1] - s["bits"].iloc[0])
        pref[name] = (min(r), max(r))
    s = fixed[(fixed["model"] == "base") & (fixed["band"] == BANDS[-1])]
    r_best = int(s.loc[s["bits"].idxmin(), "assumed_rating"])
    true = v[(v["model"] == "base") & (v["band"] == BANDS[-1]) & (v["rating_mode"] == "true")]["bits"].iloc[0]
    others = sorted({x for name in models[1:] for x in pref[name]})
    return ("A model that had learned what a rating means would predict each band best near its real ratings (▼ on ○). Instead, for "
            f"every band, the base model prefers to be told ≈{pref['base'][0]}-{pref['base'][1]} and the other two "
            f"≈{others[0]}-{others[-1]}. The bottom of each curve is flat (best and second-best rating within {max(flat):.4f} bits), so "
            "only the location of the curve is meaningful, not the exact best rating. Telling the base model the real ratings of "
            f"{BANDS[-1]} players costs {true - s['bits'].min():.2f} bits compared with telling it {r_best}.")


# ---------- F10: the model trained on 4 bands at once ----------

JOINT_TRAIN = ["1000-1199", "1400-1599", "1800-1999", "2200-2399"]
FLAT = 0.002                                   # bits: sweep ratings this close to the best count as equally good


def f10_joint(df, be, bj):
    """Left: the rating that predicts each band best vs the band's real rating (old models vs the joint model).
    Middle: the joint model's sweep curves. Right: test bits per band, joint model vs the earlier models."""
    name = bj["model"].iloc[0]
    jbands = sorted(bj["band"].unique(), key=lambda b: int(b.split("-")[0]))
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(19, 8.2))

    def best_rows(df, model, bands):
        out = []
        for b in bands:
            s = df[(df["model"] == model) & (df["band"] == b) & (df["split"] == "val") & (df["rating_mode"] == "fixed")]
            if len(s) == 0:
                continue
            real = df[(df["model"] == model) & (df["band"] == b) & (df["split"] == "val") & (df["rating_mode"] == "true")].iloc[0]
            near = s[s["bits"] <= s["bits"].min() + FLAT]["assumed_rating"]     # the flat bottom of the curve
            out.append((real["mean_rating"], s.loc[s["bits"].idxmin(), "assumed_rating"], b, near.min(), near.max()))
        return out

    lim = (800, 2600)
    a1.plot(lim, lim, ":", color="gray", lw=1.2, label="best rating = real rating")
    for model, col in (("base", "tab:orange"), ("ftall_all_s0", "#08306b"), ("scratch_all_s0", "tab:gray")):
        pts = best_rows(be, model, BANDS)
        a1.plot([p[0] for p in pts], [p[1] for p in pts], "o--", color=col, ms=6, lw=1.2, label=MODEL_STYLE[model][0])
    pts = best_rows(bj, name, jbands)
    a1.plot([p[0] for p in pts], [p[1] for p in pts], "-", color="tab:red", lw=2.5, label="ONE model trained on 4 bands at once")
    for x, y, b, lo, hi in pts:
        a1.vlines(x, lo, hi, color="tab:red", lw=5, alpha=0.25)
        a1.plot(x, y, "o", ms=9, color="tab:red", mfc="tab:red" if b in JOINT_TRAIN else "white", mew=2, zorder=3)
    a1.plot([], [], "-", color="tab:red", lw=5, alpha=0.25, label=f"joint model, ratings within {FLAT} bits of the best")
    a1.plot([], [], "o", color="tab:red", mfc="white", mew=2, label="joint model, never-seen band (open)")
    a1.set_xlim(*lim); a1.set_ylim(*lim)
    a1.set_xlabel("real mean rating of the band's players")
    a1.set_ylabel("rating that predicts the band best (rating sweep)")
    a1.set_title("Best rating to assume vs the band's real rating")
    a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)   # under the panel: no points hidden

    cmap = plt.get_cmap("viridis")
    for i, b in enumerate(jbands):
        s = bj[(bj["band"] == b) & (bj["split"] == "val") & (bj["rating_mode"] == "fixed")].sort_values("assumed_rating")
        best = s["bits"].min()
        col = cmap(i / (len(jbands) - 1))
        a2.plot(s["assumed_rating"], s["bits"] - best, "-" if b in JOINT_TRAIN else "--", color=col, lw=2,
                label=b + ("" if b in JOINT_TRAIN else " (never seen)"))
        a2.plot(s.loc[s["bits"].idxmin(), "assumed_rating"], 0, "v", color=col, ms=9, mec="black", mew=0.5)
        real = bj[(bj["band"] == b) & (bj["split"] == "val") & (bj["rating_mode"] == "true")].iloc[0]
        a2.plot(real["mean_rating"], real["bits"] - best, "o", color=col, ms=6, mfc="white", mew=2)
    a2.set_ylim(-0.005, 0.2)
    a2.set_xlabel("rating written into the rating plane for every position")
    a2.set_ylabel("val bits per move above the band's best")
    a2.set_title("Joint model: each band's U-shaped curve moves with its rating")
    a2.legend(fontsize=7.5, ncol=2, loc="upper center")

    t = bj[(bj["split"] == "test") & (bj["rating_mode"] == "true")].set_index("band").loc[jbands]
    xs = range(len(jbands))
    a3.plot(xs, t["bits"], "-", color="tab:red", lw=2.5, label="ONE model trained on 4 bands")
    for x, b in zip(xs, jbands):
        a3.plot(x, t.loc[b, "bits"], "o", ms=9, color="tab:red", mfc="tab:red" if b in JOINT_TRAIN else "white", mew=2, zorder=3)
    old = be[(be["split"] == "test") & (be["rating_mode"] == "true")]
    for model in ("base", "ftall_all_s0", "scratch_all_s0"):
        lab, col, _ = MODEL_STYLE[model]
        o = old[old["model"] == model].set_index("band")
        a3.plot([jbands.index(b) for b in BANDS], [o.loc[b, "bits"] for b in BANDS], "o--", color=col, ms=5, lw=1.2, label=lab)
    a3.set_xticks(list(xs)); a3.set_xticklabels(jbands, rotation=35, ha="right", fontsize=8)
    a3.set_ylabel("test bits per move  (lower is better)")
    a3.set_title("Every band: the joint model vs the earlier models")
    a3.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    steps = run_value(df, name, "last_step") / run_value(df, "base", "last_step")
    finish(fig, "Trained on 4 rating bands at once, the model's best rating to assume rises with each band's real rating",
           "Left: for each band, the rating that gives the lowest val bits when written into the rating plane for every position, against the "
           "band's real mean rating. The earlier models (trained on one band, or one band after another) prefer one fixed rating for every band; "
           f"the joint model's best rating rises with the real rating, including the never-seen bands (open circles). The bottom of each curve "
           f"is flat (shaded bars: ratings within {FLAT} bits of the best), so the 100-point best rating is coarse; the rating plane also stands "
           "in for time control and game length, which change with rating. Middle: the joint model's sweep; ▼ = best rating, ○ = real ratings. "
           "Right: test bits (shard 000 test rows); open circles = bands the joint model never trained on. This comparison is not controlled: "
           f"the joint model ran {steps:.1f}× the steps of the base model, trained on the bands next to each never-seen band, and some players "
           "of the never-seen bands also play in the training bands. Trained bands: 30,000 games each; 2 passes; one seed.", bottom=0.36)
    save(fig, "F10_joint_model")


def main():
    df, exp = load()
    f1_main(df, exp)
    f2_top1(df, exp)
    f3_forgetting(df, exp)
    f4_dynamics(df, exp)
    f5_seeds(df, exp)
    f6_base(df, exp)
    f7_lr(df, exp)
    if os.path.exists("results/bands_eval.csv"):
        be = pd.read_csv("results/bands_eval.csv")
        f8_bands(be)
        f9_rating_sweep(be)
        if os.path.exists("results/bands_eval_joint_4bands.csv"):
            f10_joint(df, be, pd.read_csv("results/bands_eval_joint_4bands.csv"))


if __name__ == "__main__":
    main()
