"""Plot of one training run, with a title saying what the run is and a caption saying what the curves show.

Usage:
  python3 plot.py runs/<name>          live window, redraws every REFRESH seconds until closed
  python3 plot.py runs/<name> --once   draw once, save, exit (no window)

Reads runs/<name>/config.json, history.json, test.json (if present) and logs/<name>.log (if present).
Saves runs/<name>/training_curves.png.
"""
import json
import math
import os
import sys

import matplotlib
if "--once" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

REFRESH = 15                    # seconds between redraws in live mode
W_ALL, W_BN = 459145, 6409      # trainable weights: everything / BatchNorm γ, β + head

# Baselines on each band's val rows (split == 1 of shard 000): (bits, top-1 %, positions)
UNIFORM = {
    "1000-1199": (4.580, 6.92, 33344),
    "2000-2199": (4.679, 6.03, 42433),
}
BAND_COLORS = {"1000-1199": "tab:orange", "2000-2199": "tab:green"}
OTHER_COLORS = ["tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]


def band_color(band, all_bands):
    if len(all_bands) > 3:                         # many bands: one colour per band, from low (purple) to high (yellow) rating
        order = sorted(all_bands, key=lambda b: int(b.split("-")[0]))
        return plt.get_cmap("viridis")(order.index(band) / (len(order) - 1))
    if band in BAND_COLORS:
        return BAND_COLORS[band]
    others = sorted(b for b in all_bands if b not in BAND_COLORS)
    return OTHER_COLORS[others.index(band) % len(OTHER_COLORS)]


def train_bands(cfg):
    """The training band(s) of a run as a list (old runs store one band as a string)."""
    d = cfg.get("data", "1000-1199")
    return d if isinstance(d, list) else [d]


def selection(h, cfg):
    """The val curve that picks the best checkpoint: the training band's val, or the mean over several training bands."""
    bands = train_bands(cfg)
    if len(bands) == 1:
        return bands[0], h["val"][bands[0]]
    return "mean of the training bands", h["val_mean"]


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def load_run(run_dir):
    """Returns (history, config, test, log text). Also reads run 1's old format (no config, one val record)."""
    h = read_json(os.path.join(run_dir, "history.json"))
    if h is None:
        raise FileNotFoundError("history.json")
    cfg = read_json(os.path.join(run_dir, "config.json"), {"data": "1000-1199", "init": "scratch", "tune": "all"})
    if "val" not in h:
        h["val"] = {cfg["data"]: {"step": h["val_step"], "bits": h["val_bits"], "top1": h["val_top1"]}}
    test = read_json(os.path.join(run_dir, "test.json"), {})
    name = os.path.basename(os.path.normpath(run_dir))
    log_path = os.path.join("logs", name + ".log")
    log = open(log_path).read() if os.path.exists(log_path) else ""
    return h, cfg, test, log


def describe(cfg):
    """One plain sentence: what was trained, from what, on what."""
    init, tune = cfg.get("init", "scratch"), cfg.get("tune", "all")
    if init == "scratch":
        what = "Trained from scratch (random start)"
    elif tune == "bn":
        what = f"BatchNorm-only fine-tune of {init}"
    else:
        what = f"Full fine-tune of {init}"
    games = cfg.get("games")
    tb = ", ".join(train_bands(cfg))
    on = f"on {games:,} games of {tb}" if games else f"on all training games of {tb}"
    extras = []
    if cfg.get("seed") is not None:
        extras.append(f"seed {cfg['seed']}")
    if cfg.get("lr") is not None:
        extras.append(f"lr {cfg['lr']:g}")
    extras.append(f"{(W_BN if tune == 'bn' else W_ALL):,} trainable weights")
    return f"{what} {on}  ({', '.join(extras)})"


def caption(h, cfg, test, log):
    """Two or three sentences of what the curves show, built from the numbers."""
    band, v = selection(h, cfg)
    if not v["bits"]:
        return "No evaluation yet."
    i = min(range(len(v["bits"])), key=lambda k: v["bits"][k])
    still = i == len(v["bits"]) - 1
    lines = [f"Val {band}: {v['bits'][0]:.3f} → best {v['bits'][i]:.3f} bits at step {v['step'][i]:,} (dashed line = the tested checkpoint)"
             + ("; still improving at the last evaluation." if still else f"; {v['bits'][-1]:.3f} at the last evaluation (step {v['step'][-1]:,}).")]
    if len(train_bands(cfg)) > 1:              # several bands: one short line for trained bands, one for held-out bands
        for title, group in (("Trained", [b for b in h["val"] if b in train_bands(cfg)]),
                             ("Held out", [b for b in h["val"] if b not in train_bands(cfg)])):
            if group:
                lines.append(f"{title} bands at the best checkpoint: " + ", ".join(f"{b} {h['val'][b]['bits'][i]:.3f}" for b in group if i < len(h["val"][b]["bits"])))
    for b, vb in h["val"].items():
        if b in train_bands(cfg) or i >= len(vb["bits"]) or len(train_bands(cfg)) > 1:
            continue
        delta = vb["bits"][i] - vb["bits"][0]
        word = "worse: forgetting" if delta > 0 else "better"
        lines.append(f"Also measured {b}: {vb['bits'][0]:.3f} → {vb['bits'][i]:.3f} bits at the best checkpoint ({delta:+.3f}, {word}).")
    if "Early stopping" in log:
        lines.append(f"Stopped early after {cfg.get('patience')} evaluations without improvement.")
    if test:
        lines.append("Test (★ at the tested checkpoint, measured once): " + ";  ".join(
            f"{b} {t['bits']:.3f} bits, top-1 {t['top1']:.1%} ({t.get('positions', 0):,} positions)" for b, t in test.items()))
    return "\n".join(lines)


def draw(fig, ax1, ax2, h, cfg, test, log, run_dir):
    ax1.clear()
    ax2.clear()
    for ax in fig.axes[2:]:            # old secondary axes from the previous redraw
        ax.remove()
    band, v = selection(h, cfg)
    trained = train_bands(cfg)
    multi = len(trained) > 1
    bands = list(h["val"])
    best_i = min(range(len(v["bits"])), key=lambda k: v["bits"][k]) if v["bits"] else None
    star_x = v["step"][best_i] if best_i is not None else 0     # the test uses the best checkpoint

    # left: bits per move
    if h["step"]:
        ax1.plot(h["step"], h["train_bits"], color="tab:blue", alpha=0.6, lw=1.2, label="train loss (average of 50 steps)")
    lows, highs = list(h["train_bits"]), list(h["train_bits"])
    if multi:
        ax1.plot(v["step"], v["bits"], "-", color="black", lw=2.5, label="mean val of the training bands (picks the checkpoint)")
    for b in bands:
        vb, c = h["val"][b], band_color(b, bands)
        if multi:
            role, style = ("train" if b in trained else "held out"), ("o-" if b in trained else "o--")
        else:
            role, style = ("val" if b == band else "also measured"), "o-"
        ax1.plot(vb["step"], vb["bits"], style, color=c, ms=3.5, lw=1.5, label=f"{role}: {b}")
        ax2.plot(vb["step"], [t * 100 for t in vb["top1"]], style, color=c, ms=3.5, lw=1.5, label=f"{role}: {b}")
        lows += vb["bits"]; highs += vb["bits"]
        if b in test:
            ax1.plot(star_x, test[b]["bits"], "*", color=c, ms=15, mec="black", mew=0.6, label=None if multi else f"test {b}: {test[b]['bits']:.3f}")
            ax2.plot(star_x, test[b]["top1"] * 100, "*", color=c, ms=15, mec="black", mew=0.6, label=None if multi else f"test {b}: {test[b]['top1']:.1%}")
            lows.append(test[b]["bits"])

    # y range: zoom on the data; baselines outside it become a note instead of squashing the curves
    lo, hi = min(lows), max(highs)
    pad = 0.08 * (hi - lo or 1)
    ax1.set_ylim(lo - pad, hi + pad)
    notes1, notes2 = [], []
    t_lo = min(t * 100 for b in bands for t in h["val"][b]["top1"])
    t_hi = max(t * 100 for b in bands for t in h["val"][b]["top1"])
    tpad = 0.08 * (t_hi - t_lo or 1)
    ax2.set_ylim(t_lo - tpad, t_hi + tpad)
    for b in bands:
        if b not in UNIFORM or multi:          # many bands: no baseline lines, they would only add clutter
            continue
        ub, ut, _ = UNIFORM[b]
        c = band_color(b, bands)
        if lo - pad <= ub <= hi + pad:
            ax1.axhline(ub, ls="--", color=c, alpha=0.5, label=f"uniform {b}: {ub}")
        else:
            notes1.append(f"uniform {b}: {ub} bits (off chart)")
        if t_lo - tpad <= ut <= t_hi + tpad:
            ax2.axhline(ut, ls="--", color=c, alpha=0.5, label=f"random legal move {b}: {ut}%")
        else:
            notes2.append(f"random legal move {b}: {ut}% (off chart)")
    for ax, notes in ((ax1, notes1), (ax2, notes2)):
        if notes:
            ax.text(0.99, 0.98, "\n".join(notes), transform=ax.transAxes, ha="right", va="top", fontsize=7, color="dimgray")

    # passes, best checkpoint, early stop
    for s in h["epoch_end"]:
        for ax in (ax1, ax2):
            ax.axvline(s, color="lightgray", lw=0.8, zorder=0)
    if best_i is not None:
        bs = v["step"][best_i]
        for ax in (ax1, ax2):
            ax.axvline(bs, color="black", ls="--", lw=1, alpha=0.6)
        ax1.annotate(f"best checkpoint\nstep {bs:,}", xy=(bs, v["bits"][best_i]), xytext=(6, 18), textcoords="offset points",
                     fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.6))
    if "Early stopping" in log:
        ax1.annotate("early stop", xy=(v["step"][-1], v["bits"][-1]), xytext=(-10, 22), textcoords="offset points",
                     fontsize=7, ha="right", arrowprops=dict(arrowstyle="-", lw=0.6))

    # a second x-axis on top: passes over the training games
    if h["epoch_end"] and h["epoch_end"][0] > 0:
        spp = h["epoch_end"][0]
        for ax in (ax1, ax2):
            sec = ax.secondary_xaxis("top", functions=(lambda s, spp=spp: s / spp, lambda p, spp=spp: p * spp))
            sec.set_xlabel("passes over the training games (grey lines = pass ends)", fontsize=8)
            sec.tick_params(labelsize=7)

    ax1.set_xlabel("training step (batches of 1,024 positions)")
    ax1.set_ylabel("bits per move  (lower is better)")
    ax1.set_title("How surprised the model is by the played move", fontsize=10)
    ax2.set_xlabel("training step (batches of 1,024 positions)")
    ax2.set_ylabel("top-1 accuracy, %  (higher is better)")
    ax2.set_title("How often its favourite move was the one played", fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3 if multi else 2, fontsize=7, frameon=False)

    name = os.path.basename(os.path.normpath(run_dir))
    fig.texts.clear()
    fig._suptitle = None
    if multi:                                  # many bands: taller figure so the legend and the caption both fit
        fig.set_size_inches(15, 9)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.42, wspace=0.18)
    fig.suptitle(f"{name}:  {describe(cfg)}", fontsize=10 if multi else 11, fontweight="bold", y=0.985)
    val_sizes = "" if multi else " (" + ", ".join(f"{b} {UNIFORM[b][2]:,}" for b in bands if b in UNIFORM) + " positions)"
    fig.text(0.01, 0.015, caption(h, cfg, test, log) + f"\nVal = shard 000 val rows{val_sizes}. Test = held-out test rows, never used for any choice.",
             fontsize=8, va="bottom", ha="left", family="monospace", wrap=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        raise SystemExit("usage: python3 plot.py runs/<name> [--once]")
    run_dir = args[0]
    once = "--once" in sys.argv

    if not once:
        plt.ion()                               # interactive mode: the window doesn't block the loop
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.6))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.31, wspace=0.18)
    if not once:
        fig.canvas.manager.set_window_title(run_dir)

    while once or plt.fignum_exists(fig.number):
        try:
            h, cfg, test, log = load_run(run_dir)
        except (FileNotFoundError, json.JSONDecodeError):
            if once:
                raise SystemExit(f"{run_dir}: no readable history.json")
            plt.pause(1)                        # not written yet, or caught half-written by train.py
            continue
        draw(fig, ax1, ax2, h, cfg, test, log, run_dir)
        fig.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150)
        if once:
            break
        plt.pause(REFRESH)


if __name__ == "__main__":
    main()
