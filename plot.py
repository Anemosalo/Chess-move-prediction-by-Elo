"""Live plot of one training run.

Usage: python3 plot.py runs/<name>

Reads runs/<name>/history.json (written by train.py at every evaluation), redraws every
REFRESH seconds until the window is closed, and saves runs/<name>/training_curves.png.
If runs/<name>/test.json exists ({"<band>": {"bits": ..., "top1": ...}, ...}), the test
results are drawn as stars at the end of the run.
"""
import json
import os
import sys

import matplotlib.pyplot as plt

REFRESH = 15                    # seconds between redraws

# Uniform baseline on each band's val rows (split == 1 of shard 000): (bits, top-1 %)
UNIFORM = {
    "1000-1199": (4.580, 6.92),
    "2000-2199": (4.679, 6.03),
}


def load_history(run_dir):
    """Returns (history, training band). Also reads the old format of run 1 (one val record, no config.json)."""
    with open(os.path.join(run_dir, "history.json")) as f:
        h = json.load(f)
    band = "1000-1199"
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            band = json.load(f)["data"]
    if "val" not in h:          # old format: val_step / val_bits / val_top1
        h["val"] = {band: {"step": h["val_step"], "bits": h["val_bits"], "top1": h["val_top1"]}}
    return h, band


def load_test(run_dir):
    path = os.path.join(run_dir, "test.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def draw(h, band, test, ax1, ax2):
    """Draws the loss (left) and the top-1 accuracy (right): train line, one val line per band, baselines, test stars."""
    ax1.clear()
    ax2.clear()
    last_step = max([0] + h["step"] + [s for v in h["val"].values() for s in v["step"]])

    ax1.plot(h["step"], h["train_bits"], color="tab:blue", alpha=0.6, label="train (avg of 50 steps)")
    for i, (b, v) in enumerate(h["val"].items()):
        color = f"C{i + 1}"     # matplotlib's colour cycle: C1 orange, C2 green, …
        name = "val" if b == band else "also"
        ax1.plot(v["step"], v["bits"], "o-", color=color, markersize=4, label=f"{name} {b}")
        ax2.plot(v["step"], [t * 100 for t in v["top1"]], "o-", color=color, markersize=4, label=f"{name} {b}")
        if b in UNIFORM:
            ax1.axhline(UNIFORM[b][0], linestyle="--", color=color, alpha=0.5, label=f"uniform {b} ({UNIFORM[b][0]})")
            ax2.axhline(UNIFORM[b][1], linestyle="--", color=color, alpha=0.5, label=f"random move {b} ({UNIFORM[b][1]}%)")
        if b in test:
            ax1.plot(last_step, test[b]["bits"], "*", color=color, markersize=14, label=f"test {b} ({test[b]['bits']:.3f})")
            ax2.plot(last_step, test[b]["top1"] * 100, "*", color=color, markersize=14, label=f"test {b} ({test[b]['top1']:.1%})")

    for s in h["epoch_end"]:
        ax1.axvline(s, color="lightgray")
        ax2.axvline(s, color="lightgray")

    v = h["val"][band]
    ax1.set_xlabel("training step")
    ax1.set_ylabel("bits per move")
    ax1.set_title(f"Loss on {band}  (latest val {v['bits'][-1]:.3f})" if v["bits"] else f"Loss on {band}")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("training step")
    ax2.set_ylabel("top-1 accuracy (%)")
    ax2.set_title(f"How often the favourite move was played  (latest {v['top1'][-1]:.1%})" if v["top1"] else "Top-1")
    ax2.legend(fontsize=8)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 plot.py runs/<name>")
    run_dir = sys.argv[1]

    plt.ion()                                   # interactive mode: the window doesn't block the loop
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.canvas.manager.set_window_title(run_dir)

    while plt.fignum_exists(fig.number):        # runs until you close the window
        try:
            h, band = load_history(run_dir)
        except (FileNotFoundError, json.JSONDecodeError):
            # not written yet, or caught half-written by train.py: try again shortly
            plt.pause(1)
            continue

        draw(h, band, load_test(run_dir), ax1, ax2)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150)
        plt.pause(REFRESH)                      # redraw the window and wait


if __name__ == "__main__":
    main()
