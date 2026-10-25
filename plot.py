import json

import matplotlib.pyplot as plt

HISTORY = "history.json"        # written by train.py after every shard
REFRESH = 15                    # seconds between redraws

# Baselines on the val rows (split == 1 of shard 000 of 1000-1199):
UNIFORM_BITS = 4.580            # mean log2(K): every legal move equally likely
UNIFORM_TOP1 = 6.92             # mean 1/K, in %: a random legal move as the favourite


def draw(h, ax1, ax2):
    """Draws the loss (left) and the top-1 accuracy (right) from the history dict."""
    ax1.clear()
    ax2.clear()

    ax1.plot(h["step"], h["train_bits"], label="train (avg of 50 steps)")
    ax1.plot(h["val_step"], h["val_bits"], "o-", label="val")
    ax1.axhline(UNIFORM_BITS, linestyle="--", color="gray", label=f"uniform ({UNIFORM_BITS})")
    for s in h["epoch_end"]:
        ax1.axvline(s, color="lightgray")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("bits per move")
    ax1.set_title(f"Loss  (latest val {h['val_bits'][-1]:.3f})")
    ax1.legend()

    ax2.plot(h["val_step"], [v * 100 for v in h["val_top1"]], "o-", color="tab:orange", label="val")
    ax2.axhline(UNIFORM_TOP1, linestyle="--", color="gray", label=f"random legal move ({UNIFORM_TOP1}%)")
    for s in h["epoch_end"]:
        ax2.axvline(s, color="lightgray")
    ax2.set_xlabel("training step")
    ax2.set_ylabel("top-1 accuracy (%)")
    ax2.set_title(f"How often the favourite move was played  (latest {h['val_top1'][-1]:.1%})")
    ax2.legend()


def main():
    plt.ion()                                   # interactive mode: the window doesn't block the loop
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    while plt.fignum_exists(fig.number):        # runs until you close the window
        try:
            with open(HISTORY) as f:
                h = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # not written yet, or caught half-written by train.py: try again shortly
            plt.pause(1)
            continue

        draw(h, ax1, ax2)
        fig.tight_layout()
        fig.savefig("training_curves.png", dpi=150)   # the PNG always matches the window
        plt.pause(REFRESH)                      # redraw the window and wait


if __name__ == "__main__":
    main()
