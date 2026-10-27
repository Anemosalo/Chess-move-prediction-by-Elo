"""Extracts Lichess games for given rating bands into the text format encode.py reads (written by Claude).

Output per band (in OUT_DIR):
  <name>.txt       one game per line, UCI moves separated by commas
  <name>_meta.csv  one row per line of the .txt: white_elo,black_elo,time_control,event,site

Filters: both players' Elo in [lo, hi); blitz only (Lichess definition: 180 <= base + 40*inc < 480 s);
Termination is "Normal" or "Time forfeit"; at least 2 plies; the PGN parses without errors.
Games are taken in file order, up to each band's limit. The bands don't overlap, so one pass gives the same
games per band as separate passes.

Usage: zstd -dc data/raw/lichess_2026-09_first1500MB.pgn.zst | python3 extract_games.py data/games_uci
       zstd -dc data/raw/lichess_2026-09_first1500MB.pgn.zst | python3 extract_games.py data/games_uci_joint joint
"""
import csv, io, os, sys
import chess.pgn

RANGES = [  # (name, lo, hi, how many games)
    ("lichess_2026-09_elo1000-1199_blitz", 1000, 1200, float("inf")),   # training: every game in the file
    ("lichess_2026-09_elo2000-2199_blitz", 2000, 2200, 50_000),         # few-games experiment
    ("lichess_2026-09_elo1200-1399_blitz", 1200, 1400, 5_000),          # evaluation-only bands
    ("lichess_2026-09_elo1400-1599_blitz", 1400, 1600, 5_000),
    ("lichess_2026-09_elo1600-1799_blitz", 1600, 1800, 5_000),
    ("lichess_2026-09_elo1800-1999_blitz", 1800, 2000, 5_000),
]

## multi-band experiment: 4 bands to train on (30,000 games each) and 5 bands it never trains on (5,000 games each, only for testing)
JOINT_RANGES = [
    ("lichess_2026-09_elo1000-1199_blitz", 1000, 1200, 30_000),   # train
    ("lichess_2026-09_elo1400-1599_blitz", 1400, 1600, 30_000),   # train
    ("lichess_2026-09_elo1800-1999_blitz", 1800, 2000, 30_000),   # train
    ("lichess_2026-09_elo2200-2399_blitz", 2200, 2400, 30_000),   # train
    ("lichess_2026-09_elo1200-1399_blitz", 1200, 1400, 5_000),    # held out, in between the training bands
    ("lichess_2026-09_elo1600-1799_blitz", 1600, 1800, 5_000),    # held out, in between
    ("lichess_2026-09_elo2000-2199_blitz", 2000, 2200, 5_000),    # held out, in between
    ("lichess_2026-09_elo800-999_blitz", 800, 1000, 5_000),       # held out, below all training bands
    ("lichess_2026-09_elo2400-2599_blitz", 2400, 2600, 5_000),    # held out, above all training bands
]
if len(sys.argv) > 2 and sys.argv[2] == "joint":
    RANGES = JOINT_RANGES

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
want = {n: k for n, _, _, k in RANGES}
got = {n: 0 for n, _, _, _ in RANGES}
files = {n: (open(os.path.join(out_dir, n + ".txt"), "w"), open(os.path.join(out_dir, n + "_meta.csv"), "w", newline="")) for n, _, _, _ in RANGES}
writers = {n: csv.writer(m) for n, (_, m) in files.items()}
for w in writers.values():
    w.writerow(["white_elo", "black_elo", "time_control", "event", "site"])
seen = 0
bad = 0


def headers_of(block):
    h = {}
    for line in block:
        if line.startswith("["):
            k, _, v = line[1:].partition(" ")
            h[k] = v.strip().rstrip("]").strip('"')
    return h


def handle(block):
    global seen, bad
    seen += 1
    h = headers_of(block)
    try:
        we, be = int(h["WhiteElo"]), int(h["BlackElo"])
        base, inc = (int(x) for x in h["TimeControl"].split("+"))
    except (KeyError, ValueError):
        return
    if not (180 <= base + 40 * inc < 480) or h.get("Termination") not in ("Normal", "Time forfeit"):
        return
    for name, lo, hi, _ in RANGES:
        if got[name] < want[name] and lo <= we < hi and lo <= be < hi:
            game = chess.pgn.read_game(io.StringIO("".join(block)))
            if game is None or game.errors:
                bad += 1
                return
            moves = [m.uci() for m in game.mainline_moves()]
            if len(moves) < 2:
                return
            txt, _ = files[name]
            txt.write(",".join(moves) + "\n")
            writers[name].writerow([we, be, h["TimeControl"], h.get("Event", ""), h.get("Site", "")])
            got[name] += 1
            return


block = []
for line in sys.stdin:
    if line.startswith("[Event ") and block:
        handle(block)
        block = []
        if all(got[n] >= want[n] for n in want):
            break
    block.append(line)
# The last block may be cut off by the partial download: drop it (it never reached a following [Event).

for t, m in files.values():
    t.close()
    m.close()
print(f"games scanned: {seen:,}; unparsable kept-candidates skipped: {bad}")
for n in want:
    print(f"{n}: {got[n]:,} / {want[n]}")
