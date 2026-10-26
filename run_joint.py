## Runs the whole multi-band experiment on one machine, step by step:
##   1. unpack the game files (joint_games.tar.gz, made on the Mac with: extract_games.py data/games_uci_joint joint)
##   2. encode every band into shards
##   3. train ONE model on 4 bands at once (every batch mixes the 4 bands)
##   4. measure it on all 9 bands + the rating sweep
## Every step is skipped if it is already done, so after a crash just run it again.
##
## Usage (from the repo folder):  python run_joint.py
##        python run_joint.py --quick --archive small.tar.gz     (tiny version, only to check that everything runs)
import os
import shutil
import subprocess
import sys
import tarfile
import time

import torch

TRAIN_BANDS = ["1000-1199", "1400-1599", "1800-1999", "2200-2399"]
HELD_OUT = ["1200-1399", "1600-1799", "2000-2199", "800-999", "2400-2599"] # never trained on: 3 in between, 2 outside
ALL_BANDS = sorted(TRAIN_BANDS + HELD_OUT, key=lambda b: int(b.split("-")[0]))
RUN = "joint_4bands"
EPOCHS = "2"
EVAL_EVERY = "1000"
SWEEP_RATINGS = [str(r) for r in range(800, 2601, 100)]

quick = "--quick" in sys.argv
archive = sys.argv[sys.argv.index("--archive") + 1] if "--archive" in sys.argv else "joint_games.tar.gz"
os.chdir(os.path.dirname(os.path.abspath(__file__))) # always work from the repo folder


def name(band):
    return f"lichess_2026-09_elo{band}_blitz"


def run(args, log_path=None): ## runs one of our python scripts with this same python, prints its output and saves it to a log
    print(">>> python", " ".join(args), flush=True)
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    env = dict(os.environ, PYTHONIOENCODING="utf-8") # same text encoding on Windows and Mac
    p = subprocess.Popen([sys.executable, "-u"] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", env=env)
    for line in p.stdout:
        print(line, end="", flush=True)
        if log:
            log.write(line)
            log.flush()
    p.wait()
    if log:
        log.close()
    if p.returncode != 0:
        raise SystemExit(f"FAILED: python {' '.join(args)} (exit code {p.returncode})")


t_start = time.time()
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "CPU ONLY")
print(f"device: {device}", flush=True)
if device == "CPU ONLY":
    print("WARNING: no GPU found, training will be very slow", flush=True)

#-------------#
# 1 + 2. Unpack and encode. The marker file says this exact data is ready (so old data with the same names is never used by mistake)
marker = os.path.join("data", "joint_ready_" + os.path.basename(archive) + ".txt")
if not os.path.exists(marker):
    print(f"\n== 1. unpacking {archive} into data/games_uci", flush=True)
    os.makedirs(os.path.join("data", "games_uci"), exist_ok=True)
    with tarfile.open(archive) as t:
        t.extractall(os.path.join("data", "games_uci"))

    print("\n== 2. encoding every band (2 at a time)", flush=True)
    os.makedirs(os.path.join("data", "games_encoded"), exist_ok=True)
    for band in ALL_BANDS: # remove old shards of these bands so they are made again from the unpacked games
        for f in os.listdir(os.path.join("data", "games_encoded")):
            if f.startswith(name(band) + "_"):
                os.remove(os.path.join("data", "games_encoded", f))
    for i in range(0, len(ALL_BANDS), 2):
        procs = []
        for band in ALL_BANDS[i:i + 2]:
            log = open(os.path.join("logs", f"encode_{band}.log"), "w", encoding="utf-8")
            procs.append((band, log, subprocess.Popen([sys.executable, "-u", "encode.py", name(band)], stdout=log, stderr=subprocess.STDOUT)))
        for band, log, p in procs:
            p.wait()
            log.close()
            lines = open(os.path.join("logs", f"encode_{band}.log"), encoding="utf-8", errors="replace").read().splitlines()
            for line in lines:
                if not line.startswith("  saved"):
                    print(line, flush=True)
            if p.returncode != 0:
                raise SystemExit(f"FAILED: encoding {band}, see logs/encode_{band}.log")
    with open(marker, "w") as f:
        f.write("encoded from " + archive + "\n")
else:
    print("\n== 1 + 2. data already unpacked and encoded, skipping", flush=True)
print(f"({(time.time() - t_start) / 60:.0f} min so far)", flush=True)

#-------------#
# 3. Train one model on the 4 bands at once, measuring the held-out bands too (they never pick the checkpoint)
run_name = RUN + ("_quick" if quick else "")
run_dir = os.path.join("runs", run_name)
if not os.path.exists(os.path.join(run_dir, "test.json")):
    if os.path.exists(run_dir): # a run that crashed halfway: keep it aside and start again
        shutil.move(run_dir, run_dir + "_crashed_" + time.strftime("%H%M%S"))
    print(f"\n== 3. training {run_name} on {TRAIN_BANDS}", flush=True)
    args = ["train.py", "--data"] + TRAIN_BANDS + ["--also"] + HELD_OUT + ["--test", "--out", run_name]
    if quick:
        args += ["--games", "20", "--epochs", "1", "--eval-every", "5"]
    else:
        args += ["--epochs", EPOCHS, "--eval-every", EVAL_EVERY]
    run(args, os.path.join("logs", run_name + ".log"))
else:
    print(f"\n== 3. {run_name} already trained, skipping", flush=True)
print(f"({(time.time() - t_start) / 60:.0f} min so far)", flush=True)

#-------------#
# 4. Every band with the real ratings, and the rating sweep (is the rating plane a skill dial now?)
out = os.path.join("results", "bands_eval_" + run_name + ".csv")
if not os.path.exists(out):
    print("\n== 4. measuring every band + rating sweep", flush=True)
    ratings = ["800", "1600", "2400"] if quick else SWEEP_RATINGS
    run(["eval_bands.py", "--models", f"{run_name}={run_dir}/best.pt", "--bands"] + ALL_BANDS
        + ["--sweep", run_name, "--ratings"] + ratings + ["--out", out], os.path.join("logs", "eval_" + run_name + ".log"))
else:
    print(f"\n== 4. {out} already exists, skipping", flush=True)

print(f"\nDONE in {(time.time() - t_start) / 60:.0f} min. Results: {run_dir}/ (history.json, test.json) and {out}", flush=True)
print("To bring them to the Mac: commit and push runs/" + run_name + ", " + out + " and logs/" + run_name + ".log", flush=True)
