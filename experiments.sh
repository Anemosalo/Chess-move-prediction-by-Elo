#!/bin/bash
# Fine-tuning experiment on 2000-2199. Run with:   caffeinate -i bash experiments.sh
# A run whose folder already exists is skipped, so the script can be stopped and restarted.
BASE=runs/base/best.pt
mkdir -p logs

run() {                                   # run <name> <options…>
  local name=$1; shift
  if [ -d "runs/$name" ]; then echo "skip $name (exists)"; return; fi
  echo "=== $name  $(date +%H:%M)"
  python3 train.py --data 2000-2199 --also 1000-1199 --test --out "$name" "$@" > "logs/$name.log" 2>&1 \
    && grep TEST "logs/$name.log" || echo "FAILED: $name (see logs/$name.log)"
}

schedule() {                              # the training settings for each N
  case $1 in
    100)   echo "--games 100   --eval-every 7   --patience 10 --epochs 60" ;;
    1000)  echo "--games 1000  --eval-every 37  --patience 10 --epochs 30" ;;
    10000) echo "--games 10000 --eval-every 186 --patience 8  --epochs 8" ;;
    all)   echo "--eval-every 500 --patience 6 --epochs 3" ;;
  esac
}

three_options() {                         # three_options <N> <seed>
  local sched=$(schedule $1)
  run scratch_$1_s$2 $sched --seed $2 --init scratch            --lr 1e-3
  run ftall_$1_s$2   $sched --seed $2 --init $BASE --tune all   --lr 1e-4
  run ftbn_$1_s$2    $sched --seed $2 --init $BASE --tune bn    --lr 1e-3
}

# N = 0: the base model unchanged, on the same val and test rows as every other run
[ -d runs/n0_base ] || python3 train.py --eval-only --init $BASE --data 2000-2199 --also 1000-1199 --test --out n0_base > logs/n0_base.log 2>&1

# Round 1: one seed of every N, a complete first curve
for n in 1000 100 10000 all; do three_options $n 0; done

# Round 2: more seeds where luck matters most
for seed in 1 2; do for n in 100 1000; do three_options $n $seed; done; done