# Tables

Made by `summarize.py` from `results/all_runs.csv`. Test positions (shard 000): 37,050 for 2000-2199, 28,654 for 1000-1199. ± = standard deviation over 3 seeds; no ± = 1 seed (seed 0). Bold = best option per row.

## 1. Strong players: test bits per move on 2000-2199 (lower is better)
Base model with 0 strong games: **3.290**. Uniform over legal moves: 4.694.

| N strong games | scratch | full fine-tune | BN + output layer |
|---|---|---|---|
| 100 | 4.323 ± 0.024 | **3.081 ± 0.006** | 3.144 ± 0.004 |
| 1,000 | 3.768 ± 0.024 | **2.971 ± 0.001** | 3.087 ± 0.002 |
| 10,000 | 3.052 | **2.834** | 3.074 |
| all (40,019) | 2.841 | **2.775** | 3.076 |

## 2. Strong players: top-1 accuracy (higher is better)
Base model: 33.4%. Random legal move: 5.9%.

| N strong games | scratch | full fine-tune | BN + output layer |
|---|---|---|---|
| 100 | 16.7% ± 0.6% | **36.7% ± 0.1%** | 35.8% ± 0.1% |
| 1,000 | 26.1% ± 0.4% | **38.6% ± 0.1%** | 36.6% ± 0.0% |
| 10,000 | 37.4% | **41.0%** | 36.8% |
| all (40,019) | 40.5% | **41.7%** | 36.7% |

## 3. Gain over the base model on strong players (bits per move, + = better)

| N strong games | scratch | full fine-tune | BN + output layer |
|---|---|---|---|
| 100 | -1.033 ± 0.024 | +0.209 ± 0.006 | +0.146 ± 0.004 |
| 1,000 | -0.478 ± 0.024 | +0.319 ± 0.001 | +0.203 ± 0.002 |
| 10,000 | +0.238 | +0.456 | +0.216 |
| all (40,019) | +0.449 | +0.515 | +0.214 |

## 4. Forgetting: change on weak players (1000-1199) after training on strong players (+ = worse)
Base model on weak players: 2.831 bits.

| N strong games | scratch | full fine-tune | BN + output layer |
|---|---|---|---|
| 100 | +1.924 ± 0.149 | +0.139 ± 0.025 | +0.155 ± 0.007 |
| 1,000 | +2.331 ± 0.794 | +0.262 ± 0.064 | +0.150 ± 0.006 |
| 10,000 | +0.779 | +0.157 | +0.154 |
| all (40,019) | +0.547 | +0.094 | +0.166 |

## 5. Strong-player games that pretraining saves
Games that training from scratch would need for the same test bits, interpolated linearly in log(N) between the measured scratch points (rough).

| Result | Bits | Scratch would need | Saving |
|---|---|---|---|
| base model, 0 games | 3.290 | ≈4,700 | ≈4,700 games for free |
| full fine-tune, 100 games | 3.081 | ≈9,100 | ≈91× |
| full fine-tune, 1,000 games | 2.971 | ≈17,000 | ≈17× |
| full fine-tune, 10,000 games | 2.834 | more than 40,019 (scratch on all 40,019 games: 2.841, 0.007 worse) | > 4.0× |
| full fine-tune, all (40,019) games | 2.775 | more than 40,019 (scratch on all 40,019 games: 2.841, 0.066 worse) | beats scratch on the same data |
| BN + output layer, 100 games | 3.144 | ≈7,400 | ≈74× |
| BN + output layer, 1,000 games | 3.087 | ≈8,900 | ≈8.9× |
| BN + output layer, 10,000 games | 3.074 | ≈9,300 | ≈0.93× |
| BN + output layer, all (40,019) games | 3.076 | ≈9,300 | ≈0.23× |

Scratch was not tuned for small N (lr 1e-3, no regularisation). There is one seed at 10,000 and all games, and the test set has 487 games, so a gap of a few thousandths of a bit is within test noise: "more than 40,019" for 10,000 games means about as good as scratch on all 40,019 games. With all games, both runs were still improving when they hit the pass limit.

## 6. Training cost and dynamics (means over seeds)

| N | Option | Training positions | Best checkpoint (step) | Last step | Passes | Minutes | Stopped early | Best = last eval (still improving) |
|---|---|---|---|---|---|---|---|---|
| 100 | scratch | 7,324 | 47 | 117 | 15.3 | 2.8 | 3/3 | 0/3 |
| 100 | full fine-tune | 7,324 | 84 | 154 | 20.3 | 3.6 | 3/3 | 0/3 |
| 100 | BN + output layer | 7,324 | 75 | 145 | 19.3 | 3.2 | 3/3 | 0/3 |
| 1,000 | scratch | 75,454 | 345 | 715 | 9.7 | 5.9 | 3/3 | 0/3 |
| 1,000 | full fine-tune | 75,454 | 567 | 937 | 12.7 | 8.1 | 3/3 | 0/3 |
| 1,000 | BN + output layer | 75,454 | 382 | 752 | 10.3 | 5.4 | 3/3 | 0/3 |
| 10,000 | scratch | 761,134 | 5,208 | 5,952 | 8.0 | 34.7 | 0/1 | 0/1 |
| 10,000 | full fine-tune | 761,134 | 5,952 | 5,952 | 8.0 | 34.1 | 0/1 | 1/1 |
| 10,000 | BN + output layer | 761,134 | 558 | 2,046 | 3.0 | 8.8 | 1/1 | 0/1 |
| all (40,019) | scratch | 3,006,796 | 8,823 | 8,823 | 3.0 | 46.8 | 0/1 | 1/1 |
| all (40,019) | full fine-tune | 3,006,796 | 8,823 | 8,823 | 3.0 | 46.6 | 0/1 | 1/1 |
| all (40,019) | BN + output layer | 3,006,796 | 500 | 3,500 | 2.0 | 13.6 | 1/1 | 0/1 |

## 7. Base model (trained on 1000-1199): all test positions

| Band | Positions | Bits per move | Top-1 | Uniform | Random move |
|---|---|---|---|---|---|
| 1000-1199 | 849,649 | **2.830** | 41.5% | 4.604 | 6.8% |
| 2000-2199 (never trained on) | 371,077 | **3.263** | 33.7% | 4.691 | 5.9% |

## 8. Learning-rate check (N = 100, seed 0)

| Run | Option | Learning rate | Test 2000-2199 | Test 1000-1199 | Best step | Last step |
|---|---|---|---|---|---|---|
| ftall_100_s0 | full fine-tune | 0.0001 | 3.085 | 2.947 | 70 | 140 (early stop) |
| ftall_100_s0_lr1e-3 | full fine-tune | 0.001 | 3.129 | 2.997 | 14 | 84 (early stop) |
| ftbn_100_s0 | BN + output layer | 0.001 | 3.139 | 2.979 | 70 | 140 (early stop) |
| ftbn_100_s0_lr1e-4 | BN + output layer | 0.0001 | 3.141 | 2.967 | 420 | 420 (cap) |
