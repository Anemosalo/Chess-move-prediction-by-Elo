# Results: predicting chess moves by rating, and learning strong players from a few games

All runs of the project in one place. The numbers are taken from the run folders by `collect_results.py`, which writes the CSV files in [section 9](#9-data-files). The tables in `results/tables.md` and the figures in `results/figures/` are made from those files (section 10). The four write-ups in `results/` tell the same story in a shorter form.

**Contents**
1. [The questions](#1-the-questions)
2. [Data](#2-data)
3. [Model and training setup](#3-model-and-training-setup)
4. [Run inventory](#4-run-inventory)
5. [Results: base model](#5-results-base-model-10001199)
6. [Results: the few-games experiment](#6-results-the-few-games-experiment-20002199)
7. [Findings](#7-findings)
8. [Limitations](#8-limitations)
9. [Data files](#9-data-files)
10. [Figures and scripts](#10-figures-and-scripts)
11. [Next steps](#11-next-steps)

---

## 1. The questions

1. **Base model.** Can a small neural network predict well the move a 1000–1199 rated player will choose in a blitz game? Two things were measured:
   - bits per move: −log₂(p), where p is the probability the model gave to the move that was played. This is (ideally) the number of bits a compressor using the model would need to store the move;
   - top-1 accuracy: the percentage of positions in which the model's top choice was the move actually played.
2. **Few games.** In chess, games between very high-rated players are much rarer than games at average ratings, so there is relatively little data to train on. Can we make up for the lack of data by pretraining on average-rated games first and then fine-tuning on stronger players (2000–2199)? How many strong-player games does pretraining save?
3. **Forgetting.** What does the adaptation to stronger players cost on the weak players the model was built for?
4. **The rating input.** How do the models perform on bands they never saw, and does the rating input actually act as a skill dial? What happens if we train one model on several bands at once, in the same batches?

---

## 2. Data

### 2.1 Source and filters

The data was gathered from the Lichess open database and consisted of the first 1.5 GB of the September 2026 games.

| Item | Value |
|---|---|
| Source | Lichess open database, standard rated games, September 2026 (CC0), file `lichess_db_standard_rated_2026-09.pgn.zst` |
| Part used | the first 1,500,000,000 bytes: **4,596,934 games** (file `data/raw/lichess_2026-09_first1500MB.pgn.zst`) |
| Time control | blitz: Lichess's estimated duration per player, base + 40 × increment, in [180, 480) seconds |
| Game end | ended normally or on time (Termination "Normal" or "Time forfeit"), so no abandoned games |
| Ratings | **both** players inside the band |
| Other | at least 2 plies; the PGN parses without errors |
| Split | by **game**, random, seed 0: 80% training / 10% validation / 10% test, so no game has positions in more than one split |

Every band was split with the same seed (0), so shard 000 of every band has the same split pattern: 3,973 / 540 / 487 games for training / validation / test.

The games cover about the first 1–2 days of September 2026. So the test games are played by the same players, over the same days, as the training games: most players in a band's test games also play in its training games.

### 2.2 The two bands

| Band      | Games                       | Positions | Shards | Train positions | Val positions | Test positions | Train games |
|-----------|-----------------------------|-----------|--------|-----------------|---------------|----------------|------------|
| 1000–1199 | 141,333 (all in the 1.5 GB) | 8,568,827 | 29     | 6,874,720       | 844,458       | 849,649        | 113,349    |
| 2000–2199 | 50,000 (capped)             | 3,755,357 | 10     | 3,006,796       | 377,484       | 371,077        | **40,019** |

### 2.3 The position sets used for measuring, and their baselines

**Uniform:** every legal move equally likely, so the bits are the mean of log₂ K, where K is the number of legal moves. **Random top-1:** the mean of 1/K.

| Band      | Split | Rows          | Games | Positions | Uniform bits | Random top-1 | Mean legal moves | Used for |
|-----------|-------|---------------|-------|-----------|--------------|--------------|------------------|-------------------------|
| 1000–1199 | val   | shard 000     | 540    | 33,344    | 4.580        | 6.92%        | 28.7             | validation during training (all runs) |
| 1000–1199 | test  | shard 000     | 487    | 28,654    | 4.604        | 6.90%        | 29.1             | test of the experiment runs |
| 1000–1199 | test  | all 29 shards | 13,979 | 849,649   | 4.604        | 6.82%        | 29.0             | test of the base model |
| 2000–2199 | val   | shard 000     | 540    | 42,433    | 4.679        | 6.03%        | 29.9             | validation during training (all runs) |
| 2000–2199 | test  | shard 000     | 487    | 37,050    | 4.694        | 5.91%        | 30.0             | test of the experiment runs |
| 2000–2199 | test  | all 10 shards | 4,912  | 371,077   | 4.691        | 5.92%        | 30.0             | test of the base model |

**Encoded shard format:** `planes` (N,17,8,8) uint8, `y`, `legal_flat` + `offsets`, `rating`, `game_id`, `split`.

---

## 3. Model and training setup

### 3.1 The network (the same in every run)

The input layer receives the board from the view of the player to move (the board is mirrored for Black) as an [18, 8, 8] tensor: 17 planes for the pieces, the castling rights and en passant, plus one plane with the rating of the player to move, scaled to (Elo − 1500)/400. The model outputs one score for every possible from-square/to-square combination plus 72 underpromotions, which comes out to 4,168 possible moves. Illegal moves are masked out before the softmax, and the loss is calculated as cross-entropy in bits.

| Part | What it is | Shape |
|---|---|---|
| Input | 17 board planes from the mover's view, plus 1 rating plane = (Elo − 1500)/400 of the player to move | (B, 18, 8, 8) |
| Stem | 3×3 conv 18→64, no bias, BatchNorm, ReLU | (B, 64, 8, 8) |
| Body | 6 residual blocks: conv 3×3 → BN → ReLU → conv 3×3 → BN → + input → ReLU | (B, 64, 8, 8) |
| Head | 1×1 conv 64→73 (with bias). Channels 0–63 = "to square" at each from-square, permuted to index 64·from + to; channels 64–72 = underpromotions, read on the 7th rank | (B, 4168) |
| Mask | illegal moves set to −∞ before the softmax | |
| Loss | cross-entropy ÷ ln 2 = **bits per move** | |
| Weights | **459,145** in total; BN scale/shift = 1,664, output layer = 4,745 | |

### 3.2 Training settings

| Setting | Value |
|---|---|
| Optimizer | Adam, default betas, no weight decay, no learning-rate schedule |
| Batch | 1,024 positions; shards visited in random order, rows shuffled within each shard |
| Device | Apple M1 GPU (MPS) for all runs except `joint_4bands`; ≈0.23 s per training step alone, ≈0.36 s with evaluations |
| Model selection | evaluate every `eval_every` steps on the **training band's** validation rows; save `best.pt` when the validation bits improve |
| Early stopping | stop after `patience` evaluations without improvement |
| Test | after training, load `best.pt` and measure the test rows **once** (both bands) |
| Seeds | the seed sets the game selection (numpy), the initial weights and the shuffling. For a given seed, all three options use **the same games** |

### 3.3 The three options compared

| Option | Starts from | Weights trained | Learning rate |
|---|---|---|---|
| **scratch** | random weights | all 459,145 | 10⁻³ |
| **full fine-tune** (`ftall`) | the base model (`runs/base/best.pt`) | all 459,145 | 10⁻⁴ |
| **BN + output layer** (`ftbn`) | the base model (`runs/base/best.pt`) | only the BatchNorm scale/shift (1,664) and the output layer (4,745): 6,409 | 10⁻³ |

### 3.4 Schedule per N (2000–2199 games used for training)

| N | Training positions (seed 0 / 1 / 2) | Steps per pass | eval_every | patience | max passes | Seeds |
|---|---|---|---|---|---|---|
| 100 | 7,136 / 7,586 / 7,250 | 7–8 | 7 | 10 | 60 | 0, 1, 2 |
| 1,000 | 75,656 / 75,720 / 74,987 | 74 | 37 | 10 | 30 | 0, 1, 2 |
| 10,000 | 761,134 | 744 | 186 | 8 | 8 | 0 |
| all (40,019) | 3,006,796 | 2,937 | 500 | 6 | 3 | 0 |

---

## 4. Run inventory

All runs are in `runs/<name>/`; `results/all_runs.csv` has one row per run.

| Run(s) | What | Data | Test rows |
|---|---|---|---|
| `run1_1000-1199_2ep` | first model: scratch, 2 passes, learning rate 10⁻³. Replaced by `base`; it used older code and an older version of the data, so it is not compared with the other runs | 1000–1199, first 46,802 games (older data, no longer on disk) | none |
| `base` | **the base model**: scratch, 1 pass, learning rate 10⁻³, evaluation every 500 steps | 1000–1199, all 141,333 games | all test rows (from `base_test`) |
| `base_test` | evaluation only of `base/best.pt` with `--test-shards 99` | | all test rows, both bands |
| `n0_base` | evaluation only of `base/best.pt`: the **N = 0** point | | shard 000, both bands |
| `scratch_<N>_s<seed>` | scratch on N games of 2000–2199 | 2000–2199 | shard 000, both bands |
| `ftall_<N>_s<seed>` | full fine-tune of the base model on the same games | 2000–2199 | shard 000, both bands |
| `ftbn_<N>_s<seed>` | BN + output layer fine-tune of the base model on the same games | 2000–2199 | shard 000, both bands |
| `ftall_100_s0_lr1e-3`, `ftbn_100_s0_lr1e-4` | learning-rate checks (each with the other option's learning rate) | 2000–2199, 100 games, seed 0 | shard 000, both bands |
| `joint_4bands` | **one model trained on 4 bands at once**: scratch, every batch = 256 positions from each band, 2 epochs, learning rate 10⁻³, evaluation every 1,000 steps | 1000–1199, 1400–1599, 1800–1999, 2200–2399 (30,000 games each) | shard 000 of all 9 bands |

That came out to 24 experiment runs, 2 learning-rate checks and 1 N = 0 measurement. The 24 experiment runs took 4.53 GPU-hours on the Apple M1 (4.69 with the 2 learning-rate checks).

---

## 5. Results: base model (1000–1199)

### 5.1 Training

Adam with a learning rate of 10⁻³, batches of 1,024 positions and one pass over the 141,333 games (6.87 million training positions). That came out to 6,729 steps, which took 28 minutes on an Apple M1. The checkpoint with the best validation score was kept: 2.820 bits and 41.5% top-1 on the 1000–1199 validation rows (shard 000). The model was still improving when it stopped, with the last 3 validations at 2.841 → 2.824 → 2.820, so the best checkpoint is the last evaluation (figure F6).

### 5.2 Base model test (all test rows, measured only once)

| Band | Positions | Base model | Uniform | Bits saved | Top-1 | Random top-1 |
|---|---|---|---|---|---|---|
| 1000–1199 (its own band) | 849,649 | **2.830 bits** | 4.604 | **38.5%** | **41.5%** | 6.82% |
| 2000–2199 (never seen in training) | 371,077 | **3.263 bits** | 4.691 | 30.4% | 33.7% | 5.92% |

- On its training band of 1000–1199, the model gets 41.5% top-1 accuracy and needs 38.5% fewer bits per move than choosing uniformly.
- Tested on strong players (the 2000–2199 band), it gets 33.7% top-1 accuracy. Its performance dropped a lot compared with its training band, but that is expected since it never saw any other band.
- Its predictions for 2000–2199 also jumped around during training: in the second half, the 2000–2199 validation went up and down between 3.35 and 3.71, while the 1000–1199 validation was smooth. Section 6.8 shows that the rating plane costs the base model 0.20 bits on 2000–2199; whether it also causes these jumps was not tested.
- On the test rows of shard 000, the same model scores 3.290 on 2000–2199 and 2.831 on 1000–1199. That is the N = 0 point in section 6.

---

## 6. Results: the few-games experiment (2000–2199)

N is the number of strong-player games the model was trained on, which goes from 100 → 1,000 → 10,000 and finally all 40,019. For each N, the 3 options of section 3.3 were trained on the same games. N = 100 and 1,000 were repeated with 3 random seeds, which means different games and shuffling.

The test set was measured once, at the end of training, on shard 000: 37,050 strong-player positions (from 487 games) and 28,654 weak-player positions, in order to also measure how much the model forgot about weak players' moves. ± = std over 3 seeds; rows without ± have one seed (seed 0). The spread over seeds does not include test-set noise, since every option is measured on the same 487 test games.

### 6.1 Main result: predicting strong players (test bits per move on 2000–2199, lower is better)

| N strong games | scratch | **full fine-tune** | BN + output layer |
|---|---|---|---|
| 0 (base model unchanged) | | **3.290** | 3.290 |
| 100 | 4.323 ± 0.024 | **3.081 ± 0.006** | 3.144 ± 0.004 |
| 1,000 | 3.768 ± 0.024 | **2.971 ± 0.001** | 3.087 ± 0.002 |
| 10,000 | 3.052 | **2.834** | 3.074 |
| all 40,019 | 2.841 | **2.775** | 3.076 |
| *uniform* | *4.694* | | |

Every seed, from min to max: scratch 100 [4.300, 4.348], 1,000 [3.749, 3.795]; full fine-tune 100 [3.074, 3.085], 1,000 [2.970, 2.972]; BN + output layer 100 [3.139, 3.147], 1,000 [3.086, 3.089]. For the fine-tunes, the three seeds differed by at most 0.011 bits (full fine-tune, N = 100), with a std of at most 0.006.

### 6.2 Top-1 accuracy on 2000–2199 (higher is better)

| N | scratch | full fine-tune | BN + output layer |
|---|---|---|---|
| 0 | | 33.4% (base) | 33.4% |
| 100 | 16.7% ± 0.6 | **36.7% ± 0.1** | 35.8% ± 0.1 |
| 1,000 | 26.1% ± 0.4 | **38.6% ± 0.1** | 36.6% ± 0.0 |
| 10,000 | 37.4% | **41.0%** | 36.8% |
| all | 40.5% | **41.7%** | 36.7% |
| *random legal move* | *5.91%* | | |

### 6.3 Gain over the base model on strong players (base 3.290 − result, bits per move)

| N | scratch | full fine-tune | BN + output layer |
|---|---|---|---|
| 100 | −1.033 | +0.209 | +0.146 |
| 1,000 | −0.478 | +0.319 | +0.203 |
| 10,000 | +0.238 | +0.456 | +0.216 |
| all | +0.449 | **+0.515** | +0.214 |

### 6.4 Weak players after training on strong players (test bits on 1000–1199; base model before fine-tuning: 2.831, top-1 41.3%)

| N | scratch | full fine-tune | BN + output layer |
|---|---|---|---|
| 100 | 4.755 ± 0.149 | 2.970 ± 0.025 (**+0.139**) | 2.986 ± 0.007 (+0.155) |
| 1,000 | 5.162 ± 0.794 | 3.093 ± 0.064 (**+0.262**) | 2.981 ± 0.006 (+0.150) |
| 10,000 | 3.611 | 2.988 (+0.157) | 2.986 (+0.154) |
| all | 3.379 | 2.925 (**+0.094**) | 2.997 (+0.166) |

- The value in brackets is how much the model forgot, against the base model's 2.831.
- The model forgot little, and the forgetting did not increase with more strong-player data. It did not fall steadily either, and the spread at 1,000 games was large (std ± 0.064).
- The models trained from scratch on 2000–2199 predicted weak players' moves wrongly with high confidence: 3.4 to 6.0 bits per run, worse than uniform at small N.
- Top-1 on weak players: full fine-tune 38.5 / 35.3 / 37.9 / 39.3%; BN + output layer 38.3 / 38.2 / 38.1 / 37.4%; scratch 11.7 / 13.7 / 28.8 / 30.4% (for N = 100 / 1,000 / 10,000 / all).
- **Forgetting per bit gained** (the 6.4 bracket ÷ the 6.3 gain): full fine-tune 0.66 / 0.82 / 0.34 / 0.18; BN + output layer 1.06 / 0.74 / 0.71 / 0.78.

### 6.5 How many strong-player games pretraining is worth

The scratch results (4.323, 3.768, 3.052 and 2.841 at 100, 1,000, 10,000 and 40,019 games) are interpolated linearly in log₁₀(N) to find how many games scratch would need to match each result. These are rough estimates: 4 points, and one seed above 1,000.

The scratch models are trained with the same settings as the fine-tunes: the full network at a fixed learning rate of 10⁻³, with no regularisation, which was not tuned for small N. At N = 100, their best checkpoint comes after 42–56 steps, at 4.30–4.35 bits, close to uniform (4.69). Every option also picks its checkpoint on the same 42,433 validation positions from 540 strong-player games, which is more games than the N = 100 training set. So the savings below are measured against a weak baseline.

| Result | Bits | Scratch needs ≈ | Games actually used | Saving |
|---|---|---|---|---|
| base model, 0 strong games | 3.290 | **≈4,700 games** | 0 | pretraining alone matches scratch on ~4,700 strong-player games |
| full fine-tune, 100 games | 3.081 | ≈9,100 | 100 | **≈91×** |
| full fine-tune, 1,000 games | 2.971 | ≈17,000 | 1,000 | **≈17×** |
| full fine-tune, 10,000 games | 2.834 | about 40,019 | 10,000 | **about 4×** (0.007 bits better than scratch on all 40,019 games: one seed each, within test noise) |
| full fine-tune, all | 2.775 | > 40,019 | 40,019 | 0.066 bits better than scratch on the same data, after 3 passes (both still improving) |
| BN + output layer, 100 / 1,000 games | 3.144 / 3.087 | ≈7,400 / ≈8,900 | 100 / 1,000 | ≈74× / ≈9× |
| BN + output layer, 10,000 / all | 3.074 / 3.076 | ≈9,300 | 10,000 / 40,019 | none: worse than scratch on the same data |

### 6.6 Training dynamics and cost (means over seeds)

| Option | N | Best checkpoint at step | Last step | Passes done | Minutes | Early stopped | Best = last evaluation? |
|---|---|---|---|---|---|---|---|
| scratch | 100 | 47 | 117 | 15.3 | 2.8 | 3/3 | 0/3 |
| scratch | 1,000 | 345 | 715 | 9.7 | 5.9 | 3/3 | 0/3 |
| scratch | 10,000 | 5,208 | 5,952 | 8 (cap) | 34.7 | 0/1 | 0/1 |
| scratch | all | 8,823 | 8,823 | 3 (cap) | 46.8 | 0/1 | **1/1 (still improving)** |
| full | 100 | 84 | 154 | 20.3 | 3.6 | 3/3 | 0/3 |
| full | 1,000 | 567 | 937 | 12.7 | 8.1 | 3/3 | 0/3 |
| full | 10,000 | 5,952 | 5,952 | 8 (cap) | 34.1 | 0/1 | **1/1 (still improving)** |
| full | all | 8,823 | 8,823 | 3 (cap) | 46.6 | 0/1 | **1/1 (still improving)** |
| BN + output layer | 100 | 75 | 145 | 19.3 | 3.2 | 3/3 | 0/3 |
| BN + output layer | 1,000 | 382 | 752 | 10.3 | 5.4 | 3/3 | 0/3 |
| BN + output layer | 10,000 | 558 | 2,046 | 3 | 8.8 | 1/1 | 0/1 |
| BN + output layer | all | 500 | 3,500 | 2 | 13.6 | 1/1 | 0/1 |

- At 100 and 1,000 games, the runs show the usual overfitting pattern: the training loss keeps falling after the best checkpoint while the validation rises.
- The BN + output layer runs at 10,000 and all games are different: their training loss rose as well. For `ftbn_all_s0` it went from 3.03 near step 500 to 3.47 at step 3,500, within 1.2 passes, so mostly on games it had not seen before, and the validation followed (3.046 → 3.497). So the optimization did not converge at a learning rate of 10⁻³. This is not overfitting, and these two runs do not show the limit of the option.
- At 1,000 games, the training loss of BN + output layer stays flat at about 3.03 for 10 passes, which does suggest limited capacity, but only at N ≤ 1,000.

### 6.7 Learning-rate check (N = 100, seed 0)

| Run | Learning rate | Test 2000–2199 | Test 1000–1199 | Best checkpoint at step | Last step |
|---|---|---|---|---|---|
| full fine-tune (experiment) | **10⁻⁴** | **3.085** | 2.947 | 70 | 140 (early stop) |
| full fine-tune (check) | 10⁻³ | 3.129 | 2.997 | 14 | 84 (early stop) |
| BN + output layer (experiment) | 10⁻³ | 3.139 | 2.979 | 70 | 140 (early stop) |
| BN + output layer (check) | 10⁻⁴ | 3.141 | 2.967 | 420 | 420 (60-pass cap, still improving) |

BN + output layer ends at about 3.14 with either learning rate (10⁻⁴ is only slower). Full fine-tuning is better at 10⁻⁴ than at 10⁻³, which overfits after 2 passes. **So at N = 100 the gap between the two options does not come from the learning rate.** The check was only run at N = 100 with one seed, so it does not tell whether the larger-N BN runs would converge at 10⁻⁴.

### 6.8 Testing on all rating bands, and what the rating input actually learned (`eval_bands.py`, figures F8 and F9)

The models so far were trained on only one or two bands (1000–1199 and 2000–2199). 5,000 more Lichess blitz games were added for each of the bands 1200–1399, 1400–1599, 1600–1799 and 1800–1999 (blitz, both players in the band, from the same 1.5 GB). They are used for evaluation, **not for training**. Every band is measured on the test games of its first shard (28,654 to 37,050 positions per band), with the real ratings.

**Test bits per move with the real ratings (lower is better):**

| Model | 1000–1199 | 1200–1399 | 1400–1599 | 1600–1799 | 1800–1999 | 2000–2199 |
|---|---|---|---|---|---|---|
| base (1000–1199 only) | **2.831** | **2.881** | 2.921 | 2.993 | 3.102 | 3.290 |
| full fine-tune, 1,000 games | 3.045 | 2.992 | 2.933 | 2.899 | 2.906 | 2.970 |
| full fine-tune, 10,000 games | 2.988 | 2.932 | 2.869 | 2.815 | 2.795 | 2.834 |
| full fine-tune, all 40,019 | 2.925 | 2.883 | **2.819** | **2.766** | **2.744** | **2.775** |
| scratch on 2000–2199 only | 3.379 | 3.177 | 3.000 | 2.904 | 2.831 | 2.841 |
| *uniform over legal moves* | *4.604* | *4.640* | *4.675* | *4.687* | *4.707* | *4.694* |

Top-1 for the full fine-tune on all games: 39.3 / 39.8 / 40.7 / 41.9 / 42.1 / 41.7%. For the base model: 41.3 / 40.5 / 39.5 / 38.5 / 36.4 / 33.4%.

**Rating test (sweep):** each band's validation positions were measured again by each model, with the rating plane set to one value for every position (1000, 1100, … 2200), to find the rating that predicts that band best:

| Model | 1000–1199 | 1200–1399 | 1400–1599 | 1600–1799 | 1800–1999 | 2000–2199 | Effect of the plane (max − min bits) |
|---|---|---|---|---|---|---|---|
| base | 1100 | 1100 | 1200 | 1200 | 1200 | 1200 | 0.27–0.29 |
| full fine-tune, all | 2000 | 2000 | 2000 | 2000 | 2100 | 2100 | 0.10–0.16 |
| scratch on 2000–2199 | 2100 | 2100 | 2000 | 2000 | 2100 | 2100 | 0.45–0.59 |

The bottom of each sweep curve is flat (the best and second-best ratings differ by at most 0.002 bits), so only where the curve sits is meaningful, not the exact best rating.

**Conclusion:**
- **The base model:** its predictions degrade smoothly as players get stronger, with its top-1 accuracy falling from 41.3% to 33.4%.
- **The fine-tuned model:** it is the best of all the models tested on the 1400–1999 range (2.744–2.819 bits), although no model trained on those bands. It beats both single-band models there and pretty much ties the base model on 1200–1399 (2.883 vs 2.881).
- **The main finding, though, is that each model predicts best when given the ratings it was trained on,** even when the real ratings are different: the base model prefers ≈1100–1200 for every band, and the other two ≈2000–2100. That means the rating plane acts as "my training data" vs "unfamiliar input". Maia (McIlroy-Young et al., KDD 2020) reported the same for its per-level models: each predicts best near its own training rating.
- **Consequences:**
  - The base model's predictions for strong players get worse when it is told their real ratings. Told 1200, it scores 3.066 validation bits, compared with 3.270 when told their real ratings: the rating plane costs it 0.20 bits.
  - Most of the fine-tuned model's forgetting on weak players comes from the rating input. On the 1000–1199 validation rows it scores 2.921 bits with the real ratings and 2.849 when told 2000, against the base model's 2.820, so about 0.07 of its 0.10 bits of forgetting.
  - Learning two bands one after the other does not teach the model the meaning of the rating plane input. That is probably because no batch ever contained both rating levels, so nothing required the network to tell them apart. Section 6.9 tests training on several bands at once.

### 6.9 One model trained on several bands at once (`joint_4bands`, `run_joint.py`, figure F10)

Section 6.8 showed that training the model on different bands in sequence, rather than at the same time, did not teach it the meaning of the rating plane. So here one model is trained on several bands at once, in the same batches, and then tested on bands it was never trained on. This was the idea behind Maia-2 (Tang et al., NeurIPS 2024): one model conditioned on skill instead of one model per rating level. Maia-2 gives the network both players' ratings through skill-aware attention; here the only skill input is one rating plane for the player to move.

**Setup:**
- **Training data:** 4 bands, 1000–1199, 1400–1599, 1800–1999 and 2200–2399, with 30,000 Lichess blitz games each (September 2026). That came out to 1,445,260 / 1,608,793 / 1,763,434 / 1,893,973 training positions, 6,711,460 in total.
- **Every batch of 1,024 positions takes 256 from each band,** so the network never sees one rating level on its own. When a band's shard ends in the middle of a batch, that batch gets fewer rows from that band; this happens a few times per pass.
- **Training:** the same network (459,145 weights) from scratch, for 2 epochs of 6,554 steps (13,108 steps). The smaller bands are seen more often: 3 passes started for 1000–1199 and 1400–1599, and 2 for the larger ones.
- **The checkpoint is chosen by the mean validation bits of the 4 training bands only.**
- **Evaluation-only data:** 5,000 games each for the bands 1200–1399, 1600–1799 and 2000–2199, which are the gaps between the training bands, and 800–999 and 2400–2599, which lie outside them.
- **Same rows as before:** shard 000 (the first 5,000 games) of every band is identical to the earlier runs' shard 000, so the validation and test rows are the same as in sections 6.1–6.8.
- **Hardware:** a Windows desktop (GTX 1650 Super, i3-9100F). Training took 31 minutes (0.14 s per step; CPU-bound, the GPU was mostly idle).
- **Best checkpoint:** step 13,000, with a mean validation of the training bands of 2.719, against 4.636 at step 0. The mean validation was still falling until then (2.741 → 2.727 → 2.719 over the last three evaluations); the final evaluation at the end of epoch 2 (step 13,108) was 2.726.

**Test bits per move on every band (shard 000 test rows, measured once):**

| Band | Positions | Joint model | Top-1 | Uniform | Bits saved vs uniform | Best earlier model on this band |
|---|---|---|---|---|---|---|
| 800–999 *(never seen, below)* | 27,609 | 2.883 | 40.7% | 4.580 | 37.1% | |
| 1000–1199 | 28,654 | **2.751** | 42.5% | 4.604 | 40.2% | 2.831 (base, 141,333 games of this band) |
| 1200–1399 *(never seen)* | 30,823 | **2.754** | 42.0% | 4.640 | 40.6% | 2.881 (base) |
| 1400–1599 | 32,411 | **2.726** | 42.6% | 4.675 | 41.7% | 2.819 (full fine-tune on all 2000–2199) |
| 1600–1799 *(never seen)* | 34,054 | **2.710** | 42.7% | 4.687 | 42.2% | 2.766 (full fine-tune on all 2000–2199) |
| 1800–1999 | 34,887 | **2.712** | 42.5% | 4.707 | 42.4% | 2.744 (full fine-tune on all 2000–2199) |
| 2000–2199 *(never seen)* | 37,050 | **2.750** | 41.8% | 4.694 | 41.4% | 2.775 (full fine-tune on **40,019 games of this band**) |
| 2200–2399 | 37,760 | 2.797 | 40.8% | 4.719 | 40.7% | |
| 2400–2599 *(never seen, above)* | 38,906 | 2.799 | 40.8% | 4.663 | 40.0% | |

**Rating input sweep (validation rows, the rating plane set to 800, 900, … 2600 for every position):**

| Band | Real mean rating | Rating that predicts it best | Ratings within 0.002 bits of the best | Real ratings vs best fixed rating | Effect of the plane (max − min bits) |
|---|---|---|---|---|---|
| 800–999 *(never seen)* | 914 | 1000 | 900–1100 | +0.000 | 0.192 |
| 1000–1199 | 1105 | 1200 | 1100–1300 | +0.001 | 0.152 |
| 1200–1399 *(never seen)* | 1299 | 1400 | 1300–1500 | +0.001 | 0.118 |
| 1400–1599 | 1503 | 1600 | 1500–1700 | +0.001 | 0.085 |
| 1600–1799 *(never seen)* | 1695 | 1700 | 1600–1800 | −0.000 | 0.105 |
| 1800–1999 | 1893 | 2000 | 1900–2100 | −0.000 | 0.147 |
| 2000–2199 *(never seen)* | 2087 | 2200 | 2100–2300 | +0.001 | 0.180 |
| 2200–2399 | 2289 | 2300 | 2200–2400 | −0.000 | 0.217 |
| 2400–2599 *(never seen)* | 2485 | 2400 | 2300–2500 | +0.001 | 0.219 |

**Conclusion:**
1. **The best rating to assume now rises with the band's actual rating.**
   - Each band's sweep curve is a clear U (0.09–0.22 bits from bottom to top), and the U moves with the band. This is also true for the three in-between bands the model never saw. For the single-band models it did not move (section 6.8).
   - Telling the model the real rating is now as good as the best fixed rating (within 0.001 bits), whereas before, the models performed best at one fixed rating for every band, and the real ratings cost up to 0.20 bits.
   - The bottom of each U is flat: the best and second-best ratings differ by only 0.0000–0.0013 bits. So the exact best rating is not precise at a step of 100 points; the 200-point window of ratings within 0.002 bits of the best contains, or lies just above, the real mean for every band.
   - Stronger bands also play faster time controls and longer games, and the rating plane absorbs those differences too, so it is not yet a pure skill input.
2. **Never-seen bands score well and close to their trained neighbours** on the test positions:
   - 1600–1799 (2.710) is as good as its trained neighbours or better (2.726 and 2.712);
   - 2000–2199 (2.750) is between its neighbours (2.712 and 2.797);
   - 1200–1399 (2.754) is slightly worse than both of its neighbours (2.751 and 2.726).

   The bands differ in difficulty, though (uniform bits 4.58–4.72, different game lengths and time controls), so this does not show how much was lost by not training on a band. That needs the same run with the band included.
3. **The joint-bands model has the best performance of all the models on every band they share,** for example 2.750 on 2000–2199 against 2.775 for the model fine-tuned on 40,019 games of that band. The comparison is not entirely fair, though:
   - it trained for 13,108 steps, about 2× the base model's 6,729 (which was still improving), so on 1000–1199 it is partly a comparison of training length;
   - it trained on 60,000 games of the two bands next to 2000–2199 (1800–1999 and 2200–2399);
   - some players of the never-seen bands also play in the training bands' games;
   - it is one seed, with no intervals for the differences.
4. **Outside the training range the bits are the highest:** 2.883 on 800–999 and 2.799 on 2400–2599. 800–999 also fell further behind during training (`runs/joint_4bands/training_curves.png`).
5. **The answer to section 6.8:** learning rating levels together, in the same batches, makes the model learn what the rating input represents. Maia-2 found the same on a larger scale.

Top-1 accuracy is 40.7–42.7% on all 9 bands.

---

## 7. Findings

1. **The base model** gets 2.830 bits per move and 41.5% top-1 on its training band of 1000–1199 (all 849,649 test positions): 38.5% fewer bits than choosing uniformly (4.604 bits / 6.8%).
2. **Pretraining on weak players is worth a lot of strong-player games,** compared with training from scratch with the same settings. The base model alone scores as well on 2000–2199 as a model trained from scratch on ~4,700 strong-player games. Fine-tuned on only N = 100 games, it matches scratch on ~9,100 games (about 90× fewer), and on N = 1,000 games scratch on ~17,000 (about 17×). The scratch baseline was not tuned for small N, and every option picks its checkpoint on a large validation set (540 strong-player games), which helps every small-N result (section 6.5).
3. **Full fine-tuning beat the from-scratch models at every N.** The advantage shrank as N got larger: 1.24 bits at N = 100 and 0.07 bits at all 40,019 games. At 10,000 games it is about as good as scratch on all 40,019 (0.007 bits apart, one seed each), and with all games both runs were still improving after 3 passes.
4. **The results did not depend much on which games were randomly selected:** for the fine-tunes, the three seeds differed by at most 0.011 bits, with a std of at most 0.006. This does not include test-set noise (487 test games).
5. **Fine-tuning only the BatchNorm scale and shift and the output layer is not enough at small N.** This option covers about 1.4% of the weights (6,409), most of them in the output layer. At 100 and 1,000 games it leveled off at 3.09–3.14 bits with a flat training loss. At 10,000 games or more its training loss rose, so the optimization did not converge at a learning rate of 10⁻³, and those runs do not show its limit. At N = 100, the gap to full fine-tuning does not come from the learning rate (section 6.7).
6. **The model forgot little, and the forgetting did not increase with more strong-player data.** On weak players, full fine-tuning loses 0.14 bits for N = 100, 0.26 for N = 1,000, 0.16 for N = 10,000 and 0.09 for all games, against gains of 0.21 to 0.52 bits on strong players. The spread at 1,000 games was large (std ± 0.064), and 10,000 and all games were run with only one seed. BN + output layer forgets a steady 0.15–0.17 bits for a smaller gain.
7. **The models trained from scratch on 2000–2199 predicted weak players' moves wrongly with high confidence** (3.4 to 6.0 bits, worse than uniform at small N), partly because their rating plane is outside its training range there.
8. **After fine-tuning on strong players, the model is the best of all the models tested on 1400–1999** (2.744–2.819 bits, top-1 up to 42.1%), although no model trained on those bands (section 6.8).
9. **Each single-band model predicts best when given the ratings it was trained on,** as Maia's per-level models do: the rating plane acts as "my training data" vs "unfamiliar input". The base model loses 0.20 bits on 2000–2199 when told the real ratings, and most of the fine-tuned model's forgetting on 1000–1199 comes from the rating input.
10. **Trained on 4 bands at once, in the same batches, the best rating to assume rises with the band's actual rating,** also for the three in-between bands the model never saw, and the real rating is as good as the best fixed rating (section 6.9). The bottom of each sweep curve is flat, so only where the curve sits is reliable, and the rating plane also absorbs time control and game length. Maia-2 found the same on a larger scale.
11. **The joint-bands model has the best performance of all the models on every band they share** (2.710–2.754 bits on 1000–2199), but the comparison is not entirely fair: about 2× the training steps, neighbouring bands in training, shared players and one seed (section 6.9).

---

## 8. Limitations

1. **One seed** for N = 10,000 and all, and for the joint run (three seeds only for N = 100 and 1,000).
2. **No intervals for test-set noise.** The experiment test rows come from 487 games per band, the same for every option. The ± values are seed spreads only, and there are no intervals for the differences between models.
3. **The learning rates were fixed, not tuned** (scratch 10⁻³, full fine-tune 10⁻⁴, BN + output layer 10⁻³). They were only checked at N = 100 with one seed. The rising training loss of BN + output layer at N ≥ 10,000 points to the learning rate.
4. **The largest runs stopped while still improving:** `base` (1 pass), `ftall_10000_s0`, `scratch_all_s0`, `ftall_all_s0` and the BN 10⁻⁴ check. Those numbers are not the models' limits.
5. **Early stopping uses a large validation set** (42,433 strong-player positions from 540 games). A real few-games setting would not have one, so the small-N results are optimistic for every option.
6. **The scratch baseline was not tuned for small N** (learning rate 10⁻³, no weight decay or other regularisation), so the "games saved" numbers are measured against a weak baseline.
7. **The experiment test rows are shard 000's only** (37,050 / 28,654 positions). The base model's numbers on all test rows differ slightly (3.263 vs 3.290 on 2000–2199).
8. **The rating plane outside its training range:** the base model never saw ratings of 1200 or more, and the scratch models never saw ratings below 2000. The results on other bands include this effect (up to 0.20 bits, section 6.8).
9. **Data window and players:** about 1–2 days of September 2026 Lichess blitz, from the start of the month. The test games share most of their players with the training games of the same band. A test set from a later month would be a stronger test.
10. **The BN option also updates BatchNorm's running statistics,** which `requires_grad` does not freeze. That may add to its forgetting (not tested). Three quarters of its 6,409 weights are the output layer, so it is not a test of scaling and shifting alone.
11. **The "games saved" numbers are interpolations** between 4 measured points (section 6.5).
12. **Only one baseline** (uniform over legal moves). Maia's reported numbers (over 50% top-1) were measured a bit differently: Maia skips the first 10 plies and moves made with under 30 seconds on the clock, also gives the network earlier positions of the game, and tests on a later month. So they are not directly comparable.
13. **The joint run** is one seed, its comparison with the earlier models is confounded by training length, neighbouring bands, shared players and time controls (section 6.9), and its sweep has a step of 100 rating points.
14. **The base model was trained with an earlier version of `train.py`** that was never committed.

---

## 9. Data files

All in `results/`, made by `python3 collect_results.py` (≈10 s) from `runs/` and `logs/`.

### 9.1 `all_runs.csv`: one row per run

| Column | Meaning |
|---|---|
| `run` | folder name in `runs/` |
| `kind` | `base`, `base_test`, `run1`, `n0`, `experiment`, `lr_check`, `joint` |
| `option` | `scratch`, `ftall`, `ftbn` (base and run1 are `scratch`) |
| `train_band`, `init`, `tune`, `games` (`all` = no limit), `seed`, `lr`, `max_epochs`, `eval_every`, `patience` | settings from `config.json` (several bands are joined with `+`) |
| `trainable_weights` | 459145 or 6409 |
| `train_positions` | positions actually trained on: re-created from the seed for the experiment runs, from the data for `base`, from the logs for `run1` and `joint_4bands` |
| `evals`, `last_step`, `passes_completed` | how far it ran |
| `best_step`, `best_is_last`, `early_stopped` | the tested checkpoint |
| `val_bits_<band>_start / _at_best / _last`, `val_top1_<band>_at_best` | validation numbers per band |
| `test_bits_<band>`, `test_top1_<band>`, `test_n_<band>`, `test_scope` | test results (`shard 000` or `all shards`) |
| `started`, `minutes` | timing |

### 9.2 `curves_val.csv`: every evaluation (long format)

`run, step, band, bits, top1`: one row per run × evaluation × band (1,364 rows).

### 9.3 `curves_train.csv`: the training loss

`run, step, train_bits`: the average of the last 50 steps (1,352 rows).

### 9.4 `baselines.csv`

`band, split, scope, positions, games, uniform_bits, random_top1, mean_legal_moves` (the table in section 2.3).

### 9.5 `bands_eval.csv` and `bands_eval_joint_4bands.csv` (made by `eval_bands.py`)

`model, band, split, positions, uniform_bits, random_top1, mean_rating, rating_mode, assumed_rating, bits, top1`.
- `rating_mode = true`: the real ratings, on the test and validation rows of shard 000 of each of 6 bands, for 5 models.
- `rating_mode = fixed`: the rating sweep (validation rows, every position given `assumed_rating`), for base, `ftall_all_s0` and `scratch_all_s0`.
- `bands_eval_joint_4bands.csv`: the same columns for `joint_4bands` on all 9 bands (800–2599), with the sweep from 800 to 2600. Made by step 4 of `run_joint.py`.

---

## 10. Figures and scripts

| Figure | What it shows | Sections |
|---|---|---|
| F1 `F1_main_result` | test bits on 2000–2199 against N for the three options, with the games scratch would need | 6.1, 6.5 |
| F2 `F2_top1` | the same in top-1 accuracy | 6.2 |
| F3 `F3_forgetting` | test bits on 1000–1199 after fine-tuning, and gain against forgetting | 6.3, 6.4 |
| F4 `F4_learning_dynamics` | validation and training loss over passes, seed 0, one panel per N | 6.6 |
| F5 `F5_seed_spread` | every seed at N = 100 and 1,000 | 6.1, 6.4 |
| F6 `F6_base_model` | the base model's validation on both bands during its one pass | 5 |
| F7 `F7_learning_rate_check` | the learning-rate check | 6.7 |
| F8 `F8_every_band` | five models on six bands | 6.8 |
| F9 `F9_rating_sweep` | the rating sweep of the single-band models | 6.8 |
| F10 `F10_joint_model` | the joint model: best rating vs real rating, its sweep, and every band | 6.9 |

Every figure is in `results/figures/` as PNG (200 dpi) and PDF. The training curves of each run are in `runs/<name>/training_curves.png`.

The scripts, in the order to rerun them after new runs:

```
python3 collect_results.py           # runs/ → results/*.csv
python3 summarize.py                 # results/*.csv → results/tables.md
python3 figures.py                   # results/*.csv → results/figures/F1…F10 (.png, .pdf)
python3 plot.py runs/<name> --once   # one run's training curves → runs/<name>/training_curves.png
```

---

## 11. Next steps

1. **Fairer comparisons for the joint model:**
   - a single-band model trained for the same number of steps (13,000), to separate the mix of bands from the training length;
   - the same joint run with 1600–1799 included, to measure what is lost by not training on a band;
   - bootstrap intervals over the test games for the differences between models;
   - more seeds, and training until the validation stops improving.
2. **Is the rating plane a skill input or a time-control input?** Train on a single time control, or add the time control as a separate input.
3. **The opponent's rating** as a second input, as Maia-2 does.
4. **A stricter few-games setup:** choose the checkpoint on 10–20% of the N games (or with a fixed schedule), and give scratch a small sweep of learning rates and weight decay at N = 100 and 1,000.
5. **BN + output layer at large N with a learning rate of 10⁻⁴** (or a decaying schedule), with the BatchNorm running statistics frozen; and a true BN-only run (1,664 weights).
6. **Longer runs where the caps were hit:** a second pass for `base`, and more passes for `ftall_10000`, `ftall_all` and `scratch_all`.
7. **Seeds 1–2 for N = 10,000**, to put error bars on the large-N points.
8. **Better reference points:** a move-frequency baseline, the public Maia-1 weights on the same test rows, and a test set from a later month.
9. **A regularised fine-tune** (an L2 pull towards the base weights, as in Kukleva et al. 2021) to reduce forgetting at the same gain.
10. **Faster training:** build the legal-move mask on the GPU in one operation (the joint run was CPU-bound).
